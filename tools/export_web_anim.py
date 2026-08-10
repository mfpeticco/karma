#!/usr/bin/env python3
"""Export a pre-baked motion tour per hand for the KaRMA "hand in motion" viewer.

Each hand's paper run (results/16_hand_batch/robot_<name>/current.pkl) stores a
tree of reached states: State = (voxel, healpix_bin), every node carrying the
converged joint config q, sphere centre, and sphere orientation R_sphere, plus a
parent pointer back toward the seed grasp. We replay that tree as a two-act tour:

  Act 1 (reorient): at the seed voxel, roll the sphere out to a few well-spread
                    reachable orientations and back -- KaRMA-R made visible.
  Act 2 (travel):   from the seed, roll the sphere out to far voxels in several
                    directions and back -- KaRMA-T made visible.

Both acts start and end at the seed grasp, so the loop is seamless.

The stored nodes are settled configs, one per (voxel, orientation); adjacent
nodes can differ by tens of degrees, so we interpolate in JOINT space and run
forward kinematics per frame (exactly what run_viser_app._play_path does) rather
than lerping capsule endpoints -- that keeps every link rigid. The browser then
just plays flat, ready-to-draw frames; no kinematics in JS.

Output: one static/anim/<hand>.json per hand, lazy-loaded when that hand is
picked. Read-only w.r.t. the code repo -- reads pkls, writes only into the site.

Run from the karma-hand-metric repo root in its conda env::

    PYTHONPATH=. conda run -n karma-hand-metric --cwd /path/to/karma-hand-metric \
        python /path/to/karmasite/tools/export_web_anim.py \
        --out-dir /path/to/karmasite/static/anim
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from karma.config import load_config
from karma.robot import load_robot, update_kinematics, capsule_endpoints_world
from karma.orientation import healpix_npix_lines
from karma.storage import load_result

VIEW_DIR = np.array([0.5, -1.0, 0.35])
COVERAGE_MAX = 0.25  # matches the static scene export / viser color ramp

# Interpolation budget: keep per-frame motion below these so playback is smooth
# regardless of how far apart two settled configs are.
DEG_PER_FRAME_JOINT = 8.0
DEG_PER_FRAME_SPHERE = 13.0
MM_PER_FRAME_CENTRE = 3.0
MAX_INTERP = 12

N_REORIENT_KEYS = 8       # reachable orientations sampled at the seed voxel
TRAVEL_ONEWAY_MAX = 36    # max baked poses rolled outward along the workspace's long
                          # axis before turning back; the out-and-back (~2x) fits the
                          # ~5s playback slot and returns to the seed (seamless loop)

LABELS = {
    "leap": "LEAP", "allegro": "Allegro", "dclaw": "D'Claw", "sharpa": "Sharpa",
    "wuji": "Wuji", "shadowhand": "Shadow", "ARMS_skel": "ARMS", "dex5": "Dex5",
    "DLR": "DLR", "dex3": "Dex3", "xhand1": "xHand1", "orcahand": "OrcaHand",
    "xhandlite": "xHandLite", "inspire": "Inspire", "svh": "SVH",
    "ability_hand_right": "Ability",
}


def r4(x):
    return [round(float(v), 4) for v in np.asarray(x, dtype=float).reshape(-1)]


def geodesic_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def frame_camera(centre, radius):
    view = VIEW_DIR / np.linalg.norm(VIEW_DIR)
    dist = max(float(radius), 0.03) * 2.1
    pos = np.asarray(centre, dtype=float) + view * dist
    return {"position": r3(pos), "target": r3(centre), "up": [0.0, 0.0, 1.0]}


def r3(x):
    return [round(float(v), 5) for v in np.asarray(x, dtype=float).reshape(-1)]


def fk_frame(ctx, cap_names, q, centre, R):
    """One playback frame: flat capsule endpoints + sphere centre + quaternion."""
    update_kinematics(ctx, q)
    flat = []
    for name in cap_names:
        a, b = capsule_endpoints_world(ctx, name)
        flat += r4(a) + r4(b)
    quat = Rot.from_matrix(R).as_quat()  # [x, y, z, w]
    return flat + r4(centre) + r4(quat)


_AXIS_FLIPS = [np.eye(3), np.diag([1.0, -1.0, -1.0]),
               np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])]


def deflip_sequence(Rs):
    """Return an equivalent orientation sequence with the non-physical antipodal
    jumps removed. The pinch axis is ±-identified and the sphere is a featureless
    ball, so post-multiplying R by a 180 deg body-axis flip gives an equivalent
    display orientation; at each step we pick the representative closest to the
    previous, turning ~180 deg tumbles into the small rolling rotation of actual
    translation."""
    out = [np.asarray(Rs[0], dtype=float)]
    for R in Rs[1:]:
        R = np.asarray(R, dtype=float)
        best, best_d = None, 1e9
        for f in _AXIS_FLIPS:
            RR = R @ f
            d = geodesic_deg(out[-1], RR)
            if d < best_d:
                best_d, best = d, RR
        out.append(best)
    return out


def interp_segment(ctx, cap_names, qa, ca, Ra, qb, cb, Rb):
    """Frames for a -> b (exclusive of a), joint-space lerp + FK, R slerped."""
    jd = float(np.degrees(np.max(np.abs(qb - qa)))) if len(qa) else 0.0
    sd = geodesic_deg(Ra, Rb)
    cd = float(np.linalg.norm(cb - ca)) * 1000.0
    n = int(np.clip(np.ceil(max(jd / DEG_PER_FRAME_JOINT,
                                sd / DEG_PER_FRAME_SPHERE,
                                cd / MM_PER_FRAME_CENTRE)), 1, MAX_INTERP))
    slerp = Slerp([0.0, 1.0], Rot.from_matrix(np.stack([Ra, Rb])))
    ts = np.linspace(0.0, 1.0, n + 1)[1:]
    Rs = slerp(ts).as_matrix()
    out = []
    for i, t in enumerate(ts):
        q = (1.0 - t) * qa + t * qb
        c = (1.0 - t) * ca + t * cb
        out.append(fk_frame(ctx, cap_names, q, c, Rs[i]))
    return out


def path_frames(ctx, cap_names, states, sn, R_override=None):
    """Interpolated frames along a keyframe-state list (includes the first).

    R_override (one orientation per state) replaces the stored R_sphere; used by
    the travel act to play a de-flipped, smoothly-rolling orientation."""
    Rs = R_override if R_override is not None else [sn[s].R_sphere for s in states]
    frames = [fk_frame(ctx, cap_names, sn[states[0]].q, sn[states[0]].centre_world, Rs[0])]
    for k in range(len(states) - 1):
        a, b = sn[states[k]], sn[states[k + 1]]
        frames += interp_segment(ctx, cap_names, a.q, a.centre_world, Rs[k],
                                 b.q, b.centre_world, Rs[k + 1])
    return frames


def pingpong(frames):
    """Out then back to the start (so subpaths chain seamlessly at home)."""
    if len(frames) <= 1:
        return frames
    return frames + frames[-2::-1]


def backtrack(sn, st):
    p, cur = [], st
    while cur is not None:
        p.append(cur)
        cur = sn[cur].parent
    return p[::-1]  # seed-root -> st


def spread_pick(cand_states, sn, k, seed=None):
    """Greedy max-min pick of k orientation states spread apart in R-space.

    Starts from `seed` if given (kept in the result), else from the state
    farthest from the mean; each further pick maximizes the min geodesic to the
    already-chosen set."""
    if not cand_states:
        return []
    chosen = [seed] if seed is not None else [cand_states[0]]
    while len(chosen) < k and len(chosen) < len(cand_states):
        best_s, best_mind = None, -1.0
        for s in cand_states:
            if s in chosen:
                continue
            mind = min(geodesic_deg(sn[s].R_sphere, sn[c].R_sphere) for c in chosen)
            if mind > best_mind:
                best_mind, best_s = mind, s
        if best_s is None:
            break
        chosen.append(best_s)
    return chosen


def order_nn(states, sn, home):
    """Order `states` into a loop home -> ... -> home, greedily hopping to the
    nearest unvisited orientation so each morph between real reachable grasps is
    as small as possible."""
    remaining = [s for s in states if s != home]
    order, cur = [home], home
    while remaining:
        nxt = min(remaining, key=lambda s: geodesic_deg(sn[cur].R_sphere, sn[s].R_sphere))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    order.append(home)  # close the loop back to the upright home grasp
    return order


def build_reorient_act(ctx, cap_names, sn, per_voxel):
    """Reorient the sphere in place at the seed voxel.

    The reachable orientations at a voxel are stored as independent settled
    grasps (no rolling substeps between them survive in the result), so we sample
    a spread of them, order them into a nearest-neighbour loop starting/ending at
    the upright home grasp, and morph between consecutive real grasps (joint-space
    lerp + FK). The sphere centre is constant across these states, so the sphere
    reorients strictly in place."""
    if (0, 0, 0) in per_voxel and len(per_voxel[(0, 0, 0)]) >= 4:
        vr = (0, 0, 0)
    else:  # seed voxel too poor: use the most-reorientable voxel near the centre
        cand = sorted(per_voxel, key=lambda v: (-len(per_voxel[v]), abs(v[0]) + abs(v[1]) + abs(v[2])))
        vr = cand[0]
    states_r = per_voxel[vr]
    if len(states_r) < 2:
        return vr, []
    I3 = np.eye(3)
    # home = the reachable grasp closest to upright (identity sphere orientation)
    home = min(states_r, key=lambda s: geodesic_deg(I3, sn[s].R_sphere))
    picks = spread_pick(states_r, sn, N_REORIENT_KEYS, seed=home)
    order = order_nn(picks, sn, home)
    if len(order) < 2:
        return vr, []
    return vr, path_frames(ctx, cap_names, order, sn)


def build_travel_act(ctx, cap_names, sn, per_voxel):
    """Roll the sphere out along the workspace's LONG axis and back.

    The reachable cloud is often a narrow manifold; rolling along its principal
    (max-variance) direction shows the long reach, not the short cross-section. We
    head for the voxel farthest from the seed along that axis, then cap the outward
    poses so the out-and-back fits the playback slot and returns to the seed."""
    voxels = [v for v in per_voxel if v != (0, 0, 0)]
    if not voxels:
        return []
    P = np.array([np.asarray(sn[per_voxel[v][0]].centre_world, dtype=float) for v in voxels])
    seed_c = (np.asarray(sn[per_voxel[(0, 0, 0)][0]].centre_world, dtype=float)
              if (0, 0, 0) in per_voxel else P.mean(axis=0))
    # principal (long) axis of the reachable cloud
    if len(P) >= 2:
        u = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)[2][0]
    else:
        u = np.array([1.0, 0.0, 0.0])
    proj = (P - seed_c) @ u
    tip_v = voxels[int(np.argmax(np.abs(proj)))]  # farthest along the long axis, either end

    entry = None
    for s in per_voxel[tip_v]:
        par = sn[s].parent
        if par is None or sn[par].voxel != tip_v:
            entry = s
            break
    if entry is None:
        entry = per_voxel[tip_v][0]
    states = backtrack(sn, entry)  # seed -> tip
    if len(states) < 2:
        return []
    Rs = deflip_sequence([sn[s].R_sphere for s in states])  # remove antipodal tumbles
    one_way = path_frames(ctx, cap_names, states, sn, R_override=Rs)[:TRAVEL_ONEWAY_MAX]
    return pingpong(one_way)


def build(name, cfg, ctx, result):
    sn = result.state_nodes
    per_voxel = defaultdict(list)
    for st, node in sn.items():
        per_voxel[node.voxel].append(st)

    cap_names = list(ctx.link_capsules)
    contact_flags = [1 if nm in cfg.all_contact_links else 0 for nm in cap_names]
    total_bins = healpix_npix_lines(cfg.healpix_nside)
    h = float(cfg.voxel_size_m)

    # Static voxel cloud (same world frame as the animated sphere centres).
    voxels = []
    for v, bins in result.voxel_ori_bins.items():
        c = np.asarray(sn[per_voxel[v][0]].centre_world, dtype=float)
        voxels.append(r4(c) + [round(len(bins) / total_bins, 4), int(v[0]), int(v[1]), int(v[2])])

    reorient_voxel, reorient_frames = build_reorient_act(ctx, cap_names, sn, per_voxel)
    travel_frames = build_travel_act(ctx, cap_names, sn, per_voxel)

    # Every act gets at least one frame (the seed grasp) so a hand that can't reorient
    # or barely translates still holds a still pose for that phase, and the viewer can
    # cycle Rotation -> Translation for every hand (you just see no motion when there's
    # none). fk seed frame uses the seed grasp with an upright sphere.
    seed_frame = fk_frame(ctx, cap_names, result.seed_q,
                          np.asarray(result.seed_centre, dtype=float).reshape(3), np.eye(3))
    if not reorient_frames:
        reorient_frames = [seed_frame]
    if not travel_frames:
        travel_frames = [seed_frame]

    # Camera framing fits the HAND and the sphere's trajectory, NOT the voxel cloud.
    # Cloud extent varies enormously between hands, so framing to it made small
    # workspaces ultra-zoomed and large ones tiny; the hand + short travel both scale
    # with hand size, so this gives every hand a consistent view (the cloud may spill
    # past the frame -- it's backdrop, and the sphere always stays in view).
    r = float(cfg.sphere_radius_m)
    fa = np.asarray(reorient_frames + travel_frames, dtype=float)
    ncf = len(cap_names) * 6
    cap_pts = fa[:, :ncf].reshape(-1, 3)
    ctrs = fa[:, ncf:ncf + 3]
    fpts = np.vstack([cap_pts, ctrs + r, ctrs - r])
    lo, hi = fpts.min(axis=0), fpts.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))

    dof = len(cfg.thumb_joint_names) + len(cfg.index_joint_names)
    return {
        "label": LABELS.get(name, name),
        "link_radius": round(float(cfg.link_radius_m), 5),
        "voxel_size": round(h, 5),
        "sphere_r": round(float(cfg.sphere_radius_m), 5),
        "coverage_max": COVERAGE_MAX,
        "n_caps": len(cap_names),
        "contact_flags": contact_flags,
        "voxels": voxels,
        "camera": frame_camera(centre, radius),
        "dof": dof,
        "karma_t": round(float(result.translational_score), 6),
        "karma_r": round(float(result.global_rotational_score), 4),
        "n_voxels": int(result.n_voxels_reached),
        "acts": [
            {"name": "reorient", "voxel": list(reorient_voxel), "frames": reorient_frames},
            {"name": "travel", "frames": travel_frames},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-config", default="karma_config.yaml")
    ap.add_argument("--results-dir", default="results/16_hand_batch")
    ap.add_argument("--robots-dir", default="robots")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="", help="comma-separated hand ids to limit to")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    robots_dir = Path(args.robots_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    index = []
    for run_dir in sorted(results_dir.glob("robot_*")):
        pkl = run_dir / "current.pkl"
        if not pkl.exists():
            continue
        name = run_dir.name[len("robot_"):]
        if only and name not in only:
            continue
        cfg_path = robots_dir / f"robot_{name}.yaml"
        if not cfg_path.exists():
            print(f"  ! skip {name}: no config")
            continue
        cfg = load_config(args.metric_config, str(cfg_path))
        ctx = load_robot(cfg.urdf_path, cfg)
        result, _ = load_result(pkl)
        data = build(name, cfg, ctx, result)
        out = out_dir / f"{name}.json"
        out.write_text(json.dumps(data, separators=(",", ":")))
        kb = out.stat().st_size / 1024
        nre = len(data["acts"][0]["frames"])
        ntr = len(data["acts"][1]["frames"])
        index.append({"id": name, "label": data["label"], "karma_t": data["karma_t"],
                      "karma_r": data["karma_r"], "n_voxels": data["n_voxels"]})
        print(f"  {name:20s} {data['n_voxels']:4d} vox  reorient={nre:4d}f travel={ntr:4d}f  {kb:5.0f} KB")

    # sort index by KaRMA-T desc for a sensible default picker order
    index.sort(key=lambda d: -d["karma_t"])
    (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"Wrote {len(index)} hands + index.json to {out_dir}")


if __name__ == "__main__":
    main()
