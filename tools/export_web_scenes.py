#!/usr/bin/env python3
"""Export compact JSON scene data for the KaRMA three.js workspace viewer.

For each hand's paper run (results/16_hand_batch/robot_<name>/current.pkl) this
emits the seed-grasp geometry the web viewer draws -- finger capsules, the test
sphere, the reachable voxel cloud (world centre + rotation coverage per voxel),
the two seed contact normals, and a framed camera. All geometry is procedural
(capsules + boxes + a sphere), matching what run_viser_app.py shows, so there are
no third-party meshes and no mesh-license concerns.

Output: one scenes.json keyed by hand id (all 16 hands, well under 1 MB), which
the viewer loads once so switching hands is instant.

Run from the karma-hand-metric repo root in its conda env::

    PYTHONPATH=. conda run -n karma-hand-metric --cwd /path/to/karma-hand-metric \
        python /path/to/karmasite/tools/export_web_scenes.py \
        --out /path/to/karmasite/static/scenes/scenes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from karma.config import load_config
from karma.robot import load_robot, update_kinematics, capsule_endpoints_world
from karma.orientation import healpix_npix_lines
from karma.contacts import contact_kinematics
from karma.storage import load_result

VIEW_DIR = np.array([0.5, -1.0, 0.35])
COVERAGE_MAX = 0.25  # matches viser_scene._rotation_coverage_to_color default


def r3(x):
    return [round(float(v), 5) for v in np.asarray(x, dtype=float).reshape(-1)]


def frame_camera(centre, radius):
    view = VIEW_DIR / np.linalg.norm(VIEW_DIR)
    dist = max(float(radius), 0.03) * 2.1
    pos = np.asarray(centre, dtype=float) + view * dist
    return {"position": r3(pos), "target": r3(centre), "up": [0.0, 0.0, 1.0]}


def build(cfg, ctx, result):
    seed_q = result.seed_q
    seed_centre = np.asarray(result.seed_centre, dtype=float).reshape(3)
    R_seed = np.asarray(getattr(result, "seed_frame", np.eye(3)), dtype=float)
    h = float(cfg.voxel_size_m)
    total_bins = healpix_npix_lines(cfg.healpix_nside)

    update_kinematics(ctx, seed_q)

    # Finger capsules (cap-centre endpoints a,b in world; contact links flagged).
    capsules = []
    pts = []
    for name in ctx.link_capsules:
        a, b = capsule_endpoints_world(ctx, name)
        pts.append(a); pts.append(b)
        capsules.append(r3(a) + r3(b) + [1 if name in cfg.all_contact_links else 0])

    # Reachable voxels: world centre + rotation coverage.
    voxels = []
    for v, bins in result.voxel_ori_bins.items():
        c = seed_centre + R_seed @ (np.asarray(v, dtype=float) * h)
        pts.append(c + 0.5 * h); pts.append(c - 0.5 * h)
        voxels.append(r3(c) + [round(len(bins) / total_bins, 4)])

    # Seed contact normals (thumb, index).
    normals = []
    for geom in result.seed_contact_pair:
        ck = contact_kinematics(ctx, seed_q, seed_centre, geom,
                                cfg.sphere_radius_m, cfg.link_radius_m)
        normals.append(r3(ck.sphere_point_world) + r3(ck.normal_world))

    pts.append(seed_centre + cfg.sphere_radius_m)
    pts.append(seed_centre - cfg.sphere_radius_m)
    pts = np.asarray(pts, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))

    dof = len(cfg.thumb_joint_names) + len(cfg.index_joint_names)
    return {
        "link_radius": round(float(cfg.link_radius_m), 5),
        "voxel_size": round(h, 5),
        "coverage_max": COVERAGE_MAX,
        "sphere": {"c": r3(seed_centre), "r": round(float(cfg.sphere_radius_m), 5)},
        "capsules": capsules,
        "voxels": voxels,
        "normals": normals,
        "camera": frame_camera(centre, radius),
        "dof": dof,
        "l_ref_mm": round(float(result.l_ref_m) * 1000.0, 1),
        "karma_t": round(float(result.translational_score), 6),
        "karma_r": round(float(result.global_rotational_score), 4),
        "n_voxels": int(result.n_voxels_reached),
    }


LABELS = {
    "leap": "LEAP", "allegro": "Allegro", "dclaw": "D'Claw", "sharpa": "Sharpa",
    "wuji": "Wuji", "shadowhand": "Shadow", "ARMS_skel": "ARMS", "dex5": "Dex5",
    "DLR": "DLR", "dex3": "Dex3", "xhand1": "xHand1", "orcahand": "OrcaHand",
    "xhandlite": "xHandLite", "inspire": "Inspire", "svh": "SVH",
    "ability_hand_right": "Ability",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-config", default="karma_config.yaml")
    ap.add_argument("--results-dir", default="results/16_hand_batch")
    ap.add_argument("--robots-dir", default="robots")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    robots_dir = Path(args.robots_dir)
    scenes = {}
    for run_dir in sorted(results_dir.glob("robot_*")):
        pkl = run_dir / "current.pkl"
        if not pkl.exists():
            continue
        name = run_dir.name[len("robot_"):]
        cfg_path = robots_dir / f"robot_{name}.yaml"
        if not cfg_path.exists():
            print(f"  ! skip {name}: no config")
            continue
        cfg = load_config(args.metric_config, str(cfg_path))
        ctx = load_robot(cfg.urdf_path, cfg)
        result, _ = load_result(pkl)
        s = build(cfg, ctx, result)
        s["label"] = LABELS.get(name, name)
        scenes[name] = s
        print(f"  {name:20s} {s['n_voxels']:4d} voxels  {len(s['capsules'])} links")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenes, separators=(",", ":")))
    kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({len(scenes)} hands, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
