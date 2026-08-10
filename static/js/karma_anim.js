/*
 * KaRMA "hand in motion" viewer (three.js).
 *
 * Plays a pre-baked two-act tour per hand (static/anim/<hand>.json), replayed
 * from the paper run's reached-state tree by tools/export_web_anim.py:
 *   Act 1 (reorient): the sphere reorients in place at the seed grasp -- KaRMA-R.
 *   Act 2 (travel):   the sphere rolls out across the workspace and back -- KaRMA-T.
 * Each frame is flat, ready-to-draw data: capsule axis endpoints (a,b per link),
 * the sphere centre, and the sphere orientation quaternion. Forward kinematics is
 * baked in Python, so the browser only sets transforms -- no kinematics here.
 *
 * Every link is rigid, so its two capsule endpoints keep a constant separation
 * across frames; we build each capsule once (fixed length) and, per frame, only
 * move and rotate it. The voxel cloud is drawn dim as a backdrop and the cell the
 * sphere currently occupies is outlined, so you can watch it visit the workspace.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

// Playback: a constant number of baked poses per second, the same for both acts, so
// the motion looks equally natural throughout (travel is kept short by the export's
// trajectory length, not by speeding it up). Each act runs for a fixed wall-clock
// slot (ACT_SECONDS); if its motion is shorter it holds the last pose for the rest
// of the slot, so every hand cycles Rotation -> Translation on the same clock and a
// hand that can't reorient/translate simply holds still during that phase. Rendering
// runs at the display's rAF rate (~60 fps), interpolating between poses, so it's smooth.
const SPEED = 15;              // baked poses per second
const ACT_SECONDS = 5;        // wall-clock slot per act (rotation / translation)
const COVERAGE_MAX_FALLBACK = 0.25;

function coverageColor(cov, maxCov) {
  const t = Math.min(1, Math.max(0, cov / maxCov));
  let r, g;
  const b = 50 / 255;
  if (t < 0.5) { r = 1.0; g = t * 2; }
  else { r = 1 - (t - 0.5) * 2; g = 1.0; }
  return new THREE.Color(r, g, b);
}

export async function loadAnimIndex(url) {
  return fetch(url).then((r) => r.json());
}

export async function createAnimViewer(opts) {
  const mount = opts.mount;
  const baseUrl = opts.baseUrl || "./static/anim/";

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  mount.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 100);
  camera.up.set(0, 0, 1);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;

  // Zoom only while Cmd/Ctrl is held (plain wheel scrolls the page).
  mount.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.metaKey) return;
    e.stopImmediatePropagation();
  }, { capture: true });

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const keyL = new THREE.DirectionalLight(0xffffff, 2.2); keyL.position.set(0.6, -1.0, 1.2); scene.add(keyL);
  const fillL = new THREE.DirectionalLight(0xffffff, 0.7); fillL.position.set(-0.8, 0.4, 0.5); scene.add(fillL);

  const root = new THREE.Group();
  scene.add(root);

  const capsuleMat = {
    contact: new THREE.MeshStandardMaterial({ color: 0x6495ed, roughness: 0.55, metalness: 0.0 }),
    idle: new THREE.MeshStandardMaterial({ color: 0xb4b4b4, roughness: 0.7, metalness: 0.0 }),
  };
  const sphereMat = new THREE.MeshStandardMaterial({ color: 0xff5050, transparent: true, opacity: 0.5, roughness: 0.5 });
  const yAxis = new THREE.Vector3(0, 1, 0);

  const state = {
    data: null, caps: [], sphere: null, triad: null, voxelMesh: null, highlight: null,
    voxelCentres: [], actIdx: 0, pos: 0, actElapsed: 0, playing: true,
  };

  // reusable temporaries
  const va = new THREE.Vector3(), vb = new THREE.Vector3(), vmid = new THREE.Vector3(), vdir = new THREE.Vector3();
  const a0 = new THREE.Vector3(), a1 = new THREE.Vector3(), b0 = new THREE.Vector3(), b1 = new THREE.Vector3();
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qtmp = new THREE.Quaternion();

  // render/playback loop state (declared before resize(), which runs immediately)
  let dirty = true, inView = false, rafId = null, last = 0;

  function clearRoot() {
    for (let i = root.children.length - 1; i >= 0; i--) {
      const o = root.children[i];
      root.remove(o);
      o.traverse?.((c) => { if (c.geometry) c.geometry.dispose(); });
      if (o.geometry) o.geometry.dispose();
    }
    state.caps = []; state.sphere = null; state.triad = null; state.voxelMesh = null;
    state.highlight = null; state.voxelCentres = [];
  }

  function build(data) {
    clearRoot();
    state.data = data;
    const nc = data.n_caps;
    const stride = nc * 6;
    const f0 = data.acts.find((a) => a.frames.length)?.frames[0];
    const maxCov = data.coverage_max || COVERAGE_MAX_FALLBACK;

    // --- voxel cloud backdrop ---
    // Opacity scales inversely with cloud density: a dense cloud stacks many voxels
    // along each view ray (their alpha accumulates), so each must be faint; a sparse
    // low-DOF cloud has little overlap, so each voxel needs to be much more opaque to
    // be visible at all. ~0.65 / cbrt(N) matches the dense case (N~800 -> 0.07) and
    // lifts sparse clouds (N~4 -> ~0.4), clamped to a sane range.
    const nv = data.voxels.length;
    const vopacity = Math.min(0.30, Math.max(0.06, 0.65 / Math.cbrt(Math.max(nv, 1))));
    const vgeo = new THREE.BoxGeometry(data.voxel_size, data.voxel_size, data.voxel_size);
    const vmat = new THREE.MeshBasicMaterial({ transparent: true, opacity: vopacity, depthWrite: false });
    const vmesh = new THREE.InstancedMesh(vgeo, vmat, nv);
    const m = new THREE.Matrix4();
    for (let i = 0; i < nv; i++) {
      const v = data.voxels[i];
      m.makeTranslation(v[0], v[1], v[2]);
      vmesh.setMatrixAt(i, m);
      vmesh.setColorAt(i, coverageColor(v[3], maxCov));
      state.voxelCentres.push([v[0], v[1], v[2]]);
    }
    vmesh.instanceMatrix.needsUpdate = true;
    if (vmesh.instanceColor) vmesh.instanceColor.needsUpdate = true;
    root.add(vmesh); state.voxelMesh = vmesh;

    // --- current-cell outline ---
    const hgeo = new THREE.BoxGeometry(data.voxel_size * 1.02, data.voxel_size * 1.02, data.voxel_size * 1.02);
    const hedges = new THREE.LineSegments(
      new THREE.EdgesGeometry(hgeo),
      new THREE.LineBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.55 }));
    hgeo.dispose();
    root.add(hedges); state.highlight = hedges;

    // --- capsules (fixed length per rigid link) ---
    const r = data.link_radius;
    for (let i = 0; i < nc; i++) {
      va.set(f0[i * 6], f0[i * 6 + 1], f0[i * 6 + 2]);
      vb.set(f0[i * 6 + 3], f0[i * 6 + 4], f0[i * 6 + 5]);
      const len = Math.max(vb.distanceTo(va), 1e-6);
      const geo = new THREE.CapsuleGeometry(r, len, 6, 14);
      const mesh = new THREE.Mesh(geo, data.contact_flags[i] ? capsuleMat.contact : capsuleMat.idle);
      root.add(mesh);
      state.caps.push(mesh);
    }

    // --- sphere + rotating body triad ---
    const sph = new THREE.Mesh(new THREE.SphereGeometry(data.sphere_r, 24, 16), sphereMat);
    root.add(sph); state.sphere = sph;

    const triad = new THREE.Group();
    const axcol = [0xff0000, 0x00cc00, 0x0000ff];
    const al = data.sphere_r * 2.8;         // extends ~0.4 r beyond the sphere so it reads
    const arad = data.sphere_r * 0.1;       // solid cylinders (real thickness, unlike 1px lines)
    for (let k = 0; k < 3; k++) {
      const dir = new THREE.Vector3(); dir.setComponent(k, 1);
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(arad, arad, al, 12),
        new THREE.MeshBasicMaterial({ color: axcol[k] }));
      mesh.quaternion.setFromUnitVectors(yAxis, dir);   // cylinder axis is +Y -> point along +k
      mesh.position.copy(dir).multiplyScalar(al * 0.5);
      triad.add(mesh);
    }
    root.add(triad); state.triad = triad;

    // --- camera framing ---
    camera.position.set(...data.camera.position);
    controls.target.set(...data.camera.target);
    controls.update();

    // always start on the first act (rotation); every act now has >=1 frame
    state.actIdx = 0;
    state.pos = 0;
    state.actElapsed = 0;
    applyPose(state.actIdx, 0);
    dirty = true;
  }

  function nearestVoxel(cx, cy, cz) {
    let best = -1, bd = Infinity;
    const C = state.voxelCentres;
    for (let i = 0; i < C.length; i++) {
      const dx = C[i][0] - cx, dy = C[i][1] - cy, dz = C[i][2] - cz;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  // Draw the pose at fractional position `pos` within act `actIdx`, interpolating
  // between the two nearest baked poses (linear on capsule endpoints, slerp on the
  // sphere). Baked poses are dense and close together, so this reads as smooth,
  // near-rigid motion at the display refresh rate regardless of the act's pace.
  function applyPose(actIdx, pos) {
    const data = state.data; if (!data) return;
    const act = data.acts[actIdx];
    if (!act || !act.frames.length) return;
    const frames = act.frames;
    let i0 = Math.floor(pos);
    if (i0 < 0) i0 = 0;
    if (i0 > frames.length - 1) i0 = frames.length - 1;
    const i1 = Math.min(i0 + 1, frames.length - 1);
    let t = pos - i0; if (t < 0) t = 0; else if (t > 1) t = 1;
    const f0 = frames[i0], f1 = frames[i1];
    const nc = data.n_caps;
    for (let i = 0; i < nc; i++) {
      const o = i * 6;
      a0.set(f0[o], f0[o + 1], f0[o + 2]); a1.set(f1[o], f1[o + 1], f1[o + 2]);
      b0.set(f0[o + 3], f0[o + 4], f0[o + 5]); b1.set(f1[o + 3], f1[o + 4], f1[o + 5]);
      va.copy(a0).lerp(a1, t); vb.copy(b0).lerp(b1, t);
      vmid.addVectors(va, vb).multiplyScalar(0.5);
      vdir.subVectors(vb, va);
      const cap = state.caps[i];
      cap.position.copy(vmid);
      if (vdir.lengthSq() > 1e-12) cap.quaternion.setFromUnitVectors(yAxis, vdir.normalize());
    }
    const o = nc * 6;
    const cx = f0[o] + (f1[o] - f0[o]) * t;
    const cy = f0[o + 1] + (f1[o + 1] - f0[o + 1]) * t;
    const cz = f0[o + 2] + (f1[o + 2] - f0[o + 2]) * t;
    state.sphere.position.set(cx, cy, cz);
    state.triad.position.set(cx, cy, cz);
    qa.set(f0[o + 3], f0[o + 4], f0[o + 5], f0[o + 6]);
    qb.set(f1[o + 3], f1[o + 4], f1[o + 5], f1[o + 6]);
    qtmp.copy(qa).slerp(qb, t);
    state.triad.quaternion.copy(qtmp);
    const nv = nearestVoxel(cx, cy, cz);
    if (nv >= 0) state.highlight.position.set(...state.voxelCentres[nv]);
    dirty = true;
  }

  // ---- resize ----
  function resize() {
    const w = mount.clientWidth || 1, h = mount.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    dirty = true;
  }
  const ro = new ResizeObserver(resize); ro.observe(mount); resize();

  // ---- loop: render every rAF tick; advance a fractional position at the act's pace ----
  controls.addEventListener("change", () => { dirty = true; });
  const io = new IntersectionObserver((entries) => {
    inView = entries[0].isIntersecting;
    if (inView) { last = 0; if (!rafId) rafId = requestAnimationFrame(loop); }
    else if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }, { threshold: 0.01 });
  io.observe(mount);

  function stepPlayback(dt) {
    if (!(state.playing && state.data)) return;
    const acts = state.data.acts;
    const act = acts[state.actIdx];
    if (!act || !act.frames.length) return;
    const end = act.frames.length - 1;
    // advance through this act's poses; once exhausted, LOOP back to the start to
    // keep cycling the motion for the rest of the slot (each act starts and ends at
    // the seed grasp, so the wrap is seamless). A 1-frame act (no motion) just holds.
    state.pos += SPEED * dt;
    if (end >= 1) { while (state.pos > end) state.pos -= end; }
    else state.pos = 0;
    state.actElapsed += dt;
    applyPose(state.actIdx, state.pos);
    // each act gets a fixed wall-clock slot, then hand off to the next (cycling all)
    if (state.actElapsed >= ACT_SECONDS) {
      state.actIdx = (state.actIdx + 1) % acts.length;
      state.pos = 0;
      state.actElapsed = 0;
      if (opts.onAct) opts.onAct(acts[state.actIdx].name);
    }
  }

  function loop(ts) {
    rafId = requestAnimationFrame(loop);
    if (!last) last = ts;
    let dt = (ts - last) / 1000; last = ts;
    if (dt > 0.1) dt = 0.1;   // clamp after a tab-hidden gap so it doesn't jump
    controls.update();
    stepPlayback(dt);
    if (dirty) { renderer.render(scene, camera); dirty = false; }
  }

  async function loadHand(id) {
    const data = await fetch(`${baseUrl}${id}.json`).then((r) => r.json());
    build(data);
    if (opts.onAct) opts.onAct(data.acts[state.actIdx]?.name || "");
    return data;
  }

  return {
    loadHand,
    setPlaying(v) { state.playing = v; if (v) last = 0; dirty = true; },
    isPlaying() { return state.playing; },
    currentAct() { return state.data?.acts[state.actIdx]?.name || ""; },
    meta() {
      const d = state.data; if (!d) return null;
      return { label: d.label, dof: d.dof, karma_t: d.karma_t, karma_r: d.karma_r, n_voxels: d.n_voxels };
    },
  };
}
