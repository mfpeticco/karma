/*
 * KaRMA interactive workspace explorer (three.js).
 *
 * Renders, from compact per-hand JSON (static/scenes/scenes.json), the same
 * seed-grasp scene the paper figures show: finger capsules (contact links blue,
 * rest gray), the translucent test sphere with its body-frame triad, the two
 * seed contact normals, and the reachable voxel cloud colored by rotation
 * coverage on a fixed red->green scale (matching viser_scene._rotation_coverage_
 * to_color, coverage_max = 0.25). All geometry is procedural, so there are no
 * third-party meshes. Switching hands rebuilds the scene from already-loaded
 * data -- instant, no network, no reload.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

function coverageColor(cov, maxCov) {
  const t = Math.min(1, Math.max(0, cov / maxCov));
  let r, g;
  const b = 50 / 255;
  if (t < 0.5) { r = 1.0; g = t * 2; }
  else { r = 1 - (t - 0.5) * 2; g = 1.0; }
  return new THREE.Color(r, g, b);
}

export async function loadScenes(url) {
  return fetch(url).then((r) => r.json());
}

export async function createViewer(opts) {
  const mount = opts.mount;
  // Accept preloaded scene data (shared between panels) or fetch it.
  const scenes = opts.scenes || await loadScenes(opts.scenesUrl);

  // ---- renderer / camera / controls -------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(window.devicePixelRatio || 1);   // full device resolution (crisp on retina)
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

  // Render on demand, and run the loop only while the panel is on-screen and
  // something is actually happening. Two always-on WebGL loops (plus a
  // per-frame controls.update() on each) otherwise make the whole page feel
  // laggy during scroll, even when nothing is being interacted with.
  let dirty = true;
  let inView = false;
  let rafId = null;
  let active = false, endTimer = null;
  controls.addEventListener("change", () => { dirty = true; });
  // Only spin the controls (needed for damping) while the user is interacting.
  controls.addEventListener("start", () => { active = true; if (endTimer) { clearTimeout(endTimer); endTimer = null; } });
  controls.addEventListener("end", () => {
    if (endTimer) clearTimeout(endTimer);
    endTimer = setTimeout(() => { active = false; endTimer = null; }, 1200);
  });

  // Zoom only while Cmd/Ctrl is held; a plain wheel over the canvas then scrolls
  // the page instead of being captured as zoom. The listener sits on the mount in
  // the capture phase so it runs before OrbitControls' own wheel handler. Touch
  // pinch-zoom is unaffected.
  mount.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.metaKey) return;   // modifier held -> let OrbitControls zoom
    e.stopImmediatePropagation();          // otherwise let the page keep scrolling
  }, { capture: true });

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(0.6, -1.0, 1.2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7);
  fill.position.set(-0.8, 0.4, 0.5);
  scene.add(fill);

  // Everything for the current hand lives under one group we can clear.
  const root = new THREE.Group();
  scene.add(root);

  const capsuleMat = {
    contact: new THREE.MeshStandardMaterial({ color: 0x6495ed, roughness: 0.55, metalness: 0.0 }),
    idle: new THREE.MeshStandardMaterial({ color: 0xb4b4b4, roughness: 0.7, metalness: 0.0 }),
  };
  const sphereMat = new THREE.MeshStandardMaterial({
    color: 0xff5050, transparent: true, opacity: 0.55, roughness: 0.5,
  });

  const yAxis = new THREE.Vector3(0, 1, 0);
  // colorMode: "absolute" = fixed 0..coverage_max scale, comparable across hands;
  //            "relative" = rescaled to this hand's most-reorientable voxel.
  const state = { showHand: true, showVoxels: true, voxelOpacity: 0.16, hand: null,
                  colorMode: "absolute", data: null, handMaxCov: 0.25 };

  function clearRoot() {
    for (let i = root.children.length - 1; i >= 0; i--) {
      const o = root.children[i];
      root.remove(o);
      if (o.geometry) o.geometry.dispose();
    }
  }

  let voxelMesh = null;
  let handGroup = null;

  function activeMaxCov(d) {
    return state.colorMode === "relative" ? state.handMaxCov : d.coverage_max;
  }

  function recolorVoxels() {
    const d = state.data;
    if (!voxelMesh || !d) return;
    const mc = activeMaxCov(d);
    for (let i = 0; i < d.voxels.length; i++) {
      voxelMesh.setColorAt(i, coverageColor(d.voxels[i][3], mc));
    }
    if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true;
  }

  function buildHand(id) {
    const d = scenes[id];
    if (!d) return;
    state.hand = id;
    state.data = d;
    state.handMaxCov = Math.max(1e-6, ...d.voxels.map((v) => v[3]));
    clearRoot();

    // --- capsules (finger links) ---
    handGroup = new THREE.Group();
    const r = d.link_radius;
    for (const c of d.capsules) {
      const a = new THREE.Vector3(c[0], c[1], c[2]);
      const b = new THREE.Vector3(c[3], c[4], c[5]);
      const seg = new THREE.Vector3().subVectors(b, a);
      const len = seg.length();
      const geo = new THREE.CapsuleGeometry(r, Math.max(len, 1e-6), 6, 14);
      const mesh = new THREE.Mesh(geo, c[6] ? capsuleMat.contact : capsuleMat.idle);
      mesh.position.copy(a).add(b).multiplyScalar(0.5);
      if (len > 1e-9) mesh.quaternion.setFromUnitVectors(yAxis, seg.clone().normalize());
      handGroup.add(mesh);
    }
    // --- test sphere ---
    const sph = new THREE.Mesh(new THREE.SphereGeometry(d.sphere.r, 24, 16), sphereMat);
    sph.position.set(d.sphere.c[0], d.sphere.c[1], d.sphere.c[2]);
    handGroup.add(sph);
    // --- sphere body-frame triad ---
    const axcol = [0xff0000, 0x00cc00, 0x0000ff];
    const al = d.sphere.r * 1.6;
    for (let k = 0; k < 3; k++) {
      const dir = [0, 0, 0]; dir[k] = al;
      const p0 = new THREE.Vector3(...d.sphere.c);
      const p1 = p0.clone().add(new THREE.Vector3(...dir));
      const g = new THREE.BufferGeometry().setFromPoints([p0, p1]);
      handGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: axcol[k] })));
    }
    // --- contact normals (thumb red, index blue) ---
    const ncol = [0xff5050, 0x5050ff];
    (d.normals || []).forEach((n, i) => {
      const p0 = new THREE.Vector3(n[0], n[1], n[2]);
      const p1 = p0.clone().add(new THREE.Vector3(n[3], n[4], n[5]).multiplyScalar(d.sphere.r * 2.0));
      const g = new THREE.BufferGeometry().setFromPoints([p0, p1]);
      handGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: ncol[i] || 0x333333 })));
    });
    handGroup.visible = state.showHand;
    root.add(handGroup);

    // --- voxel cloud (InstancedMesh, per-instance coverage color) ---
    const n = d.voxels.length;
    const vgeo = new THREE.BoxGeometry(d.voxel_size, d.voxel_size, d.voxel_size);
    const vmat = new THREE.MeshBasicMaterial({
      transparent: true, opacity: state.voxelOpacity, depthWrite: false, vertexColors: false,
    });
    voxelMesh = new THREE.InstancedMesh(vgeo, vmat, n);
    const m = new THREE.Matrix4();
    const mc = activeMaxCov(d);
    for (let i = 0; i < n; i++) {
      const v = d.voxels[i];
      m.makeTranslation(v[0], v[1], v[2]);
      voxelMesh.setMatrixAt(i, m);
      voxelMesh.setColorAt(i, coverageColor(v[3], mc));
    }
    voxelMesh.instanceMatrix.needsUpdate = true;
    if (voxelMesh.instanceColor) voxelMesh.instanceColor.needsUpdate = true;
    voxelMesh.visible = state.showVoxels;
    root.add(voxelMesh);

    // --- camera framing ---
    camera.position.set(...d.camera.position);
    controls.target.set(...d.camera.target);
    controls.update();
    dirty = true;
  }

  // ---- resize ------------------------------------------------------------
  function resize() {
    const w = mount.clientWidth || 1;
    const h = mount.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    dirty = true;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(mount);
  resize();

  // Run the render loop only while the panel is on-screen; stop it entirely when
  // scrolled away so two viewers don't tax the main thread during page scroll.
  const io = new IntersectionObserver((entries) => {
    inView = entries[0].isIntersecting;
    if (inView) { dirty = true; if (!rafId) rafId = requestAnimationFrame(loop); }
    else if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }, { threshold: 0.01 });
  io.observe(mount);

  // ---- render loop (on demand) ------------------------------------------
  function loop() {
    rafId = requestAnimationFrame(loop);
    if (active && controls.update()) dirty = true; // only churn while interacting
    if (dirty) { renderer.render(scene, camera); dirty = false; }
  }

  // ---- API ---------------------------------------------------------------
  const api = {
    show: buildHand,
    resetView() { if (state.hand) buildHand(state.hand); },
    setShowHand(v) { state.showHand = v; if (handGroup) handGroup.visible = v; dirty = true; },
    setShowVoxels(v) { state.showVoxels = v; if (voxelMesh) voxelMesh.visible = v; dirty = true; },
    setVoxelOpacity(v) {
      state.voxelOpacity = v;
      if (voxelMesh) voxelMesh.material.opacity = v;
      dirty = true;
    },
    setColorMode(mode) { state.colorMode = mode; recolorVoxels(); dirty = true; },
    colorInfo() {
      // Upper end of the current color scale, as a percentage of orientations.
      const pct = (state.colorMode === "relative" ? state.handMaxCov
                   : (state.data ? state.data.coverage_max : 0.25)) * 100;
      return { mode: state.colorMode, maxPct: Math.round(pct) };
    },
    hands() {
      return Object.keys(scenes).map((id) => ({ id, label: scenes[id].label, t: scenes[id].karma_t }));
    },
    meta(id) { return scenes[id]; },
    // Orbit sharing: copy the view direction (unit vector target->camera) so two
    // panels rotate together while each keeps its own target and zoom distance.
    // OrbitControls has no angle setters, so we set the camera position directly.
    getDir() {
      const d = camera.position.clone().sub(controls.target);
      const n = d.length();
      return n > 1e-9 ? [d.x / n, d.y / n, d.z / n] : [0, 0, 1];
    },
    setDir(d) {
      const r = camera.position.distanceTo(controls.target);
      camera.position.set(
        controls.target.x + d[0] * r,
        controls.target.y + d[1] * r,
        controls.target.z + d[2] * r,
      );
      controls.update();
    },
    onChange(cb) { controls.addEventListener("change", cb); },
  };
  return api;
}
