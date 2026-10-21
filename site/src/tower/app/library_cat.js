import { isCameraHitVisible } from './camera_collision.js';
import { CAT_RADIUS, createCatLife, createCatPlan, pointAt } from './cat_plan.js';

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
const smooth = t => { t = clamp(t); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

/**
 * Chestnut's body, with a rounded chest, haunch and limbs, a face, and a tail of twelve pieces, all from two
 * shared geometries. The gait follows the distance walked and each paw stands on the actual ground under it, so on
 * steps the paws stand at different heights. Breathing, blinking, ears and the tail tip move only a little.
 * Every route is first checked against the real collision geometry, a little in each frame.
 */
export function createLibraryCat(THREE, scene, { plan, collision = null, clippingPlanes = [], seed } = {}) {
  const catPlan = createCatPlan(plan), group = new THREE.Group();
  group.name = 'Library cat / 栗子'; scene.add(group);
  const body = new THREE.Group();
  group.add(body);
  const materials = [], meshes = [], geometries = [], pickTargets = [], tail = [];
  const material = (name, color, extra = {}) => {
    const m = new THREE.MeshStandardMaterial({ name: `Library cat / ${name}`, color, roughness: .94, metalness: 0, clippingPlanes,
      clipShadows: true, ...extra });
    materials.push(m);
    return m;
  };
  const coat = material('warm ginger coat', '#bd8c60'), cream = material('soft cream paws and muzzle', '#e7d7b8');
  const dark = material('nose and pupils', '#302722'), velvet = material('ear velvet', '#bc8273'), tabby = material('soft tabby markings', '#885d3f');
  const amber = material('amber eyes', '#bdb775'), lid = material('resting eyelids', '#bd8c60');
  const sphere = new THREE.SphereGeometry(1, 16, 12);
  geometries.push(sphere);
  const tube = new THREE.CylinderGeometry(1, 1, 1, 10, 1);
  geometries.push(tube);
  function blob(parent, name, mat, position, scale) {
    const mesh = new THREE.Mesh(sphere, mat);
    mesh.name = name; mesh.position.set(...position); mesh.scale.set(...scale); mesh.castShadow = true; mesh.receiveShadow = true;
    parent.add(mesh); meshes.push(mesh); pickTargets.push(mesh);
    return mesh;
  }
  const ribcage = blob(body, 'Cat ribcage', coat, [0, .355, -.025], [.165, .185, .285]);
  const haunch = blob(body, 'Cat rounded haunch', coat, [0, .325, -.24], [.176, .185, .175]);
  const chest = blob(body, 'Cat chest', cream, [0, .375, .18], [.129, .174, .125]);
  const head = new THREE.Group();
  head.position.set(0, .45, .235); body.add(head);
  blob(head, 'Cat head', coat, [0, .045, .01], [.145, .143, .135]);
  for (const sign of [-1, 1]) {
    blob(head, 'Cat muzzle', cream, [sign * .038, -.006, .123], [.052, .039, .035]);
    const ear = new THREE.Group();
    ear.position.set(sign * .094, .187, -.025); head.add(ear);
    const earGeometry = new THREE.ConeGeometry(.073, .158, 3, 1);
    earGeometry.rotateY(sign * .22); geometries.push(earGeometry);
    const outer = new THREE.Mesh(earGeometry, coat);
    outer.name = 'Cat soft pointed ear'; outer.rotation.z = -sign * .18; ear.add(outer); meshes.push(outer); pickTargets.push(outer);
    const inner = new THREE.Mesh(earGeometry, velvet);
    inner.name = 'Cat inner ear'; inner.scale.set(.56, .66, .56); inner.position.set(0, .004, .025); ear.add(inner); meshes.push(inner);
    ear.userData.sign = sign;
    const eye = blob(head, 'Cat eye', amber, [sign * .068, .063, .117], [.033, .026, .019]);
    const pupil = blob(head, 'Cat pupil', dark, [sign * .068, .063, .133], [.007, .02, .004]);
    const eyelid = blob(head, 'Cat eyelid', lid, [sign * .068, .063, .12], [.035, .026, .02]);
    eyelid.userData.eye = eye; eyelid.userData.pupil = pupil; eyelid.userData.sign = sign;
    blob(head, 'Cat sleeping eye line', dark, [sign * .068, .063, .14], [.026, .0025, .002]);
    for (const y of [-.007, .014]) {
      const root = new THREE.Vector3(sign * .062, y, .142), tip = new THREE.Vector3(sign * .16, y + .012, .123), whisker = new THREE.Mesh(tube, cream);
      whisker.name = 'Cat soft whisker'; whisker.position.copy(root).add(tip).multiplyScalar(.5);
      whisker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tip.clone().sub(root).normalize());
      whisker.scale.set(.0017, root.distanceTo(tip), .0017); head.add(whisker); meshes.push(whisker);
    }
  }
  blob(head, 'Cat nose', velvet, [0, .013, .157], [.021, .014, .012]);
  const paws = [], legs = [];
  for (const back of [false, true]) for (const sign of [-1, 1]) {
    const hip = blob(body, back ? 'Cat thigh' : 'Cat shoulder', coat, [sign * .112, .31, back ? -.22 : .2], [.083, .115, .1]);
    const upper = new THREE.Mesh(tube, coat), lower = new THREE.Mesh(tube, back ? coat : cream);
    for (const mesh of [upper, lower]) { mesh.castShadow = true; mesh.receiveShadow = true; body.add(mesh); meshes.push(mesh); pickTargets.push(mesh); }
    const paw = blob(body, 'Cat planted paw', cream, [sign * .114, .035, back ? -.22 : .24], [.055, .039, .085]);
    // The four footfalls of a walk come a quarter cycle apart.
    const leg = { back, sign, hip, upper, lower, paw, phase: back ? sign > 0 ? .25 : .75 : sign > 0 ? .5 : 0 };
    legs.push(leg); paws.push(paw);
  }
  for (let i = 0; i < 12; i++) {
    const piece = new THREE.Mesh(tube, i > 8 ? tabby : coat);
    piece.castShadow = true; body.add(piece); meshes.push(piece); pickTargets.push(piece); tail.push(piece);
  }

  const down = new THREE.Vector3(0, -1, 0), up = new THREE.Vector3(0, 1, 0), probe = new THREE.Vector3(), sceneMaterials = new Set();
  scene.traverse(object => {
    for (const mat of Array.isArray(object.material) ? object.material : object.material ? [object.material] : []) sceneMaterials.add(mat);
  });
  /** Route checks must not depend on the section view, so clipping is lifted from every scene material while they run. */
  function unclipped(check) {
    const saved = [];
    for (const mat of sceneMaterials) if (mat.clippingPlanes?.length) { saved.push([mat, mat.clippingPlanes]); mat.clippingPlanes = []; }
    try {
      return check();
    } finally {
      for (const [mat, planes] of saved) mat.clippingPlanes = planes;
    }
  }
  const verdicts = new Map(), routeKey = path => JSON.stringify(path.map(segment => ({ ...segment, progress: 0 })),
    (key, value) => typeof value === 'number' ? Math.round(value * 1e6) / 1e6 : value);
  let lastFailure = null;
  /** Points every 34 cm or less along a route, each with four more around it at the cat's own radius. */
  function* samples(path) {
    for (const segment of path) {
      const steps = Math.max(1, Math.ceil(segment.length / .34));
      for (let i = 0; i <= steps; i++) {
        const p = pointAt(segment, segment.length * i / steps);
        for (let k = 0; k < 5; k++) {
          const angle = k * Math.PI / 2, reach = k === 4 ? 0 : CAT_RADIUS;
          yield { p, s: segment, x: p.x + Math.cos(angle) * reach, z: p.z + Math.sin(angle) * reach };
        }
      }
    }
  }
  /** A sample passes when real ground lies under it at the planned height and nothing hangs low above it. */
  function clear({ p, s, x, z }) {
    probe.set(x, p.y + .26, z);
    const floor = collision.nearest(probe, down, .002, .56), expected = catPlan.sampleGround(x, z, s.support) - .025;
    const tolerance = s.support?.type === 'garden' ? .205 : .065;
    if (!floor || !Number.isFinite(expected) || Math.abs(floor.point.y - expected) > tolerance) {
      lastFailure = { reason: 'support', p: probe.toArray(), expected, actual: floor?.point.y, segment: JSON.parse(JSON.stringify(s)) };
      return false;
    }
    probe.y = Math.max(p.y + .08, floor.point.y + .065);
    const ceiling = collision.nearest(probe, up, .005, .68);
    if (ceiling) {
      lastFailure = { reason: 'headroom', p: probe.toArray(), actual: ceiling.point.y, object: ceiling.object.name,
        segment: JSON.parse(JSON.stringify(s)) };
      return false;
    }
    return true;
  }
  function validateRoute(path) {
    if (!collision) return true;
    const key = routeKey(path);
    if (verdicts.has(key)) return verdicts.get(key);
    lastFailure = null;
    const ok = unclipped(() => {
      for (const sample of samples(path)) if (!clear(sample)) return false;
      return true;
    });
    if (verdicts.size < 64) verdicts.set(key, ok);
    return ok;
  }
  // Every route between two spots is checked ahead of time, a little in each frame.
  const pending = [];
  if (collision) {
    for (const from of catPlan.spots) for (const to of catPlan.spots) if (from.id !== to.id) {
      const route = catPlan.route(from.position, from, to);
      pending.push({ key: routeKey(route), iterator: samples(route) });
    }
  }
  /** At most 24 samples and two milliseconds in a frame. */
  function checkRoutes() {
    if (!pending.length) return;
    const started = performance.now();
    unclipped(() => {
      for (let n = 0; n < 24 && pending.length; n++) {
        const next = pending[0];
        if (verdicts.has(next.key)) { pending.shift(); n--; continue; }
        const result = next.iterator.next();
        if (result.done) { verdicts.set(next.key, true); pending.shift(); }
        else if (!clear(result.value)) { verdicts.set(next.key, false); pending.shift(); }
        if (performance.now() - started >= 2) break;
      }
    });
  }

  const life = createCatLife({ plan: catPlan, validateRoute: path => !collision || verdicts.get(routeKey(path)) === true, seed });
  const actor = life.actor, between = new THREE.Vector3(), boneUp = new THREE.Vector3(0, 1, 0), point = new THREE.Vector3();
  let disposed = false, sleep = 1, sit = 0, stretch = 0, bodyY = actor.position.y, section = false;
  const diagnostics = { paws: [], tailPoints: [], sleep: 1, sit: 0, stretch: 0 };
  function stretchBetween(mesh, from, to, radius, depth = radius) {
    mesh.position.copy(from).add(to).multiplyScalar(.5); between.subVectors(to, from);
    mesh.quaternion.setFromUnitVectors(boneUp, between.clone().normalize()); mesh.scale.set(radius, between.length(), depth);
  }
  function pose(dt) {
    // The section view cuts the cat indoors only. In the garden it is always whole.
    for (const mat of materials) mat.clippingPlanes = section && actor.support?.type !== 'garden' ? clippingPlanes : [];
    const time = actor.time || 0, progress = actor.phaseProgress;
    const sleepTarget = actor.phase === 'sleep' ? 1 : actor.phase === 'wake' ? 1 - smooth(progress) : actor.phase === 'curl' ? smooth(progress) : 0;
    const sitTarget = ['sit', 'groom', 'curl'].includes(actor.phase) ? 1 : 0;
    const stretchTarget = actor.phase === 'stretch' ? Math.sin(Math.PI * progress) : 0;
    const ease = dt === 0 ? 1 : 1 - Math.exp(-dt * 5);
    sleep = lerp(sleep, sleepTarget, ease); sit = lerp(sit, sitTarget * (1 - sleep), ease); stretch = lerp(stretch, stretchTarget, ease);
    bodyY = lerp(bodyY, actor.position.y, dt === 0 ? 1 : 1 - Math.exp(-dt * 14));
    group.position.set(actor.position.x, bodyY, actor.position.z); group.rotation.y = actor.yaw;
    const gait = clamp(actor.speed / .7), breath = Math.sin(time * 1.28) * .006, stride = actor.stride / .47 * Math.PI * 2;
    body.position.y = -.155 * sleep - .075 * sit + breath; body.rotation.z = Math.sin(stride) * .025 * gait;
    ribcage.scale.set(.165 + sleep * .028, .185 - sleep * .036, .285 - sleep * .04);
    ribcage.position.set(.012 * sleep, .355 - .035 * sit - .045 * stretch, -.025 - .012 * stretch);
    haunch.position.set(-.035 * sleep, .325 - sit * .065 + stretch * .035 + sleep * .025, -.24 + .05 * sleep);
    chest.position.y = .375 - stretch * .17 - .025 * sit;
    head.position.set(sleep * .09, .45 - sleep * .09 + sit * .045 - stretch * .14, .235 - sleep * .1 + stretch * .045);
    head.rotation.set(-stretch * .22 - sleep * .2, sleep * 1.05 + (1 - sleep) * Math.sin(time * .41) * .055,
      actor.phase === 'groom' ? Math.sin(time * 2.3) * .1 : Math.sin(time * .28) * .014);
    // A blink is a brief, rare peak of a slow sine.
    const blink = Math.pow(Math.max(0, Math.sin(time * .73)), 48);
    for (const part of head.children) {
      if (part.name === 'Cat eyelid') {
        const open = (1 - sleep) * (1 - .97 * blink);
        part.scale.y = .026 * (1 - open * .92); part.position.y = .063 + .024 * open;
      }
      if (part.name === 'Cat sleeping eye line') part.scale.set(.026 * (sleep + (1 - sleep) * blink), .0025, .002);
      if (part.userData.sign) part.rotation.z = -part.userData.sign * .08 + Math.sin(time * .69 + part.userData.sign) * .025;
    }
    diagnostics.paws.length = 0;
    for (const leg of legs) {
      const cycle = ((actor.stride / .47 + leg.phase) % 1 + 1) % 1, swing = cycle > .64, t = swing ? (cycle - .64) / .36 : cycle / .64;
      const reach = (leg.back ? -.23 : .205) + gait * (swing ? lerp(-.15, .15, smooth(t)) : lerp(.15, -.15, t));
      const lift = swing ? Math.sin(Math.PI * t) * .07 * gait : 0;
      const tuck = leg.back ? -.06 : .12, tuckX = leg.sign * (leg.back ? .075 : .085);
      const x = lerp(leg.sign * .117, tuckX, sleep);
      const z = lerp(reach, tuck, sleep) + (leg.back ? -.015 * stretch : .23 * stretch) + (leg.back ? .08 * sit : .015 * sit);
      let y = .036 + lift;
      {
        const worldX = actor.position.x + Math.cos(actor.yaw) * x + Math.sin(actor.yaw) * z;
        const worldZ = actor.position.z - Math.sin(actor.yaw) * x + Math.cos(actor.yaw) * z;
        y += catPlan.sampleGround(worldX, worldZ, actor.support) - bodyY;
      }
      const paw = new THREE.Vector3(x, y - body.position.y + .022 * sleep, z);
      leg.paw.position.copy(paw); leg.paw.rotation.x = swing ? -.2 * Math.sin(Math.PI * t) * gait : 0;
      const hip = new THREE.Vector3(leg.sign * .114, leg.back ? .33 + stretch * .035 - sit * .07 : .35 - stretch * .16, leg.back ? -.22 : .2);
      hip.x = lerp(hip.x, leg.sign * .09, sleep); hip.y -= sleep * .025; leg.hip.position.copy(hip);
      const knee = hip.clone().lerp(paw, .56);
      knee.z += leg.back ? .075 : -.038;
      stretchBetween(leg.upper, hip, knee, leg.back ? .054 : .036); stretchBetween(leg.lower, knee, paw, .033, .028);
      diagnostics.paws.push({ local: paw.toArray(), swing, supportY: y - lift + bodyY - .036 });
    }
    // The tail rises in a curve while awake and wraps round the body in sleep.
    diagnostics.tailPoints.length = 0;
    let previous = new THREE.Vector3(0, .38, -.355);
    for (let i = 0; i < tail.length; i++) {
      const t = (i + 1) / tail.length;
      const raised = new THREE.Vector3(Math.sin(time * .7 + t * 2.1) * .065 * t, .38 + .36 * Math.sin(t * 1.5) + .08 * gait * t, -.355 - t * .43 + .1 * t * t);
      const wrapped = new THREE.Vector3(-.21 * Math.sin(t * Math.PI * 1.27), .175, -.31 + .51 * t);
      point.copy(raised).lerp(wrapped, sleep).addScaledVector(up, -.05 * sit);
      stretchBetween(tail[i], previous, point, .039 * (1 - t * .72));
      previous = point.clone(); diagnostics.tailPoints.push(previous.toArray());
    }
    actor.viewHeight = lerp(actor.viewHeight, .4 - .19 * sleep - .035 * sit - .06 * stretch, ease);
    diagnostics.sleep = sleep; diagnostics.sit = sit; diagnostics.stretch = stretch;
    group.updateMatrixWorld(true);
  }
  function update(dt, options = {}) {
    if (disposed) return;
    checkRoutes(); life.update(dt, options); pose(Math.min(.12, Math.max(0, dt)));
  }
  function snapshot() {
    return { ...life.snapshot(), view: { sleep, sit, stretch, bodyY } };
  }
  function restore(data) {
    // A saved walk resumes only on a route that still passes the collision checks.
    if (data?.phase === 'walk' && Array.isArray(data.path) && !validateRoute(data.path) || !life.restore(data)) return false;
    sleep = clamp(Number.isFinite(data.view?.sleep) ? data.view.sleep : actor.phase === 'sleep' ? 1 : 0);
    sit = clamp(Number.isFinite(data.view?.sit) ? data.view.sit : 0);
    stretch = clamp(Number.isFinite(data.view?.stretch) ? data.view.stretch : 0);
    bodyY = Number.isFinite(data.view?.bodyY) && Math.abs(data.view.bodyY - actor.position.y) < .32 ? data.view.bodyY : actor.position.y;
    pose(1e-8);
    return true;
  }
  pose(0);
  const triangles = meshes.reduce((sum, mesh) => sum + (mesh.geometry.index?.count || mesh.geometry.attributes.position.count) / 3, 0);
  return { group, materials, meshes, pickTargets, actor, plan: catPlan, update, snapshot, restore,
    setSection(enabled) { section = !!enabled; pose(1e-8); },
    resolveHit(hit) { return hit && pickTargets.includes(hit.object) && isCameraHitVisible(hit) ? actor.id : null; },
    get diagnostics() {
      return { ...JSON.parse(JSON.stringify(diagnostics)), stats: life.stats, validationFailure: lastFailure, pendingRouteChecks: pending.length };
    },
    validateRoute, metrics: { cats: 1, drawCalls: meshes.length, triangles, textures: 0, lights: 0, restSpots: catPlan.spots.length },
    dispose() {
      if (disposed) return;
      disposed = true; life.dispose(); group.removeFromParent();
      geometries.forEach(geometry => geometry.dispose()); materials.forEach(mat => mat.dispose());
    } };
}
