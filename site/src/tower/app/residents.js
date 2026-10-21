import { BOOK_DIMENSIONS } from './book_view.js';

/**
 * Articulated, asset-free inhabitants of the clock tower.
 *
 * Metres; +Z is forwards and actor.position is the support below the feet.
 * All residents share nine instanced geometries. The second set of materials
 * keeps passengers attached to uncut moving supports. No actor owns a Mesh.
 */
export function createResidents(THREE, scene, { count = 48, clippingPlanes = [], sampleGround = null } = {}) {
  const TAU = Math.PI * 2, capacity = Math.max(1, Math.floor(count));
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const mix = (a, b, t) => a + (b - a) * t;
  const smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  const smoother = t => { t = clamp(t, 0, 1); return t * t * t * (t * (t * 6 - 15) + 10); };
  const damp = (a, b, rate, dt) => mix(a, b, 1 - Math.exp(-rate * dt));
  const finite = (x, fallback = 0) => Number.isFinite(x) ? x : fallback;
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const len = a => Math.hypot(...a);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const scale = (a, s) => a.map(x => x * s);
  const normalize = a => scale(a, 1 / Math.max(1e-9, len(a)));
  const blendPoint = (a, b, t) => a.map((v, i) => mix(v, b[i], t));
  const palette = ['#5d7875', '#776680', '#925d52', '#6f784f', '#587080', '#ad8b63', '#66574e', '#96796b', '#476768', '#888071'];
  const skin = ['#cda586', '#e0bd9d', '#b28669', '#92735c', '#d4ab91', '#ad886c'];
  const hair = ['#3e352e', '#685142', '#9c7c58', '#4e4943', '#554b40'];
  const up = new THREE.Vector3(0, 1, 0), v = new THREE.Vector3();
  const dummy = new THREE.Object3D(), root = new THREE.Object3D();
  const boneQuaternion = new THREE.Quaternion(), partQuaternion = new THREE.Quaternion();
  const rotation = new THREE.Euler(), worldMatrix = new THREE.Matrix4();
  const hidden = new THREE.Matrix4().makeScale(1e-7, 1e-7, 1e-7);
  hidden.setPosition(0, -10000, 0);
  const states = new Map(), batches = [], meshes = [], materials = [];
  const localParts = [], definitions = {};
  let disposed = false, clock = 0;

  function randomFor(value) {
    let seed = 2166136261;
    for (const c of String(value)) seed = Math.imul(seed ^ c.charCodeAt(0), 16777619);
    return () => {
      seed += 0x6D2B79F5;
      let x = seed;
      x = Math.imul(x ^ x >>> 15, x | 1);
      x ^= x + Math.imul(x ^ x >>> 7, x | 61);
      return ((x ^ x >>> 14) >>> 0) / 4294967296;
    };
  }
  function lathe(profile, segments = 12) {
    return new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), segments);
  }
  function define(key, geometry, stride, roughness, shadows = true) {
    const offset = localParts.length;
    for (let i = 0; i < stride; i++) localParts.push({ key, index: i });
    definitions[key] = { geometry, stride, roughness, shadows, offset };
  }
  // Sculpted waist, chest and sloping shoulders rather than a pill-shaped body.
  define('torso', lathe([[0, -.5], [.67, -.5], [.81, -.43], [.70, -.16], [.82, .20], [.98, .34], [.85, .43], [.44, .50], [0, .50]], 14), 1, .94);
  define('hem', lathe([[0, -.5], [1, -.5], [.99, -.43], [.88, -.06], [.69, .37], [.68, .5], [0, .5]], 14), 1, .96);
  define('limb', new THREE.CapsuleGeometry(.5, 1, 3, 8), 8, .95);
  define('head', new THREE.SphereGeometry(1, 14, 10), 1, .86);
  define('detail', new THREE.SphereGeometry(1, 8, 6), 16, .88, false);
  define('hair', new THREE.SphereGeometry(1, 12, 8, 0, TAU, 0, Math.PI * .50), 3, .95);
  define('shoe', new THREE.SphereGeometry(1, 10, 7), 2, .86);
  define('block', new THREE.BoxGeometry(1, 1, 1), 17, .91);
  define('strap', new THREE.CylinderGeometry(1, 1, 1, 6), 3, .91, false);

  for (let index = 0; index < 2; index++) {
    const group = new THREE.Group();
    group.name = index ? 'Residents in open air or on moving supports' : 'Residents on fixed galleries';
    scene.add(group);
    const batch = { index, group, meshes: {}, ids: [], materials: [] };
    for (const [key, d] of Object.entries(definitions)) {
      const material = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: d.roughness, metalness: 0,
        clippingPlanes: index ? [] : clippingPlanes, clipShadows: true });
      const mesh = new THREE.InstancedMesh(d.geometry, material, capacity * d.stride);
      mesh.name = `${group.name}: ${key}`;
      mesh.castShadow = d.shadows; mesh.receiveShadow = true; mesh.frustumCulled = false;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      // Allocate the colour attribute once. The compacted actor-to-instance map
      // is rebuilt together with the matrices, including when a rider disembarks.
      for (let i = 0; i < capacity * d.stride; i++) { mesh.setMatrixAt(i, hidden); mesh.setColorAt(i, new THREE.Color('#ffffff')); }
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      mesh.count = 0;
      mesh.userData.residentBatch = index; mesh.userData.residentStride = d.stride;
      group.add(mesh); batch.meshes[key] = mesh; batch.materials.push(material);
      meshes.push(mesh); materials.push(material);
    }
    batches.push(batch);
  }

  function makeState(actor) {
    const rand = randomFor(actor.appearanceSeed ?? actor.id);
    const age = ['child', 'elder'].includes(actor.age) ? actor.age : 'adult';
    const bodyScale = age === 'child' ? .70 + rand() * .09 : age === 'elder' ? .94 + rand() * .055 : .96 + rand() * .10;
    const coat = palette[Math.floor(rand() * palette.length)];
    const trousers = new THREE.Color(coat).multiplyScalar(.52 + rand() * .12);
    const complexion = skin[Math.floor(rand() * skin.length)];
    const hairColor = age === 'elder' ? (rand() > .5 ? '#b9b2a3' : '#918c82') : hair[Math.floor(rand() * hair.length)];
    const variant = Math.floor(rand() * 6);
    const a = { scale: bodyScale, width: .96 + rand() * .12, head: age === 'child' ? 1.09 : .98 + rand() * .04,
      age, variant, coat, trousers, skin: complexion, hair: hairColor, scarf: palette[(Math.floor(rand() * palette.length) + 3) % palette.length],
      long: variant === 0 || variant === 3, skirt: variant === 1 || (age === 'elder' && variant === 4),
      satchel: variant === 2 || variant === 5, scarfOn: variant === 0 || variant === 4,
      hairstyle: Math.floor(rand() * 4), seed: rand() * TAU, shoes: '#49443d' };
    const state = { id: actor.id, appearance: a, phase: finite(actor.gaitPhase) / TAU, distance: finite(actor.distanceTravelled),
      speed: 0, walk: 0, stairs: 0, sit: 0, read: 0, talk: 0, work: 0, observe: 0, idle: 0,
      seatOffset: [0, -.44 / bodyScale], seatHeight: .52,
      prop: null, propVisibility: 0, propRequested: null, detailClock: 0, secondaryTime: 0,
      initialized: false, active: true, actor, batch: 0, bookHold: 0, bookAnchors: null, lifeAnchors: null,
      bounds: { center: new THREE.Vector3(), radius: 1, height: 1.8, box: new THREE.Box3() },
      matrices: localParts.map(() => hidden.clone()), colors: localParts.map(() => new THREE.Color('#ffffff')),
      diagnostics: null };
    const colorPart = (key, index, color) => state.colors[definitions[key].offset + index].set(color);
    colorPart('torso', 0, coat); colorPart('hem', 0, a.skirt ? coat : new THREE.Color(coat).multiplyScalar(.92));
    for (let i = 0; i < 8; i++) colorPart('limb', i, i < 4 ? trousers : coat);
    colorPart('head', 0, complexion);
    for (let i = 0; i < 16; i++) colorPart('detail', i, i < 8 ? complexion : i < 10 ? '#3b3833' : i < 13 ? trousers : i < 15 ? coat : '#c6b38d');
    for (let i = 0; i < 3; i++) colorPart('hair', i, hairColor);
    for (let i = 0; i < 2; i++) colorPart('shoe', i, a.shoes);
    const blockColors = [a.scarf, a.scarf, '#5e4b38', '#695541', '#896448', '#a88459', '#e0d6bd', '#e8dfc8', '#e2d7b9', '#b19470', '#695541', '#857356', '#c3ab77', '#d3c5a8', coat, '#514a40', '#967b52'];
    blockColors.forEach((color, i) => colorPart('block', i, color));
    for (let i = 0; i < 3; i++) colorPart('strap', i, '#6c5640');
    states.set(actor.id, state); return state;
  }

  function put(state, key, index, p, s, q = null) {
    dummy.position.set(...p); dummy.scale.set(...s); dummy.quaternion.identity();
    if (q) dummy.quaternion.copy(q);
    dummy.updateMatrix(); state.matrices[definitions[key].offset + index].copy(dummy.matrix);
  }
  function box(state, index, p, s, angles = null) {
    partQuaternion.identity(); if (angles) partQuaternion.setFromEuler(rotation.set(...angles));
    put(state, 'block', index, p, s, partQuaternion);
  }
  function segment(state, key, index, a, b, radius, depth = radius) {
    const d = sub(b, a), length = len(d);
    boneQuaternion.setFromUnitVectors(up, v.set(...d).normalize());
    put(state, key, index, scale(add(a, b), .5), key === 'limb' ? [radius * 2, length / 2, depth * 2] : [radius, length, depth], boneQuaternion);
  }
  function joint(a, b, length1, length2, pole) {
    const delta = sub(b, a), distance = Math.max(.001, len(delta));
    const dir = scale(delta, 1 / distance);
    const reach = clamp(distance, Math.abs(length1 - length2) + .001, length1 + length2 - .001);
    let side = sub(pole, scale(dir, dot(pole, dir)));
    if (len(side) < .001) side = cross(dir, [.99, .02, .03]);
    side = normalize(side);
    const along = (length1 * length1 - length2 * length2 + reach * reach) / (2 * reach);
    const bend = Math.sqrt(Math.max(0, length1 * length1 - along * along));
    return add(a, add(scale(dir, along), scale(side, bend)));
  }
  function groundAt(actor, local, bodyScale) {
    const yaw = finite(actor.yaw), c = Math.cos(yaw), s = Math.sin(yaw);
    const x = actor.position.x + (local[0] * c + local[2] * s) * bodyScale;
    const z = actor.position.z + (local[2] * c - local[0] * s) * bodyScale;
    let y;
    if (sampleGround) y = sampleGround(x, z, actor);
    if (!Number.isFinite(y)) y = actor.footHeights?.[local[0] < 0 ? 0 : 1];
    return { x, z, y: finite(y, actor.position.y) };
  }
  function desiredProp(actor) {
    // Personal props are tucked away while the two hands collect the spill;
    // actor.carrying is preserved so the original prop returns after filing.
    if (actor.bookTask || ['patrol', 'paper'].includes(actor.lifeTask?.kind)) return null;
    if (actor.lifeTask?.kind === 'bridge_reading') return 'book';
    const type = typeof actor.carrying === 'string' ? actor.carrying : actor.carrying?.type;
    // A hand holding an umbrella keeps a book but puts other props away.
    if (finite(actor.weatherPose?.held) > .001 && type !== 'book') return null;
    return ['book', 'parcel', 'tools'].includes(type) ? type : actor.activity === 'read' ? 'book' : null;
  }

  function pose(state, actor, dt, time, near) {
    const a = state.appearance, S = a.scale;
    // A book spill takes the hands first. Otherwise an everyday story, then the weather, may pose them.
    const life = actor.bookTask ? null : actor.lifeTask, lifeKind = life?.kind, lifePhase = life?.phase;
    const lifeProgress = clamp(finite(life?.progress), 0, 1);
    const weather = actor.bookTask || life ? null : actor.weatherPose;
    const hug = clamp(finite(weather?.hug), 0, 1), umbrellaHeld = clamp(finite(weather?.held, weather?.umbrella), 0, 1);
    const paperCrouch = lifeKind === 'paper' && lifePhase === 'pickup' ? smooth(lifeProgress / .25) * (1 - smooth((lifeProgress - .48) / .4)) : 0;
    // Paper lying lower than a desk, under 85 cm, needs a deeper stoop to reach.
    const lowReach = paperCrouch && life?.target ? paperCrouch * clamp((.85 - (life.target.y - actor.position.y)) / .15, 0, 1) : 0;
    const bookTask = actor.bookTask || null, bookPhase = bookTask?.phase;
    const bookProgress = clamp(finite(bookTask?.progress), 0, 1);
    const picking = bookPhase === 'pickup', shelving = bookPhase === 'shelve';
    const crouch = picking ? smooth(bookProgress / .34) * (1 - smooth((bookProgress - .57) / .35)) : 0;
    const toLocal = p => {
      const dx = p.x - actor.position.x, dz = p.z - actor.position.z, yaw = finite(actor.yaw);
      return [(dx * Math.cos(yaw) - dz * Math.sin(yaw)) / S, (p.y - actor.position.y) / S,
        (dx * Math.sin(yaw) + dz * Math.cos(yaw)) / S];
    };
    const bookTarget = bookTask?.target ? toLocal(bookTask.target) : [0, 0, .36 / S];
    const pickupHand = bookTarget[0] < -.015 ? 0 : 1, pickupSide = pickupHand ? 1 : -1;
    const priorBooks = Math.max(0, finite(bookTask?.carriedCount));
    let holdTarget = bookTask && (priorBooks > 0 || picking && bookProgress > .42 || shelving) ? 1 : 0;
    if (shelving) holdTarget *= 1 - smooth((bookProgress - .65) / .30);
    state.bookHold = !state.initialized && bookTask ? holdTarget : damp(state.bookHold, holdTarget, 10, dt);
    // The bridge chair moves with its bridge, so its seat comes from the story rather than a fixed destination.
    const onBridgeSeat = lifeKind === 'bridge_reading';
    const posePosition = onBridgeSeat ? life.seatPosition : actor.destination?.posePosition || actor.posePosition;
    const seatFacing = onBridgeSeat ? life.seatFacing : actor.destination?.facing;
    const seatYawError = Number.isFinite(seatFacing) ? Math.abs(Math.atan2(Math.sin(actor.yaw - seatFacing), Math.cos(actor.yaw - seatFacing))) : 0;
    const seatIsHere = !posePosition || (Math.hypot(posePosition.x - actor.position.x, posePosition.z - actor.position.z) < .85 &&
      (onBridgeSeat || Math.abs(finite(posePosition.y, actor.position.y) - actor.position.y) < .20));
    const canSit = (onBridgeSeat ? finite(life.seated) > 1e-5 : actor.activity === 'sit') && seatIsHere && seatYawError < .30;
    const isWalking = Math.abs(finite(actor.speed)) > .025 && (actor.activity === 'walk' || actor.activity === 'ride' || !actor.activity);
    const targetSpeed = isWalking ? Math.abs(finite(actor.speed)) : 0;
    state.speed = damp(state.speed, targetSpeed, 10, dt);
    state.walk = damp(state.walk, clamp(targetSpeed / .44, 0, 1), 9, dt);
    for (const key of ['sit', 'read', 'talk', 'work', 'observe']) {
      const target = key === 'sit' ? canSit ? onBridgeSeat ? clamp(finite(life.seated), 0, 1) : 1 : 0
        : key === 'read' && onBridgeSeat ? ['sit', 'read', 'stand'].includes(lifePhase) ? 1 : 0 : actor.activity === key ? 1 : 0;
      // On the bridge chair the story times sitting down and standing up, so the seat follows it directly.
      state[key] = key === 'sit' && onBridgeSeat ? target : damp(state[key], target, key === 'sit' ? 3.9 : 5, dt);
    }
    const distance = finite(actor.distanceTravelled, state.distance);
    let travelled = Math.max(0, distance - state.distance);
    if (travelled > 2 || distance === state.distance) travelled = state.speed * dt;
    state.distance = distance;
    // Each child takes genuinely shorter steps at a higher cadence. Passive
    // lift/bridge motion never contributes to the locomotion phase.
    const onSteps = actor.support?.type === 'stairs' || actor.support?.type === 'bridge' || !!actor.support?.steps;
    state.stairs = damp(state.stairs, onSteps ? 1 : 0, 6, dt);
    const cycleDistance = mix(.79, .65, state.stairs) * S * (a.age === 'elder' ? .87 : a.age === 'child' ? .91 : 1);
    state.phase += travelled / cycleDistance;
    if (state.phase > 10000) state.phase %= 1;
    state.detailClock += dt;
    if (near || state.detailClock >= 1 / 15) { state.secondaryTime = time; state.detailClock = 0; }
    const secondary = state.secondaryTime + a.seed;
    const breathing = Math.sin(secondary * 1.7) * .004;
    const phase = state.phase, walk = state.walk * (1 - state.sit);
    const stride = cycleDistance / S * .30;
    const feet = [], supports = [], lifts = [], bodySupports = [];
    for (let index = 0; index < 2; index++) {
      const side = index ? 1 : -1, p = ((phase + index * .5) % 1 + 1) % 1;
      const stance = p < .60, u = stance ? p / .60 : (p - .60) / .40;
      const cycle = cycleDistance / S;
      // Lift before advancing, then settle vertically before weight transfer.
      // Subtracting body travel keeps the supporting foot still in the world.
      let z = stance ? mix(stride, -stride, u) : -stride - cycle * .4 * u + cycle * smoother((u - .13) / .74);
      z = z * walk + .025 * (1 - walk);
      // Short legs hang beneath the front of the bench, rather than reaching
      // for an adult's standing foot position in front of the seat.
      if (a.age === 'child') z = mix(z, -.085 / S, state.sit);
      else if (onBridgeSeat) z = mix(z, -.125, state.sit);
      const x = side * (.108 + state.sit * .023);
      const support = groundAt(actor, [x, 0, z], S);
      let supportTrajectory = support.y, soleY = support.y;
      if (!stance && walk > .001) {
        const takeoff = groundAt(actor, [x, 0, -stride - cycle * .4 * u], S);
        const landing = groundAt(actor, [x, 0, stride + cycle * .4 * (1 - u)], S);
        const peak = Math.max(takeoff.y, landing.y) + (.10 + state.stairs * .04) * S;
        const planned = u < .42 ? mix(takeoff.y, peak, smooth(u / .42)) : mix(peak, landing.y, smooth((u - .42) / .58));
        soleY = Math.max(support.y, mix(support.y, planned, walk));
        supportTrajectory = mix(support.y, mix(takeoff.y, landing.y, smooth(u)), walk);
      }
      const seatedDangling = a.age === 'child' ? Math.max(0, (state.seatHeight - .10) / S - .40) * state.sit : 0;
      soleY += seatedDangling * S;
      feet.push([x, (soleY - actor.position.y) / S + .073, z]);
      supports.push(support); lifts.push(soleY - support.y); bodySupports.push(supportTrajectory);
    }
    const gaitBob = (1 - Math.cos(phase * TAU * 2)) * .007 * walk;
    const sway = Math.sin(phase * TAU) * .018 * walk;
    let pelvis = [sway, .914 - state.stairs * walk * .13 + gaitBob + breathing, -.008];
    if (crouch) { pelvis[1] -= .52 * crouch; pelvis[2] -= .075 * crouch; }
    if (paperCrouch) { pelvis[1] -= .035 * paperCrouch; pelvis[2] -= .012 * paperCrouch; }
    if (lowReach) { pelvis[1] -= .2 * lowReach; pelvis[2] += .025 * lowReach; }
    // Navigation may reserve its next destination before a seated person has
    // stood up. Keep the current seat through the blend instead of pulling the
    // hips towards a newly reserved bench on another floor.
    if (canSit && posePosition) {
      const dx = finite(posePosition.x, actor.position.x) - actor.position.x, dz = finite(posePosition.z, actor.position.z) - actor.position.z;
      const yaw = finite(actor.yaw);
      state.seatOffset[0] = clamp((dx * Math.cos(yaw) - dz * Math.sin(yaw)) / S, -.55, .55);
      state.seatOffset[1] = clamp((dx * Math.sin(yaw) + dz * Math.cos(yaw)) / S, -.65, .12);
    }
    if (canSit) state.seatHeight = onBridgeSeat && posePosition ? posePosition.y - actor.position.y : finite(actor.destination?.seatHeight ?? actor.seatHeight, .52);
    const [seatX, seatBack] = state.seatOffset, seatHeight = state.seatHeight;
    pelvis = blendPoint(pelvis, [seatX, seatHeight / S + .105, seatBack], state.sit);
    // On steps, the legs solve against each physical tread. The pelvis uses
    // their mean height and a gentle speed-dependent rise, not a rigid root Y.
    const supportMean = ((bodySupports[0] + bodySupports[1]) * .5 - actor.position.y) / S;
    pelvis[1] += supportMean * (1 - state.sit);
    const legLength1 = .424, legLength2 = .422;
    // Keep two-bone reach valid while descending; floor support is never moved
    // down to accommodate the hip. A small squat is the physical solution.
    if (state.sit < .85) {
      for (let i = 0; i < feet.length; i++) {
        const foot = feet[i], hipX = pelvis[0] + (i ? 1 : -1) * .105;
        const horizontal = Math.hypot(foot[0] - hipX, foot[2] - pelvis[2]);
        const maxHeight = foot[1] + .012 + Math.sqrt(Math.max(.1, .835 * .835 - horizontal * horizontal));
        pelvis[1] = Math.min(pelvis[1], maxHeight);
      }
    }
    const lean = .025 * walk + (a.age === 'elder' ? .032 : 0) + state.read * .035 + state.work * .115 + hug * .025;
    const bookBend = 1.10 * crouch + .24 * paperCrouch + .26 * lowReach;
    const chest = add(pelvis, [0, .278 * Math.cos(bookBend) + breathing, lean + .278 * Math.sin(bookBend)]);
    const shoulder = add(pelvis, [0, .487 * Math.cos(bookBend), lean * 1.8 + .487 * Math.sin(bookBend)]);
    const tilt = .035 * Math.sin(phase * TAU) * walk;
    partQuaternion.setFromEuler(rotation.set(lean * .8 + bookBend, -.035 * Math.sin(phase * TAU) * walk, tilt));
    put(state, 'torso', 0, chest, [.238 * a.width, .548, .177], partQuaternion);
    put(state, 'detail', 10, add(pelvis, [0, -.013, 0]), [.183 * a.width, .12, .146]);
    const hemHeight = a.skirt ? .56 : a.long ? .34 : .13;
    const hemTop = pelvis[1] + .115;
    partQuaternion.setFromEuler(rotation.set(state.sit * -.72 - crouch * .42, 0, tilt * .5));
    put(state, 'hem', 0, [pelvis[0], hemTop - hemHeight * .5, pelvis[2] + state.sit * .10],
      [a.skirt ? .245 : .218, hemHeight, a.skirt ? .208 + state.sit * .12 : .166], partQuaternion);

    // Both knees are independently articulated. Foot pitch rolls over the toe
    // only during swing so the sole cannot slice through the supporting tread.
    const knees = [];
    for (let index = 0; index < 2; index++) {
      const side = index ? 1 : -1, hip = add(pelvis, [side * .105, -.012, 0]);
      const foot = feet[index], knee = joint(hip, foot, legLength1, legLength2, [side * .10, .025, 1]);
      knees.push(knee);
      const base = index * 2;
      segment(state, 'limb', base, hip, knee, a.skirt ? .065 : .072, .069);
      segment(state, 'limb', base + 1, knee, foot, .053, .051);
      put(state, 'detail', 11 + index, knee, [.056, .057, .059]);
      const p = ((phase + index * .5) % 1 + 1) % 1;
      const pitch = p > .60 ? Math.sin((p - .60) / .40 * TAU) * .16 * walk : 0;
      partQuaternion.setFromEuler(rotation.set(pitch, side * .055, 0));
      put(state, 'shoe', index, [foot[0], foot[1] - .018, foot[2] + .035], [.069, .055, .125], partQuaternion);
    }

    const propRequested = desiredProp(actor);
    state.propRequested = propRequested;
    // An exchanged object is first tucked away; its mesh is never simply
    // replaced in a visible hand. Reading and carrying share the same book.
    if (!state.prop && propRequested) state.prop = propRequested;
    const matching = state.prop === propRequested;
    state.propVisibility = damp(state.propVisibility, matching && propRequested ? 1 : 0, 7, dt);
    if (!matching && state.propVisibility < .025) { state.prop = propRequested; state.propVisibility = .015; }
    const carrying = state.propVisibility, reading = state.read * (state.prop === 'book' ? carrying : 0);
    const bookY = mix(pelvis[1] + .13, pelvis[1] + .32, reading);
    const bookZ = mix(pelvis[2] + .205, pelvis[2] + .325, reading);
    let propPosition = [pelvis[0] + .015 * Math.sin(secondary) * walk, bookY, bookZ];
    // In the rain a carried book is hugged against the chest.
    if (state.prop === 'book') propPosition = blendPoint(propPosition, add(pelvis, [-.105, .29, .198]), hug);
    if (state.prop === 'tools') propPosition = blendPoint(propPosition, [pelvis[0] + .07, 1.03 / S, .61 / S], state.work);
    // Stacked books are rendered by book_view. These are the actual hand
    // targets, in body coordinates; world anchors below expose the same solve.
    let stackBottom = [pelvis[0] - pickupSide * .070 * crouch, pelvis[1] + .165, pelvis[2] + .315];
    let shelfLocal = null;
    if (bookTask?.shelfPosition) shelfLocal = toLocal(bookTask.shelfPosition);
    if (shelving && shelfLocal) stackBottom = blendPoint(stackBottom, shelfLocal, smooth(bookProgress / .57));
    const stackQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, finite(actor.yaw), 0));
    if (shelving && Number.isFinite(bookTask?.shelfYaw)) stackQ.slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, bookTask.shelfYaw, 0)), smooth(bookProgress / .57));
    const elbows = [], hands = [], wantedHands = [];
    const lifeTarget = life?.handTarget || life?.target, lifeHand = lifeTarget ? toLocal(lifeTarget) : null;
    const pocket = add(pelvis, [.235, -.005, -.01]);
    const weatherHand = weather?.handTarget ? toLocal(weather.handTarget) : null;
    for (let index = 0; index < 2; index++) {
      const side = index ? 1 : -1;
      const top = add(shoulder, [side * .207 * a.width, -.016, 0]);
      const swing = Math.sin((phase + index * .5) * TAU) * walk;
      let hand = add(pelvis, [side * .235, -.091, -.15 * swing + .034]);
      hand = blendPoint(hand, add(pelvis, [side * .19, -.035, .25]), state.sit);
      const talkHand = add(pelvis, [side * (.27 + .05 * Math.sin(secondary * 1.5)), .20 + .035 * Math.sin(secondary * 2.3 + index), .21]);
      hand = blendPoint(hand, talkHand, state.talk * (index ? .84 : .40));
      const workHand = [pelvis[0] + side * .16, .997 / S + Math.sin(secondary * 1.35 + index) * .018, .59 / S + Math.sin(secondary * .9 + index) * .025];
      hand = blendPoint(hand, workHand, state.work);
      const lookHand = add(pelvis, [side * .25, .10, .28]);
      hand = blendPoint(hand, lookHand, state.observe * .55);
      const carryHand = add(propPosition, [side * (state.prop === 'parcel' ? .145 : .115), -.045, .018]);
      hand = blendPoint(hand, carryHand, carrying * (1 - (state.prop === 'tools' ? state.work * .78 : 0)));
      if (reading && index === 1) {
        const pageCycle = ((time + a.seed) / 7.7) % 1;
        const turning = Math.sin(Math.PI * clamp((pageCycle - .68) / .23, 0, 1));
        hand[0] -= turning * .105 * reading; hand[1] += turning * .030 * reading;
      }
      if (state.bookHold > .00001) {
        const supportHand = add(stackBottom, [side * .145 / S, -.024 / S, 0]);
        hand = blendPoint(hand, supportHand, state.bookHold);
      }
      if (picking && index === pickupHand) {
        const restingCenter = add(bookTarget, [0, BOOK_DIMENSIONS.height / (2 * S), 0]);
        const carryCenter = add(stackBottom, [0, (BOOK_DIMENSIONS.height / 2 + finite(bookTask.bookIndex) * BOOK_DIMENSIONS.spacing) / S, 0]);
        const bookCenter = blendPoint(restingCenter, carryCenter, smooth((bookProgress - .46) / .42));
        const grasp = add(bookCenter, [0, .026 / S, 0]);
        const reach = smooth(bookProgress / .28) * (1 - smooth((bookProgress - .88) / .12));
        hand = blendPoint(hand, grasp, reach);
      }
      if (lifeKind === 'patrol') {
        if (index === 0) {
          // The left hand carries the lantern beside the body, reaching to the rack to take it down and hang it up.
          const carry = add(pelvis, [-.285, .29 + Math.sin(secondary * 1.7) * .004, .24]);
          let target = carry, weight = clamp(finite(life.holding, 1), 0, 1);
          if (lifePhase === 'take' && lifeHand) {
            const reachOut = smooth(lifeProgress / .32), bringBack = smooth((lifeProgress - .48) / .42);
            target = blendPoint(lifeHand, carry, bringBack); weight = Math.max(weight, reachOut);
          } else if (lifePhase === 'hang' && lifeHand) {
            target = blendPoint(carry, lifeHand, smooth(lifeProgress / .6)); weight = 1 - smooth((lifeProgress - .82) / .18);
          }
          hand = blendPoint(hand, target, weight);
        } else if (lifePhase === 'inspect') {
          const point = Math.sin(Math.PI * lifeProgress) * .55;
          hand = blendPoint(hand, add(pelvis, [.2, .34, .4]), point);
        }
      } else if (lifeKind === 'paper') {
        // Both hands hold the paper. They move apart as it opens and together as it folds.
        const open = lifePhase === 'open' ? smooth(lifeProgress) : lifePhase === 'fold' ? 1 - smooth(lifeProgress) : lifePhase === 'read' ? 1 : 0;
        let center = add(pelvis, [0, .29, .34]), weight = ['open', 'read', 'fold', 'store'].includes(lifePhase) ? 1 : 0;
        if (lifePhase === 'pickup' && lifeHand) { center = blendPoint(lifeHand, center, smooth((lifeProgress - .45) / .43)); weight = smooth(lifeProgress / .3); }
        else if (lifePhase === 'store') center = blendPoint(center, pocket, smooth(lifeProgress / .83));
        const spread = mix(.062, .118, open) / S;
        hand = blendPoint(hand, add(center, [side * spread, 0, 0]), weight);
      } else if (weather) {
        if (index === 0 && hug > 0) hand = blendPoint(hand, add(propPosition, [-.085, -.02, .04]), hug);
        if (index === 1) {
          // The right hand holds the umbrella over the shoulder and works it at the stand.
          const carry = add(pelvis, [.25, .31, .255]);
          let target = carry, weight = umbrellaHeld;
          const taking = ['take', 'retrieve', 'pickup'].includes(weather.phase), placing = ['stow', 'store', 'dropoff', 'hang'].includes(weather.phase);
          const progress = clamp(finite(weather.progress), 0, 1);
          if (taking && weatherHand) {
            target = blendPoint(weatherHand, carry, smooth((progress - .48) / .42)); weight = Math.max(weight, smooth(progress / .32));
          } else if (placing && weatherHand) {
            target = blendPoint(carry, weatherHand, smooth(progress / .6)); weight = Math.max(umbrellaHeld, 1 - smooth((progress - .82) / .18));
          } else if (['coat', 'adjust', 'shake'].includes(weather.phase)) {
            target = add(shoulder, [-.09, -.19, .18]); weight = Math.sin(Math.PI * progress) * .85;
          }
          hand = blendPoint(hand, target, weight);
        }
      }
      wantedHands.push(hand);
      const handOffset = sub(hand, top), handReach = len(handOffset);
      if (handReach > .598) hand = add(top, scale(handOffset, .598 / handReach));
      const elbow = joint(top, hand, .312, .288, [side * .42, -.65, -.4]);
      elbows.push(elbow); hands.push(hand);
      segment(state, 'limb', 4 + index * 2, top, elbow, .061, .061);
      segment(state, 'limb', 5 + index * 2, elbow, hand, .046, .045);
      put(state, 'detail', index, hand, [.043, .055, .039]);
      put(state, 'detail', 13 + index, elbow, [.047, .045, .047]);
    }
    const nod = Math.sin(secondary * .8) * .025 * (state.talk + state.observe);
    const headCenter = add(pelvis, [.010 * Math.sin(secondary * .37) * (1 - walk), .712 * Math.cos(bookBend) + breathing, lean * 2 + .012 + .712 * Math.sin(bookBend)]);
    const noticeLook = bookPhase === 'notice' ? -.23 * Math.sin(Math.PI * bookProgress) : 0;
    const paperLook = lifeKind === 'paper' && ['pickup', 'open', 'read', 'fold', 'store'].includes(lifePhase) ? .2 : 0;
    const chartLook = lifeKind === 'patrol' && lifePhase === 'inspect' ? .1 * Math.sin(Math.PI * lifeProgress) : 0;
    const headYaw = Math.sin(secondary * .43) * .085 * (1 - walk);
    const headPitch = reading * .24 + state.work * .09 + nod + crouch * .44 + noticeLook + paperLook + chartLook + hug * .045;
    const headQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(headPitch, headYaw, -tilt * .6));
    const headPart = (key, index, offset, s, q = null) => {
      const p = new THREE.Vector3(...offset).applyQuaternion(headQ).add(new THREE.Vector3(...headCenter));
      partQuaternion.copy(headQ); if (q) partQuaternion.multiply(q);
      put(state, key, index, p.toArray(), s, partQuaternion);
    };
    headPart('head', 0, [0, 0, 0], [.137 * a.head, .178 * a.head, .138 * a.head]);
    put(state, 'detail', 2, add(headCenter, [0, -.176, -.007]), [.069, .100, .067]);
    headPart('detail', 3, [-.135 * a.head, -.005, 0], [.027, .043, .027]);
    headPart('detail', 4, [.135 * a.head, -.005, 0], [.027, .043, .027]);
    headPart('detail', 5, [0, -.012, .137 * a.head], [.024, .038, .030]);
    // Small inset eye points read at follow distance without a doll-like face.
    headPart('detail', 8, [-.047, .036, .124 * a.head], [.008, .010, .008]);
    headPart('detail', 9, [.047, .036, .124 * a.head], [.008, .010, .008]);
    headPart('hair', 0, [0, .052 * a.head, -.014], [.146 * a.head, .147 * a.head, .145 * a.head]);
    if (a.hairstyle === 0 || a.hairstyle === 3) {
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI * .30, 0, 0));
      headPart('hair', 1, [-.028, .045, -.055], [.13, a.hairstyle === 3 ? .23 : .18, .12], q);
    }
    if (a.hairstyle === 1 || a.age === 'elder') {
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
      headPart('hair', 2, [0, .005, -.146], [.075, .082, .081], q);
    }

    const flutter = Math.sin(secondary * 2.2 + phase * TAU) * .035 * (.20 + walk);
    if (a.scarfOn) {
      box(state, 0, add(shoulder, [0, -.018, .03]), [.153, .058, .162], [0, 0, .04]);
      box(state, 1, add(shoulder, [.063, -.16, .169]), [.065, .31, .026], [flutter, 0, flutter * .3]);
    }
    // Narrow waist belt and small collar keep the silhouette tailored.
    box(state, 2, add(pelvis, [0, .076, .004]), [.318 * a.width, .031, .286]);
    box(state, 14, add(shoulder, [0, -.007, .036]), [.123, .071, .117], [0, 0, -.035]);
    put(state, 'detail', 15, add(pelvis, [.028, .07, .152]), [.027, .025, .012]);
    if (a.satchel) {
      const bag = add(pelvis, [.224, -.075, -.046 + flutter * .3]);
      box(state, 3, bag, [.143, .214, .143], [0, -.16, flutter * .35]);
      box(state, 10, add(bag, [0, .087, .015]), [.155, .042, .154], [0, -.16, flutter * .35]);
      segment(state, 'strap', 0, add(shoulder, [-.115, .016, .13]), add(bag, [-.02, .11, .075]), .014, .008);
      segment(state, 'strap', 1, add(shoulder, [-.115, .016, -.10]), add(bag, [-.02, .11, -.05]), .014, .008);
    }
    if (carrying > .001 && state.prop) {
      const visibility = smooth(carrying), sc = size => size.map(x => x * Math.max(.001, visibility));
      const p = propPosition;
      if (state.prop === 'book') {
        // Covers and page blocks turn about the spine, from closed while carried to almost flat while read.
        const half = mix(Math.PI / 2, .18, reading);
        const bookQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(mix(-.10, -2.16, reading), mix(Math.PI / 2, 0, reading), 0));
        const onBook = (offset, angle) => ({
          position: new THREE.Vector3(...offset).multiplyScalar(visibility).applyQuaternion(bookQ).add(new THREE.Vector3(...p)).toArray(),
          quaternion: bookQ.clone().multiply(new THREE.Quaternion().setFromAxisAngle(up, angle)) });
        for (let i = 0; i < 2; i++) {
          const side = i ? 1 : -1, angle = side * half, hinge = side * .074 * Math.cos(half);
          const cover = onBook([hinge - Math.sin(angle) * .026, 0, -Math.cos(angle) * .026], angle);
          const pages = onBook([hinge - Math.sin(angle) * .010, 0, -Math.cos(angle) * .010], angle);
          put(state, 'block', 4 + i, cover.position, sc([.154, .224, .012]), cover.quaternion);
          put(state, 'block', 6 + i, pages.position, sc([.143, .208, .018]), pages.quaternion);
        }
        const pageCycle = ((time + a.seed) / 7.7) % 1, turn = smooth((pageCycle - .68) / .23);
        if (reading > .02 && pageCycle > .68 && pageCycle < .91) {
          const pageAngle = mix(half, Math.PI - half, turn);
          const page = onBook([.069 * Math.cos(pageAngle), .002, .069 * Math.sin(pageAngle) + .003], -pageAngle);
          put(state, 'block', 8, page.position, sc([.138, .205, .0025]), page.quaternion);
        }
      } else if (state.prop === 'parcel') {
        box(state, 9, p, sc([.277, .21, .185]), [0, flutter * .5, 0]);
        box(state, 12, add(p, [0, .001, .095]), sc([.014, .215, .009]));
        box(state, 13, add(p, [0, .109, 0]), sc([.281, .01, .015]));
      } else {
        box(state, 11, p, sc([.265, .13, .15]));
        segment(state, 'strap', 2, add(p, [-.088 * visibility, .12 * visibility, 0]), add(p, [.088 * visibility, .12 * visibility, 0]), .019 * visibility);
        box(state, 15, add(p, [0, -.036, .076]), sc([.271, .033, .016]));
        box(state, 16, add(p, [.06, .071, 0]), sc([.035, .025, .132]));
      }
    }
    const toWorld = local => {
      const yaw = finite(actor.yaw), x = local[0] * S, z = local[2] * S;
      return new THREE.Vector3(actor.position.x + x * Math.cos(yaw) + z * Math.sin(yaw),
        actor.position.y + local[1] * S, actor.position.z + z * Math.cos(yaw) - x * Math.sin(yaw));
    };
    // Hand anchors for the paper, lantern and umbrella views, solved with the same arms that are drawn.
    const worldHands = hands.map(toWorld), bodyQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, finite(actor.yaw), 0));
    const centerQ = bodyQ.clone().multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -.22));
    state.lifeAnchors = { left: { position: worldHands[0], quaternion: bodyQ.clone() }, right: { position: worldHands[1], quaternion: bodyQ.clone() },
      center: { position: worldHands[0].clone().add(worldHands[1]).multiplyScalar(.5), quaternion: centerQ },
      stash: { position: toWorld(pocket), quaternion: bodyQ.clone() }, pelvis: toWorld(pelvis), scale: S, batch: state.batch,
      kind: lifeKind || (weather ? 'weather' : null), phase: lifePhase || weather?.phase || null,
      progress: life ? lifeProgress : finite(weather?.progress), target: lifeTarget || weather?.handTarget || null,
      handErrors: hands.map((hand, i) => len(sub(hand, wantedHands[i])) * S) };
    if (bookTask) {
      const pickupCenter = toWorld(hands[pickupHand]); pickupCenter.y -= .026;
      state.bookAnchors = { pickup: { position: pickupCenter, quaternion: stackQ.clone() },
        stack: { position: toWorld(stackBottom), quaternion: stackQ },
        shelf: { position: shelfLocal ? toWorld(shelfLocal) : toWorld(stackBottom),
          quaternion: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, finite(bookTask.shelfYaw, actor.yaw), 0)) },
        pickupHand, scale: S, phase: bookPhase, progress: bookProgress,
        handPositions: hands.map(toWorld), floorY: actor.position.y };
    } else state.bookAnchors = null;
    state.diagnostics = { id: actor.id, age: a.age, batch: state.batch, scale: S, activity: actor.activity,
      feet: feet.map((f, i) => ({ x: supports[i].x, y: actor.position.y + (f[1] - .073) * S, z: supports[i].z,
        supportY: supports[i].y, lift: lifts[i], stance: lifts[i] < .001 })),
      pelvis: pelvis.map((n, i) => n * S + (i === 1 ? actor.position.y : 0)),
      legReach: feet.map((f, i) => len(sub(add(pelvis, [(i ? 1 : -1) * .105, -.012, 0]), f))),
      knees: knees.map(p => p.map(n => n * S)), hands: hands.map(p => p.map(n => n * S)), elbows: elbows.map(p => p.map(n => n * S)),
      prop: state.prop, propVisibility: carrying, sitting: state.sit, phase, height: (headCenter[1] + .19 * a.head) * S,
      detailRate: near ? 60 : 15,
      life: life || weather ? { kind: state.lifeAnchors.kind, phase: state.lifeAnchors.phase, seated: state.sit,
        pelvis: state.lifeAnchors.pelvis.toArray(), handErrors: state.lifeAnchors.handErrors, hands: worldHands.map(p => p.toArray()) } : null,
      bookTask: bookTask ? { phase: bookPhase, progress: bookProgress, crouch,
        hold: state.bookHold, pickupHand, priorBooks, pickupReachError: picking ? state.bookAnchors.pickup.position.distanceTo(new THREE.Vector3(bookTask.target.x, bookTask.target.y + BOOK_DIMENSIONS.height / 2, bookTask.target.z)) : null } : null };
    const height = state.diagnostics.height;
    state.bounds.height = height; state.bounds.radius = Math.max(.42, height * .52);
    state.bounds.center.set(actor.position.x, actor.position.y + height * .52, actor.position.z);
    const width = .44 * S;
    state.bounds.box.min.set(actor.position.x - width, Math.min(...supports.map(s => s.y)), actor.position.z - width - state.sit * .48);
    state.bounds.box.max.set(actor.position.x + width, actor.position.y + height, actor.position.z + width);
  }

  function update(actors, dt = 1 / 60, time = null, camera = null) {
    if (disposed) return;
    dt = clamp(finite(dt, 1 / 60), 0, .15); clock = Number.isFinite(time) ? time : clock + dt;
    for (const state of states.values()) state.active = false;
    for (const batch of batches) batch.ids.length = 0;
    const activeActors = actors.filter(actor => actor && actor.active !== false && actor.id !== undefined && actor.position &&
      ['x', 'y', 'z'].every(key => Number.isFinite(actor.position[key]))).slice(0, capacity);
    // Only micro gestures are sampled less often in the distance. Bodies, IK,
    // props and world transforms remain continuous; no mesh is ever swapped.
    const nearIds = new Set((camera?.position ? [...activeActors].sort((a, b) => {
      const d = actor => (actor.position.x - camera.position.x) ** 2 + (actor.position.y - camera.position.y) ** 2 + (actor.position.z - camera.position.z) ** 2;
      return d(a) - d(b);
    }).slice(0, 12) : activeActors).map(actor => actor.id));
    for (const actor of activeActors) {
      let state = states.get(actor.id); if (!state) state = makeState(actor);
      state.actor = actor; state.active = true;
      const openAir = actor.support?.type === 'garden' && (Math.hypot(actor.position.x,actor.position.z)<22 || Math.hypot(actor.position.x,actor.position.z)>33);
      state.batch = openAir || ['bridge', 'lift'].includes(actor.support?.type) ? 1 : 0;
      const batch = batches[state.batch], index = batch.ids.length; batch.ids.push(actor.id);
      for (const matrix of state.matrices) matrix.copy(hidden);
      pose(state, actor, dt, clock, nearIds.has(actor.id));
      root.position.set(actor.position.x, actor.position.y, actor.position.z);
      root.rotation.set(0, finite(actor.yaw), 0); root.scale.setScalar(state.appearance.scale); root.updateMatrix();
      for (const [key, definition] of Object.entries(definitions)) {
        const mesh = batch.meshes[key];
        for (let i = 0; i < definition.stride; i++) {
          const local = definition.offset + i, instance = index * definition.stride + i;
          worldMatrix.multiplyMatrices(root.matrix, state.matrices[local]); mesh.setMatrixAt(instance, worldMatrix);
          mesh.setColorAt(instance, state.colors[local]);
        }
      }
      state.initialized = true;
    }
    for (const batch of batches) for (const [key, mesh] of Object.entries(batch.meshes)) {
      mesh.count = batch.ids.length * definitions[key].stride; mesh.visible = !!mesh.count;
      // Raycaster caches the aggregate bounds. Residents continue travelling,
      // so a subsequent click must rebuild that bound at the current position.
      mesh.boundingSphere = null; mesh.boundingBox = null;
      mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
    }
  }
  function resolveHit(hit) {
    const data = hit?.object?.userData;
    if (!data || !Number.isInteger(hit.instanceId) || data.residentStride === undefined) return null;
    const batch = batches[data.residentBatch];
    return batch?.ids[Math.floor(hit.instanceId / data.residentStride)] ?? null;
  }
  function boundsForActor(id) {
    const state = states.get(id);
    if (!state?.active) return null;
    return state.bounds;
  }
  function getBookAnchors(id) {
    const state = states.get(id);
    return state?.active ? state.bookAnchors : null;
  }
  function getLifeAnchors(id) {
    const state = states.get(id);
    return state?.active ? state.lifeAnchors : null;
  }
  function getPoseDiagnostics() {
    return { capacity, active: [...states.values()].filter(s => s.active).length,
      drawCalls: meshes.filter(mesh => mesh.visible && mesh.count).length,
      maxDrawCalls: meshes.length, materials: materials.length,
      instances: meshes.reduce((total, mesh) => total + mesh.count, 0),
      actors: [...states.values()].filter(s => s.active).map(s => s.diagnostics) };
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    batches.forEach(batch => scene.remove(batch.group));
    Object.values(definitions).forEach(d => d.geometry.dispose()); materials.forEach(material => material.dispose());
    states.clear();
  }
  return { update, count: capacity, materials, fixedMaterials: batches[0].materials, movingMaterials: batches[1].materials,
    meshes, groups: batches.map(b => b.group), pickTargets: meshes, resolveHit, boundsForActor, getBookAnchors, getLifeAnchors, getPoseDiagnostics, dispose };
}
