import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { clamp, lerp, TAU } from './math.js';

/* Small, articulated woodland visitors. Metres; +Z is the direction of travel.
 * All anatomy is geometry, including paws, ear interiors, whiskers and feathers.
 * Instanced limbs keep the three species inexpensive without flat billboards.
 */
const UP = new THREE.Vector3(0, 1, 0);
const temp = new THREE.Object3D(), va = new THREE.Vector3(), vb = new THREE.Vector3();
const q = new THREE.Quaternion(), color = new THREE.Color();

// The people are deliberately small garden figures. Keep visitors in the same
// visual scale, and share this factor with navigation and physical contacts.
export const WILDLIFE_SCALE = Object.freeze({ cat: .72, rabbit: .70, sparrow: .74 });

export function createWildlifeModels(root, { quality = 'high' } = {}) {
  const geometries = new Set(), rigs = [];
  const sphere = new THREE.SphereGeometry(1, quality === 'low' ? 14 : 22, quality === 'low' ? 9 : 14);
  const cone = new THREE.ConeGeometry(1, 1, 3, 1);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 5);
  const pixels = new Uint8Array(64 * 64 * 4);
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
    const i = (y * 64 + x) * 4;
    const n = Math.sin(x * 13.91 + y * 41.37) * 43758.5453;
    const v = 226 + (n - Math.floor(n)) * 27 + Math.sin(y * 2.1 + Math.sin(x * .23)) * 2;
    pixels[i] = pixels[i + 1] = pixels[i + 2] = v; pixels[i + 3] = 255;
  }
  const fur = new THREE.DataTexture(pixels, 64, 64);
  fur.wrapS = fur.wrapT = THREE.RepeatWrapping;
  fur.repeat.set(3, 3); fur.colorSpace = THREE.SRGBColorSpace;
  fur.minFilter = THREE.LinearMipmapLinearFilter; fur.generateMipmaps = true; fur.needsUpdate = true;
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, map: fur, roughness: .82, metalness: 0 });
  material.name = '小动物 · 细绒';

  function part(g, c, pos, scale, rotation = [0, 0, 0]) {
    const result = g.clone();
    q.setFromEuler(new THREE.Euler(...rotation));
    result.applyMatrix4(new THREE.Matrix4().compose(new THREE.Vector3(...pos), q, new THREE.Vector3(...scale)));
    const p = result.getAttribute('position'), cols = new Float32Array(p.count * 3);
    if (typeof c !== 'function') color.set(c);
    for (let i = 0; i < p.count; i++) {
      if (typeof c === 'function') color.copy(c(p.getX(i), p.getY(i), p.getZ(i)));
      const grain = .965 + .035 * Math.sin(p.getX(i) * 287 + p.getY(i) * 431 + p.getZ(i) * 131);
      cols[i * 3] = color.r * grain; cols[i * 3 + 1] = color.g * grain; cols[i * 3 + 2] = color.b * grain;
    }
    result.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    return result;
  }
  const oval = (c, x, y, z, rx, ry, rz, rotation) => part(sphere, c, [x, y, z], [rx, ry, rz], rotation);
  function rod(c, a, b, radius = .003) {
    va.set(...a); vb.set(...b);
    temp.position.copy(va).add(vb).multiplyScalar(.5);
    temp.quaternion.setFromUnitVectors(UP, vb.sub(va).normalize());
    temp.scale.set(radius, va.distanceTo(new THREE.Vector3(...b)), radius);
    temp.updateMatrix();
    const result = cylinder.clone().applyMatrix4(temp.matrix);
    const colors = new Float32Array(result.attributes.position.count * 3); color.set(c);
    for (let i = 0; i < colors.length; i += 3) { colors[i] = color.r; colors[i + 1] = color.g; colors[i + 2] = color.b; }
    result.setAttribute('color', new THREE.BufferAttribute(colors, 3)); return result;
  }
  function mesh(parent, parts, name) {
    // Sphere, cone and line primitives share position/normal/uv/color attributes.
    const g = mergeGeometries(parts, false); parts.forEach(p => p.dispose());
    if (!g) throw new Error('Could not assemble animal anatomy: ' + name);
    geometries.add(g); g.computeBoundingSphere();
    const m = new THREE.Mesh(g, material); m.name = name; m.castShadow = m.receiveShadow = true; parent.add(m); return m;
  }
  function instances(parent, count, name) {
    // Sphere vertex colour is white; per-instance colours define fur and feet.
    const g = sphere.clone();
    const colors = new Float32Array(g.attributes.position.count * 3).fill(1);
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3)); geometries.add(g);
    const m = new THREE.InstancedMesh(g, material, count);
    m.name = name; m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    m.castShadow = m.receiveShadow = true; m.frustumCulled = false; parent.add(m); return m;
  }
  function instance(m, i, c, position, scale) {
    temp.position.set(...position); temp.rotation.set(0, 0, 0); temp.scale.set(...scale); temp.updateMatrix();
    m.setMatrixAt(i, temp.matrix); m.setColorAt(i, color.set(c));
  }
  function link(m, i, c, a, b, rx, rz = rx) {
    va.set(...a); vb.set(...b);
    const length = va.distanceTo(vb);
    temp.position.copy(va).add(vb).multiplyScalar(.5);
    temp.quaternion.setFromUnitVectors(UP, vb.sub(va).normalize());
    temp.scale.set(rx, length * .61, rz); temp.updateMatrix();
    m.setMatrixAt(i, temp.matrix); m.setColorAt(i, color.set(c));
  }
  function finishInstances(m) { m.instanceMatrix.needsUpdate = true; if (m.instanceColor) m.instanceColor.needsUpdate = true; }
  function knee(hip, foot, upper, lower, direction) {
    const delta = new THREE.Vector3(foot[0] - hip[0], foot[1] - hip[1], foot[2] - hip[2]);
    const distance = Math.max(.001, Math.min(delta.length(), upper + lower - .001));
    delta.normalize();
    const along = clamp((upper * upper - lower * lower + distance * distance) / (2 * distance), 0, upper);
    const normal = new THREE.Vector3(0, 0, direction).addScaledVector(delta, -delta.z * direction).normalize();
    return new THREE.Vector3(...hip).addScaledVector(delta, along).addScaledVector(normal, Math.sqrt(Math.max(0, upper * upper - along * along))).toArray();
  }
  function footWorld(actor, p) {
    const sn = Math.sin(actor.heading), cs = Math.cos(actor.heading), s = WILDLIFE_SCALE[actor.species];
    return { x: actor.x + (p[0] * cs + p[2] * sn) * s, y: actor.y + actor.lift + p[1] * s, z: actor.z + (-p[0] * sn + p[2] * cs) * s };
  }
  // A single, softly tapered surface defines each torso. Shoulder, belly and
  // rump share a silhouette instead of looking like independent stacked balls.
  function torso(profile, tint) {
    const rings = quality === 'low' ? 36 : 56, sides = quality === 'low' ? 16 : 24;
    const positions = [], colors = [], uv = [], index = [];
    for (let r = 0; r <= rings; r++) {
      const z = lerp(profile[0][0], profile.at(-1)[0], r / rings);
      let j = 0; while (j < profile.length - 2 && z > profile[j + 1][0]) j++;
      const u = clamp((z - profile[j][0]) / (profile[j + 1][0] - profile[j][0]));
      const p0 = profile[Math.max(0, j - 1)], p1 = profile[j], p2 = profile[j + 1], p3 = profile[Math.min(profile.length - 1, j + 2)];
      const smooth = k => Math.max(k === 2 ? -1 : .001, .5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * u + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u * u + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u * u * u));
      const w = smooth(1), y = smooth(2), h = smooth(3);
      for (let side = 0; side <= sides; side++) {
        const a = side / sides * TAU, x = Math.sin(a) * w, py = y + Math.cos(a) * h;
        positions.push(x, py, z); color.copy(tint(x, py, z, Math.cos(a))); colors.push(color.r, color.g, color.b); uv.push(side / sides, r / rings);
        if (r < rings && side < sides) { const v = r * (sides + 1) + side, b = v + sides + 1; index.push(v, b, v + 1, v + 1, b, b + 1); }
      }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.setIndex(index); g.computeVertexNormals(); return g;
  }
  function earGeometry(w, h, depth) {
    const shape = new THREE.Shape(); shape.moveTo(-w * .5, 0); shape.quadraticCurveTo(-w * .60, h * .48, -w * .1, h); shape.quadraticCurveTo(w * .26, h * .78, w * .5, 0); shape.quadraticCurveTo(0, -h * .13, -w * .5, 0);
    const g = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelSize: .005, bevelThickness: .004, bevelSegments: 2, curveSegments: 5 });
    g.setIndex(Array.from({ length: g.attributes.position.count }, (_, i) => i)); geometries.add(g); return g;
  }
  function look(clock, seed = 0) {
    // Long still looks interrupted by a gentle glance; no perpetual head shaking.
    const u = ((clock + seed * 1.37) % 8.2) / 8.2;
    if (u < .30) return 0;
    if (u < .47) return Math.sin((u - .30) / .17 * Math.PI / 2) * .22;
    if (u < .69) return .22;
    if (u < .88) return Math.cos((u - .69) / .19 * Math.PI / 2) * .22;
    return 0;
  }
  function blink(clock, seed = 0) {
    const at = (clock + seed * 1.83) % 5.7;
    return 1 - Math.pow(Math.max(0, 1 - Math.abs(at - 4.9) / .115), 1.2) * .94;
  }
  function actorBlend(actor, key, state) { return Number.isFinite(actor[key]) ? clamp(actor[key]) : actor.state === state ? 1 : 0; }
  function makeMammal(species, id) {
    const rabbit = species === 'rabbit', scale = WILDLIFE_SCALE[species];
    const group = new THREE.Group(); group.name = rabbit ? '林兔' : '小花猫'; group.scale.setScalar(scale); root.add(group);
    const body = new THREE.Group(), head = new THREE.Group(); group.add(body); body.add(head);
    const pivotY = rabbit ? .217 : .273; body.position.y = pivotY;
    const tan = rabbit ? 0x978571 : 0xaf8051, dark = rabbit ? 0x746754 : 0x775138;
    const cream = rabbit ? 0xcabca7 : 0xd9c5a8, pink = rabbit ? 0xa47e74 : 0xa97769;
    const coatBase = new THREE.Color(tan), coatDark = new THREE.Color(dark), belly = new THREE.Color(cream), coatColor = new THREE.Color();
    const coat = (x, y, z, facing = 1) => {
      const stripe = rabbit ? 0 : Math.pow(Math.max(0, Math.cos(z * 38 + Math.abs(x) * 7 + y * 9)), 7) * .26;
      return coatColor.copy(coatBase).lerp(coatDark, stripe * clamp((facing + .2) * 1.3)).lerp(belly, clamp((-facing - .28) * 1.7) * .72);
    };
    const profile = rabbit ? [
      [-.26, .003, -.015, .010], [-.235, .061, -.006, .089], [-.18, .108, .005, .137],
      [-.095, .119, .013, .157], [-.015, .111, .019, .154], [.06, .090, .017, .131],
      [.12, .065, .017, .097], [.165, .034, .009, .059], [.183, .003, .005, .015],
    ] : [
      [-.33, .010, -.009, .022], [-.276, .074, -.006, .083], [-.19, .082, .001, .097],
      [-.07, .073, -.008, .082], [.065, .075, -.003, .088], [.16, .081, .007, .098], [.241, .060, .015, .075], [.287, .014, .020, .035],
    ];
    const bodyMesh = mesh(body, [torso(profile, coat), ...(rabbit ? [oval(cream, 0, .006, -.257, .042, .043, .038)] : [])], species + ' · 连续躯干');
    const bodyRest = bodyMesh.geometry.attributes.position.array.slice(), bodyWeights = new Float32Array(bodyRest.length);
    if (!rabbit) for (let i = 0; i < bodyRest.length; i += 3) { const z = bodyRest[i + 2]; bodyWeights[i] = Math.exp(-Math.pow((z - .15) / .10, 2)); bodyWeights[i + 1] = Math.exp(-Math.pow((z + .21) / .11, 2)); bodyWeights[i + 2] = Math.sin(z / .34 * Math.PI * .7); }
    const baseHeadY = rabbit ? .275 : .328, baseHeadZ = rabbit ? .175 : .292;
    head.position.set(0, baseHeadY - pivotY, baseHeadZ);
    const face = [
      oval(tan, 0, 0, -.003, rabbit ? .074 : .085, rabbit ? .079 : .073, rabbit ? .087 : .074, [rabbit ? -.17 : .04, 0, 0]),
      oval(cream, 0, -.026, .048, rabbit ? .048 : .059, rabbit ? .036 : .032, rabbit ? .053 : .035),
      oval(cream, -.019, -.027, rabbit ? .076 : .066, .026, .022, .025),
      oval(cream, .019, -.027, rabbit ? .076 : .066, .026, .022, .025),
      oval(pink, 0, -.021, rabbit ? .104 : .090, .009, .006, .006),
      rod(0x655145, [0, -.027, rabbit ? .107 : .093], [0, -.037, rabbit ? .103 : .090], .0015),
    ];
    // Small side-set eyes and an actual triangular pinna replace heavy brows,
    // oversized staring eyes and horn-like cones.
    for (const side of [-1, 1]) for (let i = 0; i < 3; i++) face.push(rod(0xdbd0bd, [side * .025, -.028 + i * .006, rabbit ? .094 : .081], [side * (rabbit ? .093 : .127), -.037 + i * .012, rabbit ? .110 : .098], .00065));
    mesh(head, face, species + ' · 小脸');
    const eyes = [];
    for (const side of [-1, 1]) {
      const eye = new THREE.Group(); eye.position.set(side * (rabbit ? .060 : .056), .008, rabbit ? .044 : .053); eye.rotation.y = side * (rabbit ? .62 : .30); head.add(eye);
      mesh(eye, [oval(rabbit ? 0x211e19 : 0x868257, 0, 0, 0, rabbit ? .0095 : .012, rabbit ? .011 : .010, .0055), ...(rabbit ? [] : [oval(0x202319, 0, 0, .0055, .003, .008, .0015)]), oval(0xf3ead8, -.002, .0035, .0065, .0016, .0018, .001)], species + ' · 眼睛'); eyes.push(eye);
    }
    const ears = [];
    const catEar = rabbit ? null : earGeometry(.072, .088, .011), catInner = rabbit ? null : earGeometry(.038, .057, .002);
    for (const side of [-1, 1]) {
      const ear = new THREE.Group(); ear.position.set(side * (rabbit ? .039 : .061), rabbit ? .054 : .044, rabbit ? -.024 : -.019); head.add(ear);
      if (rabbit) mesh(ear, [
        oval(tan, 0, .098, -.010, .025, .118, .018, [.18, 0, -side * .13]),
        oval(pink, 0, .105, .005, .013, .088, .0055, [.18, 0, -side * .13]),
      ], '林兔 · 柔软长耳');
      else mesh(ear, [part(catEar, tan, [0, 0, 0], [1, 1, 1], [-.20, side * .2, -side * .24]), part(catInner, pink, [0, .013, .015], [1, 1, 1], [-.20, side * .2, -side * .24])], '小猫 · 薄耳廓');
      ears.push(ear);
    }
    const limb = instances(group, 16, species + ' · 四足与脚掌');
    const tailRings = 22, tailSides = quality === 'low' ? 6 : 9;
    let tailMesh = null;
    if (!rabbit) {
      const g = new THREE.BufferGeometry(), positions = new Float32Array((tailRings + 1) * (tailSides + 1) * 3), colors = new Float32Array(positions.length), uv = new Float32Array((tailRings + 1) * (tailSides + 1) * 2), index = [];
      for (let ring = 0; ring <= tailRings; ring++) for (let side = 0; side <= tailSides; side++) {
        const at = (ring * (tailSides + 1) + side) * 3, t = ring / tailRings;
        color.copy(coatBase).lerp(coatDark, Math.pow(Math.max(0, Math.cos(t * 30)), 5) * .45);
        colors[at] = color.r; colors[at + 1] = color.g; colors[at + 2] = color.b; const uvAt = at / 3 * 2; uv[uvAt] = side / tailSides; uv[uvAt + 1] = t;
        if (ring < tailRings && side < tailSides) { const a = ring * (tailSides + 1) + side, b = a + tailSides + 1; index.push(a, a + 1, b, a + 1, b + 1, b); }
      }
      g.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)); g.setAttribute('color', new THREE.BufferAttribute(colors, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); g.setIndex(index); geometries.add(g);
      tailMesh = new THREE.Mesh(g, material); tailMesh.name = '花猫 · 松弛弯尾'; tailMesh.castShadow = tailMesh.receiveShadow = true; tailMesh.frustumCulled = false; group.add(tailMesh);
    }
    const live = { feet: [], paw: null, minFootY: 0, gait: 'standing', head: null };
    function update(actor, groundAt) {
      group.visible = actor.active; group.position.set(actor.x, actor.y + actor.lift, actor.z); group.rotation.y = actor.heading; if (!actor.active) return;
      const move = clamp(actor.speed / ((rabbit ? .62 : .55) * scale)), clock = actor.time, action = actorBlend(actor, 'actionBlend', 'scratch'), inspect = actorBlend(actor, 'inspectBlend', 'inspect'), alert = actorBlend(actor, 'alertBlend', 'alert');
      const phase = actor.phase % 1, breath = Math.sin(clock * (rabbit ? 3.1 : 2.1) + id) * .003;
      const hopU = clamp((phase - .24) / .65), air = Math.sin(hopU * Math.PI), crouch = rabbit ? (Math.max(0, 1 - Math.abs(phase - .15) / .16) + Math.max(0, 1 - Math.abs(phase - .93) / .08)) * .017 * move : 0;
      const bob = rabbit ? -crouch : -Math.cos(phase * TAU * 2) * .0025 * move;
      body.position.set(action * (rabbit ? 0 : .012), pivotY + bob + breath - action * .016, 0);
      body.rotation.x = rabbit ? (-.10 * Math.sin(hopU * TAU) * move + action * .08) : action * .10 + .018 * Math.sin(phase * TAU * 2) * move;
      body.rotation.z = rabbit ? 0 : .018 * Math.sin(phase * TAU) * move - action * .045;
      body.scale.set(1 + breath * .9, 1 + breath * .55, 1);
      if (!rabbit) {
        const positions = bodyMesh.geometry.attributes.position.array, swing = Math.sin(phase * TAU) * move, opposite = Math.sin(phase * TAU + 1.1) * move;
        for (let i = 0; i < positions.length; i += 3) { positions[i] = bodyRest[i] + bodyWeights[i + 2] * swing * .007; positions[i + 1] = bodyRest[i + 1] + bodyWeights[i] * opposite * .005 + bodyWeights[i + 1] * swing * .004; }
        bodyMesh.geometry.attributes.position.needsUpdate = true;
      }
      const sniff = inspect * (.11 + Math.sin(clock * 2.2) * .035);
      head.position.set(0, baseHeadY - pivotY - action * (rabbit ? .017 : .025), baseHeadZ + action * .008);
      head.rotation.x = action * (rabbit ? .30 : .40) + sniff - alert * .12;
      head.rotation.y = look(clock, id) * (1 - action) * (1 - move * .8);
      head.rotation.z = look(clock + 2.1, id) * (rabbit ? .10 : .16) * (1 - action) * (1 - move);
      for (let i = 0; i < 2; i++) {
        eyes[i].scale.y = blink(clock, id);
        const twitchAt = (clock + i * 1.67 + id * .9) % (rabbit ? 4.8 : 6.3);
        const twitch = Math.sin(clamp((twitchAt - 4.1) / .35) * TAU) * (twitchAt > 4.1 && twitchAt < 4.45 ? 1 : 0);
        ears[i].rotation.z = (i ? -1 : 1) * (rabbit ? .12 : .06) + twitch * (rabbit ? .12 : .075);
        ears[i].rotation.x = rabbit ? -.15 + (i ? .08 : -.12) - air * .22 * move + twitch * .12 - alert * .18 : -.02 + twitch * .07 - alert * .22;
      }
      live.feet.length = 0;
      for (let i = 0; i < 4; i++) {
        const front = i < 2, side = i % 2 ? 1 : -1;
        const restZ = rabbit ? front ? .138 : -.175 : front ? .182 : -.244;
        const hipY = rabbit ? front ? .173 : .221 : front ? .267 : .268;
        const x = side * (rabbit ? front ? .047 : .068 : .060);
        let stepZ = 0, footLift = 0;
        if (rabbit) {
          // Hind feet push together. Forefeet reach first during descent;
          // the long rear paws swing underneath before the next compression.
          stepZ = (front ? -.035 * Math.sin(hopU * Math.PI) + .032 * Math.sin(hopU * TAU) : -.058 * Math.sin(hopU * TAU)) * move;
          footLift = actor.lift > .035 * scale ? (front ? .021 : .038) * air : 0;
        } else {
          const p = (phase + [0, .5, .76, .26][i]) % 1, stance = .66, stride = .26 * move;
          if (p < stance) stepZ = (.5 - p / stance) * stride;
          else { const u = (p - stance) / (1 - stance); stepZ = (-.5 + u * u * (3 - 2 * u)) * stride; footLift = Math.sin(u * Math.PI) * .050 * move; }
        }
        if (front && (rabbit || i === 0)) {
          const p = ((actor.actionTime ?? actor.stateTime) * 8.6 / TAU) % 1;
          let drag = 0, raise = 0;
          if (p < .62) drag = .083 - p / .62 * .13;
          else { const u = (p - .62) / .38; drag = -.047 + u * u * (3 - 2 * u) * .13; raise = Math.sin(u * Math.PI) * .039; }
          stepZ = lerp(stepZ, drag, action); footLift = lerp(footLift, raise, action);
        }
        const foot = [x, .018 + footLift, restZ + stepZ];
        if (groundAt) { const world = footWorld(actor, foot), t = clamp(1 - actor.lift / (.05 * scale)), plant = t * t * (3 - 2 * t); foot[1] += (clamp(groundAt(world.x, world.z) - actor.y, -.05 * scale, .08 * scale) - actor.lift) * plant / scale; }
        const hip = [x + body.position.x, hipY + bob + breath - action * .012, restZ + (front ? -.008 : .035)];
        const k = knee(hip, foot, rabbit ? front ? .087 : .126 : front ? .143 : .151, rabbit ? front ? .103 : .138 : front ? .150 : .157, front ? -1 : 1);
        // A covered upper joint and a narrow lower limb avoid the stick-legged,
        // straight-kneed mannequin look while retaining grounded contact points.
        link(limb, i * 4, tan, hip, k, rabbit ? front ? .018 : .033 : front ? .022 : .027, rabbit ? front ? .020 : .038 : .024);
        const ankle = [lerp(k[0], foot[0], .78), lerp(k[1], foot[1], .82), lerp(k[2], foot[2], .80)];
        link(limb, i * 4 + 1, tan, k, ankle, rabbit ? .018 : .016, rabbit ? .020 : .016);
        link(limb, i * 4 + 2, tan, ankle, foot, rabbit ? .017 : .014, rabbit ? .020 : .015);
        instance(limb, i * 4 + 3, rabbit ? tan : cream, [foot[0], foot[1] + .004, foot[2] + (rabbit && !front ? .023 : .014)], rabbit && !front ? [.028, .020, .068] : rabbit ? [.018, .018, .035] : [.023, .018, .035]);
        live.feet.push(footWorld(actor, foot));
      }
      if (!rabbit) {
        // A loose, tapered S curve follows the rump. The tip leads and settles
        // at different rates along the tail instead of waving a rigid rod.
        const center = t => new THREE.Vector3(Math.sin(t * 2.6) * .027 + Math.sin(clock * .80 - t * 2.7) * .038 * t * t,
          .275 + bob + breath - Math.sin(t * Math.PI) * (.135 - alert * .12) + t * t * (.155 + Math.sin(clock * .78 - t * 1.7) * .018) - action * .012,
          -.319 - t * .375 + t ** 4 * (.060 + Math.sin(clock * .65) * .014));
        const attribute = tailMesh.geometry.attributes.position;
        for (let ring = 0; ring <= tailRings; ring++) {
          const t = ring / tailRings, p = center(t), tangent = center(t + .002).sub(center(t - .002)).normalize();
          const side = new THREE.Vector3(1, 0, 0).addScaledVector(tangent, -tangent.x).normalize(), normal = new THREE.Vector3().crossVectors(tangent, side).normalize();
          const radius = .021 * (1 - t * .76) * (ring === tailRings ? .18 : 1);
          for (let j = 0; j <= tailSides; j++) { const angle = j / tailSides * TAU, v = p.clone().addScaledVector(side, Math.cos(angle) * radius).addScaledVector(normal, Math.sin(angle) * radius); attribute.setXYZ(ring * (tailSides + 1) + j, v.x, v.y, v.z); }
        }
        attribute.needsUpdate = true; tailMesh.geometry.computeVertexNormals();
      }
      head.updateWorldMatrix(true, false); const hp = head.getWorldPosition(new THREE.Vector3()); live.head = { x: hp.x, y: hp.y, z: hp.z };
      live.paw = live.feet[0]; live.minFootY = Math.min(...live.feet.map(p => p.y)); live.gait = actor.lift > .035 * scale ? 'airborne' : move > .1 ? 'walking' : action > .2 ? 'pawing' : 'standing'; finishInstances(limb);
    }
    const rig = { group, update, live, species }; rigs.push(rig); return rig;
  }
  function makeSparrow(id) {
    const scale = WILDLIFE_SCALE.sparrow;
    const group = new THREE.Group(); group.name = '麻雀'; group.scale.setScalar(scale); root.add(group);
    const body = new THREE.Group(), head = new THREE.Group(); group.add(body); body.add(head); body.position.y = .112;
    const brown = id % 2 ? 0x88745a : 0x927a5c, chest = 0xb9a98e, dark = 0x554b3e;
    const cb = new THREE.Color(brown), cc = new THREE.Color(chest), cd = new THREE.Color(dark), mixed = new THREE.Color();
    mesh(body, [torso([[-.123, .004, -.012, .008], [-.080, .041, .003, .048], [-.020, .061, 0, .068], [.040, .049, -.001, .061], [.076, .033, .001, .043], [.105, .001, -.003, .002]], (x, y, z, facing) => mixed.copy(cb).lerp(cc, clamp((-facing + .42) * 1.1))),], '麻雀 · 连续羽衣');
    head.position.set(0, .064, .067);
    const face = [oval(brown, 0, 0, -.005, .043, .042, .044), oval(dark, 0, -.024, .025, .013, .008, .010), part(cone, 0x77634b, [0, -.006, .051], [.017, .035, .014], [Math.PI / 2, Math.PI / 2, 0])];
    for (const side of [-1, 1]) { face.push(oval(chest, side * .030, -.010, .010, .012, .015, .023)); face.push(oval(dark, side * .041, -.011, .006, .0015, .006, .006)); }
    mesh(head, face, '麻雀 · 栗色头颊与短喙');
    const eyes = [];
    for (const side of [-1, 1]) {
      const eye = new THREE.Group(); eye.position.set(side * .035, .008, .020); eye.rotation.y = side * .52; head.add(eye);
      mesh(eye, [oval(0x202018, 0, 0, 0, .0055, .0058, .0035), oval(0xe7dcc8, -.001, .002, .003, .0011, .0012, .0007)], '麻雀 · 小眼睛'); eyes.push(eye);
    }
    const wings = [];
    for (const side of [-1, 1]) {
      const wing = new THREE.Group(); wing.position.set(side * .048, .011, -.033); body.add(wing);
      const wingColor = (x, y, z) => mixed.copy(cb).lerp(cc, (Math.exp(-Math.pow((z + y * .75 + .038) / .006, 2)) + Math.exp(-Math.pow((z + y * .75 + .065) / .005, 2))) * .38);
      const feathers = [oval(wingColor, side * .007, -.003, -.022, .014, .032, .072, [.29, 0, side * -.13])];
      for (let i = 0; i < 4; i++) feathers.push(oval(i % 2 ? dark : brown, side * .009, -.007 - i * .003, -.052 - i * .009, .008, .012, .048, [.25, side * .025 * i, 0]));
      mesh(wing, feathers, '麻雀 · 合拢飞羽'); wings.push(wing);
    }
    const tail = new THREE.Group(); tail.position.set(0, .002, -.102); body.add(tail);
    mesh(tail, [-1, 0, 1].map(i => oval(i ? dark : brown, i * .011, 0, -.039, .010, .005, .059, [-.06, i * -.06, 0])), '麻雀 · 细尾羽');
    const limbs = instances(group, 10, '麻雀 · 细爪'); const live = { feet: [], paw: null, gait: 'standing' };
    function update(actor, groundAt) {
      group.visible = actor.active; group.position.set(actor.x, actor.y + actor.lift, actor.z); group.rotation.y = actor.heading; if (!actor.active) return;
      const flying = actor.flying, flight = Number.isFinite(actor.flightBlend) ? actor.flightBlend : flying ? 1 : 0, action = actorBlend(actor, 'actionBlend', 'scratch'), inspect = actorBlend(actor, 'inspectBlend', 'inspect'), alert = actorBlend(actor, 'alertBlend', 'alert');
      const peck = Math.pow(Math.max(0, Math.sin((actor.actionTime ?? actor.stateTime) * 8.2)), 2) * action;
      const move = clamp(actor.speed / (.6 * scale)), compress = !flying ? Math.max(0, 1 - Math.abs(actor.phase - .16) / .14) * .009 * move : 0;
      body.position.y = .112 - peck * .028 - compress + Math.sin(actor.time * 4.1 + id) * .0008;
      body.rotation.x = lerp(peck * .75 - alert * .10, .10, flight);
      head.position.set(0, .064 - peck * .028, .067);
      head.rotation.x = peck * .95;
      const glance = look(actor.time * 1.85, id);
      head.rotation.y = glance * 1.9 * (1 - peck) * (1 - move * .6);
      head.rotation.z = inspect * glance * -.45;
      for (const eye of eyes) eye.scale.y = blink(actor.time * 1.13, id);
      tail.rotation.x = -.06 + peck * .28 - alert * .15 + Math.sin(actor.time * 1.4 + id) * .025;
      for (let i = 0; i < 2; i++) {
        wings[i].rotation.z = (i ? -1 : 1) * lerp(.018, 1.28 + Math.sin(actor.time * 42 + id) * .96, flight);
        wings[i].rotation.x = -.21 * flight;
      }
      live.feet.length = 0;
      for (let side = 0; side < 2; side++) {
        const x = side ? .024 : -.024, tuck = flying ? 1 : clamp(actor.lift / (.070 * scale));
        const foot = [x, .005 + tuck * .025, .022 - tuck * .027];
        if (!flying && groundAt) { const world = footWorld(actor, foot), t = clamp(1 - actor.lift / (.04 * scale)), plant = t * t * (3 - 2 * t); foot[1] += (clamp(groundAt(world.x, world.z) - actor.y, -.04 * scale, .05 * scale) - actor.lift) * plant / scale; }
        const hip = [x, .086 - peck * .020 - compress, -.004], ankle = [x, lerp(.041, .065, tuck) - peck * .014, .005 - tuck * .020];
        link(limbs, side * 5, 0x8b6b51, hip, ankle, .0048); link(limbs, side * 5 + 1, 0x997457, ankle, foot, .004);
        for (let toe = 0; toe < 3; toe++) link(limbs, side * 5 + 2 + toe, 0x997457, foot, [x + (toe - 1) * .012, foot[1], foot[2] + .019 - Math.abs(toe - 1) * .007], .0024);
        live.feet.push(footWorld(actor, foot));
      }
      head.updateWorldMatrix(true, false); const beak = new THREE.Vector3(0, -.006, .0685).applyMatrix4(head.matrixWorld); live.paw = { x: beak.x, y: beak.y, z: beak.z }; live.gait = flying ? 'flying' : actor.lift > .02 * scale ? 'hopping' : action > .2 ? 'pecking' : 'standing'; finishInstances(limbs);
    }
    const rig = { group, update, live, species: 'sparrow' }; rigs.push(rig); return rig;
  }
  function create(species, id = 0) { return species === 'sparrow' ? makeSparrow(id) : makeMammal(species, id); }
  function dispose() {
    for (const rig of rigs) rig.group.removeFromParent(); for (const g of geometries) g.dispose();
    material.dispose(); fur.dispose(); sphere.dispose(); cone.dispose(); cylinder.dispose();
  }
  return { create, dispose, rigs };
}
