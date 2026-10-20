import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { clamp, lerp, rng, wrapAngle } from './math.js';

/*
 * Three generations, one shared garden. Everything is actual geometry.
 * +Z is forward; toolX/toolZ locate the working edge, independently of the body.
 * Poses own time and task state. This module never moves, clears, or redraws sand.
 */
const TAU = Math.PI * 2;
const UP = new THREE.Vector3(0, 1, 0);
export const FAMILY_PROFILES = Object.freeze([
  { id: 0, name: '阿婆', height: 1.08, cloth: 0x9b8ca2, trousers: 0x66675d, trim: 0xd5c7b4, skin: 0xc89e7f, hair: 0xb5b6a9, lean: .022, girth: 1.05, style: 'grandmother', headwear: 'scarf', rakeWidth: .16, smoothWidth: .38, reach: .47 },
  { id: 1, name: '妈妈', height: 1.135, cloth: 0x81947b, trousers: 0x66705e, trim: 0xe0cda4, skin: 0xd3aa88, hair: 0x414238, lean: 0, girth: 1.02, style: 'mother', headwear: 'none', rakeWidth: .16, smoothWidth: .38, reach: .47 },
  { id: 2, name: '爸爸', height: 1.22, cloth: 0xad9573, trousers: 0x67705d, trim: 0xd9c7a1, skin: 0xc79a78, hair: 0x47473b, lean: .006, girth: 1.09, style: 'father', headwear: 'straw', rakeWidth: .16, smoothWidth: .38, reach: .47 },
  { id: 3, name: '姐姐', height: 1.165, cloth: 0x7897a6, trousers: 0x637581, trim: 0xd3d0b9, skin: 0xd6ad8d, hair: 0x424338, lean: 0, girth: .98, style: 'sister', headwear: 'none', rakeWidth: .16, smoothWidth: .38, reach: .47 },
  { id: 4, name: '妹妹', height: .80, cloth: 0xbb7568, trousers: 0x5b6755, trim: 0xe2c99f, skin: 0xcaa080, hair: 0x38322a, lean: 0, girth: .91, style: 'littleSister', headwear: 'none', child: true, rakeWidth: .12, smoothWidth: .285, reach: .34 },
  { id: 5, name: '弟弟', height: .91, cloth: 0xc2a064, trousers: 0x456c79, trim: 0xbfc4af, skin: 0xc39676, hair: 0x39392d, lean: .005, girth: .94, style: 'littleBrother', headwear: 'cap', child: true, rakeWidth: .12, smoothWidth: .285, reach: .36 },
].map(profile => Object.freeze(profile)));
export const FAMILY_COUNT = FAMILY_PROFILES.length;

function proportions(profile) {
  const younger = profile.id === 4, older = profile.id === 5;
  const hip = younger ? .398 : older ? .415 : .455;
  const torso = younger ? .90 : older ? .94 : 1;
  const headScale = younger ? 1.43 : older ? 1.34 : 1.24;
  const headY = .370 * torso + .077 * headScale;
  // Soft, deliberately illustrated proportions: shorter legs and a readable
  // face, instead of a tiny realistic head on a long, rigid mannequin body.
  const headTop = profile.headwear === 'straw' ? .158 : profile.headwear === 'cap' ? .103 : .088;
  const modelHeight = hip + headY + headTop * headScale;
  return {
    hipBase: hip, torsoScale: torso, shoulderY: .309 * torso, headY, headScale,
    legA: younger ? .182 : older ? .190 : .211,
    legB: younger ? .184 : older ? .192 : .214,
    armA: younger ? .171 : older ? .181 : .197,
    armB: younger ? .168 : older ? .178 : .192,
    footSpread: younger ? .060 : older ? .062 : .066,
    toolScale: profile.rakeWidth / .16,
    scale: profile.height / modelHeight,
  };
}

function makeTexture(size, fn, color = true) {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const c = fn(x, y), i = (y * size + x) * 4;
    data[i] = c[0]; data[i + 1] = c[1]; data[i + 2] = c[2]; data[i + 3] = 255;
  }
  const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  t.colorSpace = color ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

function makeTextures() {
  const random = rng(78231);
  const cloth = makeTexture(128, (x, y) => {
    const thread = ((x & 3) === 0 ? -8 : 2) + ((y & 3) === 0 ? -6 : 2);
    const value = clamp(236 + thread + (random() - .5) * 9, 0, 255);
    return [value, value, value];
  });
  cloth.repeat.set(4, 3);
  const clothNormal = makeTexture(128, (x, y) => [128 + Math.sin(x * Math.PI / 2) * 10, 128 + Math.sin(y * Math.PI / 2) * 10, 253], false);
  clothNormal.repeat.copy(cloth.repeat);
  const straw = makeTexture(256, (x, y) => {
    const dx = x - 127.5, dy = y - 127.5, r = Math.hypot(dx, dy), a = Math.atan2(dy, dx);
    const ring = Math.sin(r * 2.0 + .45 * Math.sin(a * 13));
    const spoke = Math.sin(a * 112 + Math.sin(r * .24));
    const over = (ring > .15) === (spoke > 0) ? 4 : -4;
    const value = 225 + ring * 9 + spoke * 5 + over + (random() - .5) * 8;
    return [clamp(value, 0, 255), clamp(value - 12, 0, 255), clamp(value - 35, 0, 255)];
  });
  straw.wrapS = straw.wrapT = THREE.ClampToEdgeWrapping;
  const wood = makeTexture(128, (x, y) => {
    const grain = Math.sin(x * .69 + Math.sin(y * .075) * 1.6) * 8 + Math.sin(x * 2.4 + y * .024) * 3;
    const v = 225 + grain + (random() - .5) * 8;
    return [v, v - 9, v - 20];
  });
  wood.repeat.set(2, 3);
  return { cloth, clothNormal, straw, wood };
}

function transformed(geometry, position = [0, 0, 0], scale = [1, 1, 1], rotation = null) {
  const g = geometry.clone();
  const q = rotation ? new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation)) : new THREE.Quaternion();
  g.applyMatrix4(new THREE.Matrix4().compose(new THREE.Vector3(...position), q, new THREE.Vector3(...scale)));
  return g;
}

function merge(parts) {
  const geometry = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  if (!geometry) throw new Error('Family geometry could not be merged.');
  geometry.computeBoundingSphere();
  return geometry;
}

function makeRingBody(rows, columns, shape, fold = .0025) {
  const position = [], uv = [], index = [];
  for (let j = 0; j <= rows; j++) {
    const v = j / rows, [y, rx, rz] = shape(v);
    for (let i = 0; i <= columns; i++) {
      const u = i / columns, a = u * TAU;
      const dr = fold * Math.sin(a * 11 + v * 6) * Math.sin(v * Math.PI) + fold * .45 * Math.sin(a * 19 - v * 15);
      position.push(Math.cos(a) * (rx + dr), y, Math.sin(a) * (rz + dr));
      uv.push(u, v);
      if (j < rows && i < columns) {
        const k = j * (columns + 1) + i;
        index.push(k, k + columns + 1, k + 1, k + 1, k + columns + 1, k + columns + 2);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(index); g.computeVertexNormals();
  return g;
}

function sampleProfile(profile, t) {
  const p = t * (profile.length - 1), i = Math.min(profile.length - 2, Math.floor(p)), f = p - i;
  const u = f * f * (3 - 2 * f);
  return [lerp(profile[i][0], profile[i + 1][0], u), lerp(profile[i][1], profile[i + 1][1], u)];
}

const torsoProfile = [[.137, .086], [.146, .093], [.139, .090], [.137, .091], [.145, .093], [.153, .080], [.055, .048]];
function makeTorso(girth) {
  return makeRingBody(32, 32, v => { const [x, z] = sampleProfile(torsoProfile, v); return [v * .370, x * girth, z]; }, .0018);
}

function drapeOverTorso(geometry, girth) {
  const p = geometry.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    if (y < 0 || y > .370 || z < 0) continue;
    const [rx, rz] = sampleProfile(torsoProfile, y / .370);
    const front = rz * Math.sqrt(Math.max(0, 1 - (x / (rx * girth)) ** 2));
    p.setZ(i, Math.max(z, front + .0045));
  }
  p.needsUpdate = true; geometry.computeVertexNormals(); geometry.computeBoundingSphere();
  return geometry;
}

function makeLimb(radiusStart, radiusEnd, pleats = .0016) {
  return makeRingBody(10, 16, v => {
    const rounded = .76 + .24 * Math.sin(Math.PI * clamp(v * 2.6, 0, 1));
    const r = lerp(radiusStart, radiusEnd, v) * rounded;
    return [v, r, r * .91];
  }, pleats);
}

function tube(points, radius, tubularSegments = 16, radialSegments = 5) {
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p))), tubularSegments, radius, radialSegments, false);
}

function ribbon(points, widths, depth = .002) {
  const samples = [], sampledWidths = [];
  for (let i = 0; i < points.length - 1; i++) for (let j = 0; j < 6; j++) {
    const t = j / 6;
    samples.push(points[i].map((v, axis) => lerp(v, points[i + 1][axis], t)));
    sampledWidths.push(lerp(widths[i], widths[i + 1], t));
  }
  samples.push(points.at(-1)); sampledWidths.push(widths.at(-1));
  points = samples; widths = sampledWidths;
  const p = [], uv = [], indices = [];
  for (let i = 0; i < points.length; i++) {
    const prev = points[Math.max(0, i - 1)], next = points[Math.min(points.length - 1, i + 1)];
    const dx = next[0] - prev[0], dy = next[1] - prev[1], inv = 1 / Math.max(.0001, Math.hypot(dx, dy));
    const nx = -dy * inv * widths[i] * .5, ny = dx * inv * widths[i] * .5, [x, y, z] = points[i];
    p.push(x + nx, y + ny, z + depth, x - nx, y - ny, z + depth);
    uv.push(0, i / points.length, 1, i / points.length);
    if (i) { const k = i * 2; indices.push(k - 2, k - 1, k, k - 1, k + 1, k); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(indices); g.computeVertexNormals();
  return g;
}

function apronGeometry(child = false) {
  const positions = [], uv = [], indices = [], rows = 12, columns = 12;
  for (let j = 0; j <= rows; j++) {
    const v = j / rows, width = v < .60 ? lerp(.228, .168, v / .60) : lerp(.168, .082, (v - .60) / .40);
    const y = (child ? .010 : -.080) + v * (child ? .278 : .375);
    for (let i = 0; i <= columns; i++) {
      const u = i / columns, x = (u - .5) * width;
      const z = .098 - .024 * Math.abs(u - .5) * 2 + .0025 * Math.sin(u * 17 + v * 4) * (1 - v);
      positions.push(x, y, z); uv.push(u, v);
      if (i < columns && j < rows) {
        const k = j * (columns + 1) + i;
        indices.push(k, k + 1, k + columns + 1, k + 1, k + columns + 2, k + columns + 1);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(indices); g.computeVertexNormals();
  return g;
}

function makeHat(underside = false) {
  const positions = [], normals = [], uvs = [], indices = [];
  const segments = 72, rows = 12;
  for (let j = 0; j <= rows; j++) {
    const t = j / rows, r = .211 * t;
    for (let i = 0; i <= segments; i++) {
      const a = TAU * i / segments;
      const rib = Math.cos(a * 36) * .0008 * t;
      const y = .083 * (1 - Math.pow(t, .86)) + rib - (underside ? .006 : 0);
      positions.push(Math.cos(a) * r, y, Math.sin(a) * r);
      normals.push(0, underside ? -1 : 1, 0);
      uvs.push(.5 + Math.cos(a) * t * .5, .5 + Math.sin(a) * t * .5);
      if (j < rows && i < segments) {
        const k = j * (segments + 1) + i, n = k + segments + 1;
        if (underside) indices.push(k, n, k + 1, k + 1, n, n + 1);
        else indices.push(k, k + 1, n, k + 1, n + 1, n);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(indices); g.computeVertexNormals();
  return g;
}

function solveJoint(start, end, lengthA, lengthB, pole, out) {
  const direction = end.clone().sub(start), distance = Math.max(.00001, direction.length());
  direction.multiplyScalar(1 / distance);
  // A tiny amount of cloth-covered extension is preferable to a floating hand.
  const stretch = Math.max(1, distance / (lengthA + lengthB) * 1.00002);
  const a = lengthA * stretch, b = lengthB * stretch;
  const along = clamp((a * a - b * b + distance * distance) / (2 * distance), -a, a);
  const height = Math.sqrt(Math.max(0, a * a - along * along));
  const bend = pole.clone().sub(start);
  bend.addScaledVector(direction, -bend.dot(direction));
  if (bend.lengthSq() < .000001) bend.set(0, 0, 1).addScaledVector(direction, -direction.z);
  bend.normalize();
  out.copy(start).addScaledVector(direction, along).addScaledVector(bend, height);
  return stretch;
}

function connect(mesh, a, b) {
  const delta = b.clone().sub(a), length = Math.max(.00001, delta.length());
  mesh.position.copy(a);
  mesh.quaternion.setFromUnitVectors(UP, delta.multiplyScalar(1 / length));
  mesh.scale.y = length;
}

function worldToLocalPoint(world, rig, out = new THREE.Vector3()) {
  const dx = world.x - rig.x, dz = world.z - rig.z, c = Math.cos(rig.heading), s = Math.sin(rig.heading);
  return out.set((c * dx - s * dz) / rig.scale, (world.y - rig.groundY) / rig.scale, (s * dx + c * dz) / rig.scale);
}

function localToWorldPoint(local, rig, out = new THREE.Vector3()) {
  const c = Math.cos(rig.heading), s = Math.sin(rig.heading), x = local.x * rig.scale, z = local.z * rig.scale;
  return out.set(rig.x + c * x + s * z, rig.groundY + local.y * rig.scale, rig.z - s * x + c * z);
}

export function createFamily(scene) {
  const textures = makeTextures(), geometries = new Set(), materials = new Set();
  const jointMatrix = new THREE.Matrix4(), jointRotation = new THREE.Quaternion();
  const kneeCoverScale = new THREE.Vector3(.047, .042, .044), elbowCoverScale = new THREE.Vector3(.041, .038, .037);
  const root = new THREE.Group(); root.name = '庭院一家'; scene.add(root);
  const geometry = g => { geometries.add(g); return g; };
  const material = p => { const m = new THREE.MeshStandardMaterial(p); materials.add(m); return m; };
  const mesh = (g, m, parent, name, shadows = true) => {
    const node = new THREE.Mesh(g, m); node.name = name || ''; node.castShadow = shadows; node.receiveShadow = true; parent.add(node); return node;
  };
  const sphere = new THREE.SphereGeometry(1, 16, 12);
  const tinySphere = new THREE.SphereGeometry(1, 10, 8);
  const cube = new THREE.BoxGeometry(1, 1, 1);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 12);
  const hatTop = geometry(makeHat());
  const hatBottom = geometry(makeHat(true));
  const hatRim = geometry(transformed(new THREE.TorusGeometry(.210, .0035, 5, 72), [0, -.001, 0], [1, 1, 1], [Math.PI / 2, 0, 0]));
  const hatTie = geometry(merge([
    tube([[-.115, -.001, .01], [-.082, -.065, .031], [-.029, -.127, .038], [0, -.134, .04]], .0018),
    tube([[.115, -.001, .01], [.082, -.065, .031], [.029, -.127, .038], [0, -.134, .04]], .0018),
  ]));
  const faceGeometry = geometry(merge([
    transformed(sphere, [0, 0, .001], [.063, .077, .057]),
    transformed(tinySphere, [-.060, -.004, 0], [.010, .020, .012]),
    transformed(tinySphere, [.060, -.004, 0], [.010, .020, .012]),
    transformed(tinySphere, [0, -.008, .056], [.009, .010, .011]),
    transformed(cylinder, [0, -.086, -.002], [.026, .041, .027]),
  ]));
  const eyeGeometry = geometry(merge([
    transformed(tinySphere, [-.023, 0, .054], [.0031, .0041, .0025]),
    transformed(tinySphere, [.023, 0, .054], [.0031, .0041, .0025]),
  ]));
  const expressionGeometry = geometry(merge([
    tube([[-.029, .028, .051], [-.023, .029, .052], [-.017, .028, .053]], .0009, 5),
    tube([[.017, .028, .053], [.023, .029, .052], [.029, .028, .051]], .0009, 5),
    tube([[-.010, -.030, .052], [0, -.033, .055], [.010, -.030, .052]], .0010, 5),
  ]));
  const hairGeometry = geometry(merge([
    transformed(new THREE.SphereGeometry(1, 20, 12, 0, TAU, 0, 1.42), [0, .011, -.001], [.063, .070, .060]),
    transformed(tinySphere, [0, -.005, -.049], [.045, .040, .018]),
  ]));
  const handGeometry = geometry(merge([
    transformed(tinySphere, [.004, 0, 0], [.021, .029, .018]),
    transformed(tinySphere, [-.014, -.004, .006], [.012, .017, .011], [.28, 0, .30]),
  ]));
  const shoeUpper = geometry(merge([
    transformed(sphere, [0, .031, .014], [.034, .027, .070]),
    transformed(tinySphere, [-.014, .025, .063], [.018, .021, .026]),
    transformed(tinySphere, [.014, .025, .063], [.016, .020, .026]),
    transformed(cylinder, [0, .055, -.015], [.026, .039, .028]),
  ]));
  const shoeSole = geometry(transformed(sphere, [0, .010, .016], [.036, .009, .074]));
  const shoeSeam = geometry(merge([
    tube([[0, .019, .086], [0, .037, .068], [0, .047, .044]], .0014, 6),
    tube([[-.023, .047, .009], [0, .054, .017], [.023, .047, .009]], .0012, 7),
  ]));
  const armUpper = geometry(makeLimb(.053, .042));
  const armLower = geometry(makeLimb(.042, .029, .0011));
  const legUpper = geometry(makeLimb(.060, .050, .0019));
  const legLower = geometry(makeLimb(.048, .034, .0014));
  const pelvisGeometry = geometry(transformed(sphere, [0, -.020, -.005], [.116, .067, .080]));
  const clothJoint = geometry(new THREE.SphereGeometry(1, 12, 8));
  const toolShaft = geometry(transformed(cylinder, [0, .5, 0], [.010, 1, .010]));
  const toolCollar = geometry(merge(Array.from({ length: 7 }, (_, i) => transformed(new THREE.TorusGeometry(.0106, .0013, 4, 12), [0, .54 + i * .014, 0], [1, 1, 1], [Math.PI / 2, 0, 0]))));
  const toolBar = geometry(merge([
    transformed(cube, [0, 0, 0], [.190, .026, .043]),
    transformed(cube, [0, .023, -.005], [.190, .012, .052]),
    transformed(cube, [0, .007, -.018], [.028, .050, .030], [.36, 0, 0]),
  ]));
  const toolTeeth = geometry(merge(Array.from({ length: 5 }, (_, i) => transformed(cube, [(i - 2) * .040, -.037, .003], [.010, .050, .014], [-.08, 0, 0]))));
  const toolWing = geometry(transformed(cube, [.0475, 0, 0], [.095, .012, .052]));
  const strawMaterial = material({ color: 0xb69b68, roughness: .91, map: textures.straw, side: THREE.DoubleSide });
  const undersideMaterial = material({ color: 0x796848, roughness: 1, map: textures.straw, side: THREE.DoubleSide });
  const ropeMaterial = material({ color: 0x796340, roughness: 1 });
  const woodMaterial = material({ color: 0xa98856, roughness: .84, map: textures.wood });
  const wornWoodMaterial = material({ color: 0x786447, roughness: .91, map: textures.wood });
  const soleMaterial = material({ color: 0x363b34, roughness: 1 });
  const seamMaterial = material({ color: 0x696b59, roughness: 1 });
  const rigs = [];

  function hairFor(profile) {
    const parts = [hairGeometry.clone()];
    if (profile.style === 'grandmother') {
      parts.push(transformed(sphere, [0, .036, -.069], [.039, .036, .032]));
      parts.push(tube([[-.053, .026, .026], [-.040, .057, .036], [-.007, .069, .041]], .009, 12, 6));
    } else if (profile.style === 'mother') {
      parts.push(transformed(sphere, [0, .010, -.079], [.042, .033, .032]));
      parts.push(tube([[-.053, .021, .029], [-.037, .053, .040], [.006, .064, .039], [.040, .043, .029]], .010, 14, 6));
      for (const side of [-1, 1]) parts.push(transformed(sphere, [side * .055, -.009, -.005], [.016, .043, .026]));
    } else if (profile.style === 'sister') {
      parts.push(transformed(sphere, [0, .026, -.071], [.030, .030, .028]));
      parts.push(tube([[-.054, .011, .029], [-.048, .045, .042], [-.013, .063, .044]], .009, 12, 6));
      for (const side of [-1, 1]) parts.push(transformed(sphere, [side * .055, -.023, -.006], [.017, .055, .025]));
    } else if (profile.style === 'littleSister') {
      for (const side of [-1, 1]) {
        parts.push(transformed(sphere, [side * .067, -.008, -.011], [.029, .029, .027]));
        parts.push(transformed(sphere, [side * .080, -.044, -.008], [.027, .038, .025], [0, 0, side * .28]));
        parts.push(transformed(sphere, [side * .073, -.073, .001], [.020, .022, .021]));
      }
      for (let i = 0; i < 5; i++) parts.push(tube([[(i - 2) * .016, .055, .037], [(i - 2) * .018, .039, .051], [(i - 2) * .016 - .004, .031, .052]], .0068, 6, 5));
    } else if (profile.style === 'littleBrother') {
      for (let i = 0; i < 4; i++) parts.push(transformed(tinySphere, [(i - 1.5) * .022, .036, .045], [.016, .024, .013], [0, 0, -.25]));
    } else {
      for (const side of [-1, 1]) parts.push(transformed(tinySphere, [side * .050, -.011, .008], [.011, .024, .018]));
    }
    return geometry(merge(parts));
  }

  for (const profile of FAMILY_PROFILES) {
    const dimensions = proportions(profile), { scale } = dimensions;
    const group = new THREE.Group(); group.name = `家人-${profile.id}-${profile.name}`; group.visible = false; root.add(group);
    const body = new THREE.Group(); body.scale.setScalar(scale); group.add(body);
    const torsoPivot = new THREE.Group(); torsoPivot.position.y = dimensions.hipBase; body.add(torsoPivot);
    const clothing = new THREE.Group(); clothing.scale.y = dimensions.torsoScale; torsoPivot.add(clothing);
    const clothMaterial = material({ color: profile.cloth, map: textures.cloth, normalMap: textures.clothNormal, normalScale: new THREE.Vector2(.13, .13), roughness: .96 });
    const trousersMaterial = material({ color: profile.trousers, map: textures.cloth, normalMap: textures.clothNormal, normalScale: new THREE.Vector2(.11, .11), roughness: .99 });
    const trimMaterial = material({ color: profile.trim, roughness: .98, map: textures.cloth, side: THREE.DoubleSide });
    const skinMaterial = material({ color: profile.skin, roughness: .73 });
    const hairMaterial = material({ color: profile.hair, roughness: .95 });
    const faceMaterial = material({ color: 0x413b34, roughness: 1 });
    const shoeMaterial = material({ color: profile.id === 2 ? 0x9d9b7d : 0xb9b8a0, map: textures.cloth, roughness: .98 });
    const pelvis = mesh(pelvisGeometry, trousersMaterial, body, 'looseTrousers');
    mesh(geometry(makeTorso(profile.girth)), clothMaterial, clothing, profile.child ? 'childWorkShirt' : 'workJacket');
    const lapels = !profile.child ? geometry(merge([
      ribbon([[-.042, .366, .047], [-.046, .324, .072], [0, .257, .087], [.071, .161, .079]], [.019, .019, .021, .021]),
      ribbon([[.042, .366, .047], [.046, .324, .073], [0, .257, .088], [-.073, .172, .080]], [.019, .019, .021, .021]),
      ribbon([[-.112, .130, .017], [-.087, .127, .061], [0, .122, .082], [.087, .127, .061], [.112, .130, .017]], [.021, .021, .021, .021, .021]),
    ])) : null;
    if (lapels) mesh(drapeOverTorso(lapels, profile.girth), trimMaterial, clothing, 'lapelsAndWaistTie');
    else {
      const collar = geometry(merge([
        ribbon([[-.050, .357, .049], [-.045, .321, .079], [0, .311, .085]], [.017, .024, .017]),
        ribbon([[.050, .357, .049], [.045, .321, .079], [0, .311, .085]], [.017, .024, .017]),
        ribbon([[-.087, .102, .072], [-.085, .067, .083], [-.045, .065, .094]], [.005, .005, .005]),
        ribbon([[.045, .065, .094], [.085, .067, .083], [.087, .102, .072]], [.005, .005, .005]),
      ]));
      mesh(drapeOverTorso(collar, profile.girth), trimMaterial, clothing, 'roundedCollarAndPockets');
    }
    const seamGeometry = !profile.child ? geometry(merge([
      tube([[-.108, .018, .044], [-.063, .015, .074], [0, .014, .081], [.063, .015, .074], [.108, .018, .044]], .0016, 22),
      tube([[.070, .133, .079], [.086, .084, .084], [.077, .028, .087]], .0042, 8),
      tube([[.071, .133, .081], [.105, .086, .060], [.105, .037, .058]], .0040, 8),
    ])) : null;
    if (seamGeometry) mesh(drapeOverTorso(seamGeometry, profile.girth), trimMaterial, clothing, 'hemAndHangingTies');
    if (profile.style === 'mother' || profile.style === 'littleBrother') {
      const child = profile.child;
      const apron = geometry(merge([
        apronGeometry(child),
        ribbon([[-.042, .275, .099], [-.074, .322, .082], [-.098, .329, .007], [-.085, .268, -.080]], [.016, .021, .021, .016]),
        ribbon([[.042, .275, .099], [.074, .322, .082], [.098, .329, .007], [.085, .268, -.080]], [.016, .021, .021, .016]),
      ]));
      mesh(drapeOverTorso(apron, profile.girth), child ? trousersMaterial : trimMaterial, clothing, child ? 'childOverallsBib' : 'linenGardenApron');
      const pocket = geometry(merge([
        ribbon([[-.039, .080, .108], [-.037, .029, .109], [0, .025, .112], [.037, .029, .109], [.039, .080, .108]], [.003, .003, .003, .003, .003]),
        ribbon([[-.040, .080, .108], [0, .078, .113], [.040, .080, .108]], [.005, .005, .005]),
      ]));
      mesh(drapeOverTorso(pocket, profile.girth), child ? trimMaterial : clothMaterial, clothing, 'apronPocketStitching');
    }
    const head = new THREE.Group(); head.position.set(0, dimensions.headY, .008); head.scale.setScalar(dimensions.headScale); torsoPivot.add(head);
    mesh(faceGeometry, skinMaterial, head, 'faceAndNeck');
    const eyes = mesh(eyeGeometry, faceMaterial, head, 'softEyes'); eyes.position.y = .016;
    mesh(expressionGeometry, faceMaterial, head, 'quietSmile');
    mesh(hairFor(profile), hairMaterial, head, 'hair');
    let ponytail = null;
    if (profile.style === 'sister') {
      ponytail = new THREE.Group(); ponytail.position.set(0, .018, -.068); head.add(ponytail);
      mesh(geometry(merge([
        transformed(sphere, [0, -.028, -.012], [.026, .041, .025]),
        transformed(sphere, [.002, -.068, -.018], [.022, .031, .023]),
      ])), hairMaterial, ponytail, 'tiedPonytail');
      mesh(geometry(transformed(new THREE.TorusGeometry(.024, .004, 5, 16), [0, -.006, -.006], [1, 1, 1], [Math.PI / 2, 0, 0])), trimMaterial, ponytail, 'ponytailRibbon');
    }
    if (profile.style === 'grandmother') {
      const glasses = geometry(merge([
        transformed(new THREE.TorusGeometry(.016, .0012, 5, 20), [-.025, .013, .055]),
        transformed(new THREE.TorusGeometry(.016, .0012, 5, 20), [.025, .013, .055]),
        tube([[-.009, .016, .056], [0, .019, .060], [.009, .016, .056]], .0012, 6),
        tube([[-.041, .016, .052], [-.054, .017, .015]], .0012, 4),
        tube([[.041, .016, .052], [.054, .017, .015]], .0012, 4),
      ]));
      mesh(glasses, wornWoodMaterial, head, 'roundSpectacles');
    }
    if (profile.style === 'littleSister') {
      const bows = geometry(merge([-1, 1].flatMap(side => [
        transformed(tinySphere, [side * .070, -.019, .015], [.012, .018, .008], [0, 0, .60]),
        transformed(tinySphere, [side * .078, -.018, .016], [.012, .018, .008], [0, 0, -.60]),
      ])));
      mesh(bows, trimMaterial, head, 'twoHairRibbons');
    }
    const hat = new THREE.Group(); hat.position.y = .075; head.add(hat);
    if (profile.headwear === 'straw') {
      mesh(hatTop, strawMaterial, hat, 'wovenHat');
      mesh(hatBottom, undersideMaterial, hat, 'hatInside');
      mesh(hatRim, ropeMaterial, hat, 'boundHatRim');
      mesh(hatTie, ropeMaterial, hat, 'hatChinCord');
    } else if (profile.headwear === 'scarf') {
      const scarf = geometry(merge([
        tube([[-.062, -.050, .009], [-.060, -.012, -.017], [0, .001, -.054], [.060, -.012, -.017], [.062, -.050, .009]], .010, 26, 6),
        ribbon([[-.009, -.018, -.077], [-.027, -.071, -.081], [-.013, -.124, -.067]], [.025, .030, .015]),
        ribbon([[.009, -.018, -.079], [.029, -.055, -.085], [.022, -.096, -.076]], [.025, .030, .012]),
      ]));
      mesh(scarf, trimMaterial, hat, 'linenHairScarf');
    } else if (profile.headwear === 'cap') {
      const cap = geometry(merge([
        transformed(new THREE.SphereGeometry(1, 24, 12, 0, TAU, 0, 1.60), [0, -.027, -.004], [.069, .054, .067]),
        transformed(sphere, [0, -.029, .055], [.073, .007, .056]),
      ]));
      mesh(cap, trousersMaterial, hat, 'softCottonCap');
      mesh(geometry(tube([[0, .027, -.006], [0, .012, .042], [0, -.023, .063]], .0018, 12)), trimMaterial, hat, 'capSeam');
    }
    const arms = [-1, 1].map(side => {
      const upper = mesh(armUpper, clothMaterial, body, side < 0 ? 'leftUpperArm' : 'rightUpperArm');
      const lower = mesh(armLower, profile.style === 'sister' ? skinMaterial : clothMaterial, body, side < 0 ? 'leftForearm' : 'rightForearm');
      const hand = mesh(handGeometry, skinMaterial, body, side < 0 ? 'leftHand' : 'rightHand');
      return { side, upper, lower, hand, start: new THREE.Vector3(), end: new THREE.Vector3(), elbow: new THREE.Vector3() };
    });
    const elbowCovers = new THREE.InstancedMesh(clothJoint, clothMaterial, 2);
    elbowCovers.name = 'clothElbowCovers'; elbowCovers.castShadow = true; elbowCovers.receiveShadow = true; elbowCovers.frustumCulled = false; body.add(elbowCovers);
    elbowCovers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const feet = [-1, 1].map(side => {
      const upper = mesh(legUpper, trousersMaterial, body, side < 0 ? 'leftThigh' : 'rightThigh');
      const lower = mesh(legLower, trousersMaterial, body, side < 0 ? 'leftShin' : 'rightShin');
      const foot = new THREE.Group(); foot.name = side < 0 ? 'leftFoot' : 'rightFoot'; body.add(foot);
      foot.scale.z = .84;
      mesh(shoeUpper, shoeMaterial, foot, 'splitToeShoe');
      mesh(shoeSole, soleMaterial, foot, 'flatSole');
      mesh(shoeSeam, seamMaterial, foot, 'shoeSeams');
      return { side, upper, lower, foot, planted: new THREE.Vector3(), from: new THREE.Vector3(), target: new THREE.Vector3(), current: new THREE.Vector3(), yaw: 0, targetYaw: 0, step: 1, duration: .38, swinging: false, knee: new THREE.Vector3() };
    });
    const kneeCovers = new THREE.InstancedMesh(clothJoint, trousersMaterial, 2);
    kneeCovers.name = 'clothKneeCovers'; kneeCovers.castShadow = true; kneeCovers.receiveShadow = true; kneeCovers.frustumCulled = false; body.add(kneeCovers);
    kneeCovers.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const tool = new THREE.Group(); tool.name = 'gardenRake'; group.add(tool);
    const shaft = mesh(toolShaft, woodMaterial, tool, 'woodenShaft');
    shaft.scale.x = shaft.scale.z = dimensions.toolScale;
    const shaftWrap = mesh(toolCollar, ropeMaterial, shaft, 'handleWrapping');
    // The wrap is in normalized shaft coordinates; compensate its radial scale.
    shaftWrap.scale.y = 1;
    const toolHead = new THREE.Group(); toolHead.name = 'reversibleRakeHead'; tool.add(toolHead);
    toolHead.scale.setScalar(dimensions.toolScale);
    mesh(toolBar, woodMaterial, toolHead, 'rakeCrossbar');
    mesh(toolTeeth, wornWoodMaterial, toolHead, 'fiveRakeTeeth');
    const wings = [-1, 1].map(side => {
      const hinge = new THREE.Group(); hinge.position.set(side * .095, .023, -.005); toolHead.add(hinge);
      const wing = mesh(toolWing, woodMaterial, hinge, 'foldingSmoothingBlade');
      wing.scale.x = side;
      return { side, hinge };
    });
    const rig = {
      ...profile, ...dimensions, group, body, torsoPivot, pelvis, head, eyes, hat, ponytail, arms, feet, elbowCovers, kneeCovers, tool, shaft, toolHead, wings,
      clothMaterial, trousersMaterial, trimMaterial, skinMaterial, shoeMaterial,
      clothColor: new THREE.Color(profile.cloth), trousersColor: new THREE.Color(profile.trousers),
      x: 0, z: 0, groundY: 0, heading: 0, initialized: false, wasVisible: false, activeFoot: null, nextFoot: 0,
      leanAmount: 0, moving: 0, moveSign: 1, flip: 0, workBlend: 0, stoneBend: 0, stoneReachAdjust: 0, carryWeight: 0, gaitPhase: 0, stepsTaken: 0, speed: 0, sway: 0, turn: 0, toolWidth: profile.rakeWidth, stance: dimensions.footSpread, toeOut: .045, headScaleX: dimensions.toolScale,
      diagnostics: { gripError: 0, legStretch: 1, armStretch: 1, contactY: 0, feet: [] },
    };
    group.userData.live = rig.diagnostics;
    rigs.push(rig);
  }
  sphere.dispose(); tinySphere.dispose(); cube.dispose(); cylinder.dispose();
  let wetness = -1;

  function initializeFeet(rig) {
    for (const f of rig.feet) {
      localToWorldPoint(new THREE.Vector3(f.side * rig.stance, 0, 0), rig, f.planted);
      f.current.copy(f.planted); f.target.copy(f.planted); f.from.copy(f.planted);
      f.yaw = rig.heading + f.side * rig.toeOut; f.targetYaw = f.yaw; f.swinging = false; f.step = 1;
    }
    rig.activeFoot = null; rig.nextFoot = 0; rig.stepsTaken = 0;
  }

  function soleCorners(rig, foot, at = foot.planted, yaw = foot.yaw) {
    const c = Math.cos(yaw), s = Math.sin(yaw), corners = [];
    // A conservative rectangle around the actual sole, including its toe.
    for (const x of [-.037, .037]) for (const z of [-.059 * .84, .091 * .84]) {
      corners.push({ x: at.x + (c * x + s * z) * rig.scale, z: at.z + (-s * x + c * z) * rig.scale });
    }
    return corners;
  }

  function supportTime(rig, foot, vx, vz, desired) {
    const speedSq = vx * vx + vz * vz;
    if (speedSq < .000001) return desired;
    let duration = desired;
    // Keep the planted sole inside the planner's 18 cm foot disc. Reserve
    // one frame of travel; a supporting foot never slides to chase the body.
    for (const corner of soleCorners(rig, foot)) {
      const dx = corner.x - rig.x, dz = corner.z - rig.z;
      const b = -2 * (dx * vx + dz * vz), c = dx * dx + dz * dz - .169 ** 2;
      const root = (-b + Math.sqrt(Math.max(0, b * b - 4 * speedSq * c))) / (2 * speedSq);
      duration = Math.min(duration, Math.max(.085, root));
    }
    return duration;
  }

  function fitLanding(rig, foot, expectedX, expectedZ) {
    // Only an airborne foot can make this small balance adjustment. It matters
    // when the walking path bends while a step is already in progress.
    for (let i = 0; i < 3; i++) {
      let radius = 0;
      for (const corner of soleCorners(rig, foot, foot.target, foot.targetYaw)) radius = Math.max(radius, Math.hypot(corner.x - expectedX, corner.z - expectedZ));
      if (radius < .168) break;
      const f = clamp(.07 + (radius - .168) / Math.max(.001, radius), .07, .65);
      foot.target.x = lerp(foot.target.x, expectedX, f);
      foot.target.z = lerp(foot.target.z, expectedZ, f);
    }
  }

  function updateFeet(rig, pose, dt, time, measuredSpeed) {
    const requestedSpeed = Math.abs(Number(pose.walkSpeed) || 0);
    const speed = Math.max(requestedSpeed, measuredSpeed);
    rig.speed = lerp(rig.speed, speed, 1 - Math.exp(-dt * 9));
    rig.moving = lerp(rig.moving, speed > .012 ? 1 : 0, 1 - Math.exp(-dt * 7));
    rig.gaitPhase += measuredSpeed * dt / (rig.child ? .34 : lerp(.46,.40,rig.carryWeight)) * TAU;
    const hasVelocity = Math.hypot(rig.velocityX, rig.velocityZ) > .00001;
    const walking = speed > .012, directionX = hasVelocity ? rig.velocityX : Math.sin(rig.heading) * rig.moveSign, directionZ = hasVelocity ? rig.velocityZ : Math.cos(rig.heading) * rig.moveSign;
    const directionLength = Math.max(.000001, Math.hypot(directionX, directionZ));
    const vx = directionX / directionLength * speed, vz = directionZ / directionLength * speed;
    if (rig.activeFoot === null) {
      const candidates = rig.feet.map((f, index) => {
        const local = worldToLocalPoint(f.planted, rig);
        const lateral = local.x - f.side * rig.stance;
        return { f, index, error: Math.hypot(lateral, local.z), local };
      });
      const preferred = candidates[rig.nextFoot];
      const other = candidates[1 - rig.nextFoot];
      const choice = preferred.error > .036 || other.error <= .065 ? preferred : other;
      const turn = Math.abs(wrapAngle(rig.heading + choice.f.side * rig.toeOut - choice.f.yaw));
      if ((walking && rig.stepsTaken === 0) || choice.error > (walking ? .039 : .066) || turn > .31) {
        const f = choice.f;
        rig.activeFoot = choice.index; f.swinging = true; f.step = 0; f.from.copy(f.planted);
        const naturalDuration = walking ? clamp(.26 + .025 / Math.max(.13, speed), .27, .41) * (rig.child ? .87 : 1 + .08 * rig.carryWeight) : .29;
        f.duration = supportTime(rig, rig.feet[1 - choice.index], vx, vz, rig.stepsTaken === 0 && walking ? .16 : naturalDuration);
        localToWorldPoint(new THREE.Vector3(f.side * rig.stance, 0, .004), rig, f.target);
        if (walking) {
          const anticipation = f.duration + (rig.child ? .022 : .034) / Math.max(.10, speed);
          f.target.x += vx * anticipation;
          f.target.z += vz * anticipation;
        }
        f.target.y = rig.groundY;
        f.targetYaw = rig.heading + f.side * rig.toeOut;
        fitLanding(rig, f, rig.x + vx * f.duration, rig.z + vz * f.duration);
      }
    }
    const bob = Math.sin(time * 2.0 + rig.id * 1.7) * .0017 * (1 - rig.moving);
    rig.diagnostics.legStretch = 1;
    rig.diagnostics.feet.length = 0;
    rig.diagnostics.maxPlantedSoleRadius = 0;
    let stepLift = 0;
    for (const f of rig.feet) {
      if (f.swinging) {
        f.step = Math.min(1, f.step + dt / f.duration);
        fitLanding(rig, f, rig.x + vx * f.duration * (1 - f.step), rig.z + vz * f.duration * (1 - f.step));
        const eased = f.step * f.step * (3 - 2 * f.step);
        f.current.lerpVectors(f.from, f.target, eased);
        const lift = Math.sin(f.step * Math.PI);
        f.current.y += lift * (walking ? lerp(.035,.027,rig.carryWeight) : .023) * rig.scale;
        stepLift = Math.max(stepLift, lift);
        if (f.step >= 1) {
          f.planted.copy(f.target); f.current.copy(f.target); f.yaw = f.targetYaw;
          f.swinging = false; rig.activeFoot = null; rig.nextFoot = f.side < 0 ? 1 : 0; rig.stepsTaken++;
        }
      } else f.current.copy(f.planted);
      f.foot.position.copy(worldToLocalPoint(f.current, rig));
      const turnMix = f.swinging ? f.step * f.step * (3 - 2 * f.step) : 0;
      const worldYaw = f.yaw + wrapAngle(f.targetYaw - f.yaw) * turnMix;
      f.foot.rotation.set(f.swinging ? Math.sin(f.step * Math.PI * 2) * .115 : 0, wrapAngle(worldYaw - rig.heading), 0);
      rig.diagnostics.feet.push({ side: f.side, planted: !f.swinging, x: f.current.x, y: f.current.y, z: f.current.z });
      if (!f.swinging) for (const corner of soleCorners(rig, f)) rig.diagnostics.maxPlantedSoleRadius = Math.max(rig.diagnostics.maxPlantedSoleRadius, Math.hypot(corner.x - rig.x, corner.z - rig.z));
    }
    // A person lowers their pelvis slightly when their feet are spread. Keeping
    // the feet fixed and adapting the hips avoids telescoping knees at turns.
    let hipHeight = rig.hipBase + bob - stepLift * .008 * rig.moving - rig.workBlend * .012 - rig.hipBase * (.56 * rig.stoneBend + .03 * rig.stoneReachAdjust);
    for (const f of rig.feet) {
      const ankle = f.foot.position.clone().add(new THREE.Vector3(0, .057, -.014));
      const horizontalSq = (ankle.x - f.side * .064 * rig.girth) ** 2 + (ankle.z + .007) ** 2;
      const reachableHeight = ankle.y + Math.sqrt(Math.max(.01, ((rig.legA + rig.legB) * .99) ** 2 - horizontalSq));
      hipHeight = Math.min(hipHeight, reachableHeight);
    }
    const supportSide = rig.activeFoot === null ? 0 : -rig.feet[rig.activeFoot].side;
    rig.sway = lerp(rig.sway, supportSide * .007 * stepLift, 1 - Math.exp(-dt * 8));
    const hipZ = -.004 * rig.workBlend - .015 * rig.stoneBend;
    rig.torsoPivot.position.set(rig.sway, hipHeight, hipZ);
    rig.pelvis.position.copy(rig.torsoPivot.position);
    for (const [index, f] of rig.feet.entries()) {
      const hip = new THREE.Vector3(f.side * .064 * rig.girth + rig.sway, hipHeight, -.007 + hipZ);
      const ankle = f.foot.position.clone().add(new THREE.Vector3(0, .057, -.014));
      const stretch = solveJoint(hip, ankle, rig.legA, rig.legB, new THREE.Vector3(f.side * .072, hipHeight * .55, .18), f.knee);
      rig.diagnostics.legStretch = Math.max(rig.diagnostics.legStretch, stretch);
      connect(f.upper, hip, f.knee); connect(f.lower, f.knee, ankle);
      rig.kneeCovers.setMatrixAt(index, jointMatrix.compose(f.knee, jointRotation, kneeCoverScale));
    }
    rig.kneeCovers.instanceMatrix.needsUpdate = true;
    rig.diagnostics.steps = rig.stepsTaken;
    rig.diagnostics.stepDuration = rig.activeFoot === null ? null : rig.feet[rig.activeFoot].duration;
  }

  function headLowestY(rig) {
    const a = -Math.PI * rig.flip, ca = Math.cos(a), sa = Math.sin(a);
    let minimum = Infinity;
    const sample = (x, y) => { minimum = Math.min(minimum, sa * x * rig.headScaleX + ca * y * rig.toolScale); };
    for (const x of [-.095, .095]) for (const y of [-.013, .029]) sample(x, y);
    for (const x of [-.014, .014]) for (const y of [-.022, .036]) sample(x, y);
    for (const x of [-.085, .085]) sample(x, -.0625);
    for (const { side } of rig.wings) {
      const wa = side * Math.PI / 2 * (1 - rig.flip), cw = Math.cos(wa), sw = Math.sin(wa);
      for (const x of [0, side * .095]) for (const y of [-.006, .006]) sample(side * .095 + cw * x - sw * y, .023 + sw * x + cw * y);
    }
    return minimum;
  }

  function updateToolAndArms(rig, pose, dt, time) {
    const working = !!pose.working, contact = clamp(Number(pose.contact) || 0);
    const smooth = pose.activity === 'smooth' || pose.activity === 'sweep';
    const preparing = /lowering|Lowering|working|accessOpening/.test(pose.motion || '');
    rig.workBlend = lerp(rig.workBlend, working || preparing || contact > .018 ? 1 : 0, 1 - Math.exp(-dt * 7));
    rig.flip = lerp(rig.flip, smooth ? 1 : 0, 1 - Math.exp(-dt * 11));
    if (Math.abs(rig.flip - (smooth ? 1 : 0)) < .0001) rig.flip = smooth ? 1 : 0;
    const baseWidth = smooth ? rig.smoothWidth : rig.rakeWidth;
    const requestedWidth = Number.isFinite(pose.toolWidth) ? clamp(pose.toolWidth, .06, .78) : baseWidth;
    rig.headScaleX = lerp(rig.headScaleX, rig.toolScale * requestedWidth / baseWidth, 1 - Math.exp(-dt * 11));
    rig.toolHead.scale.set(rig.headScaleX, rig.toolScale, rig.toolScale);
    const placement = contact > .975 ? 1 : Math.max(rig.workBlend, 1 - (1 - contact) ** 4);
    const sourceHeading = Number.isFinite(pose.toolHeading) ? pose.toolHeading : rig.heading;
    const toolHeading = rig.heading + wrapAngle(sourceHeading - rig.heading) * placement;
    const sx = Math.sin(toolHeading), cz = Math.cos(toolHeading);
    const c = Math.cos(rig.heading), s = Math.sin(rig.heading);
    const carriedForward = .115 * rig.scale + Math.sin(rig.gaitPhase) * .014 * rig.moving;
    const carriedSide = (rig.headwear === 'straw' ? .30 : .181) * rig.scale;
    const carriedX = rig.x + c * carriedSide + s * carriedForward;
    const carriedZ = rig.z - s * carriedSide + c * carriedForward;
    const tx = lerp(carriedX, Number.isFinite(pose.toolX) ? pose.toolX : rig.x + sx * rig.reach, placement);
    const tz = lerp(carriedZ, Number.isFinite(pose.toolZ) ? pose.toolZ : rig.z + cz * rig.reach, placement);
    const ty = lerp(rig.groundY, Number.isFinite(pose.toolY) ? pose.toolY : 0, placement);
    const relativeHeading = wrapAngle(toolHeading - rig.heading);
    const liftedHeight = Number.isFinite(pose.toolLift) ? clamp(pose.toolLift, .08, .36) : .17;
    const carryLift = .125 + .004 * Math.sin(rig.gaitPhase * 2) * rig.moving;
    const headElevation = -headLowestY(rig) + .001 + (1 - contact) * lerp(carryLift, liftedHeight, placement);
    rig.tool.position.set(c * (tx - rig.x) - s * (tz - rig.z), ty + headElevation - rig.groundY, s * (tx - rig.x) + c * (tz - rig.z));
    rig.tool.rotation.y = relativeHeading;
    rig.toolHead.rotation.z = -Math.PI * rig.flip;
    for (const { side, hinge } of rig.wings) hinge.rotation.z = side * Math.PI / 2 * (1 - rig.flip);
    // A relaxed one-handed carry becomes a two-handed working grip. The blade
    // is driven by the simulation as it touches sand; breathing and arm swing
    // can never move a visible groove away from its actual working edge.
    const toBodyX = rig.x - tx, toBodyZ = rig.z - tz;
    const bodyDistance = Math.max(.001, Math.hypot(toBodyX, toBodyZ));
    const shaftX = (Math.cos(toolHeading) * toBodyX - Math.sin(toolHeading) * toBodyZ) / bodyDistance;
    const shaftZ = (Math.sin(toolHeading) * toBodyX + Math.cos(toolHeading) * toBodyZ) / bodyDistance;
    const shaftReach = clamp(bodyDistance - .185 * rig.scale, .025, .98);
    const shaftA = new THREE.Vector3(0, .011 * rig.toolScale, 0);
    const shaftHeight = (rig.torsoPivot.position.y + rig.shoulderY + .080) * rig.scale - (ty + headElevation - rig.groundY);
    // Aim the upper handle beside the right shoulder, so a long rigid shaft
    // cannot continue through the face when the blade is extended forward.
    const sideReach = (rig.child ? .085 : .13) * rig.scale;
    const shaftB = new THREE.Vector3(lerp(.012 * rig.scale, shaftX * shaftReach + Math.cos(relativeHeading) * sideReach, placement), shaftHeight, lerp(-.128 * rig.scale, shaftZ * shaftReach + Math.sin(relativeHeading) * sideReach, placement));
    // One solid handle, at every reach. Hands slide along it and its angle
    // changes; the wood itself never telescopes during a stroke or a walk.
    const handleLength = rig.headwear === 'straw' ? 1.0 : rig.id === 4 ? .67 : rig.child ? .90 : 1.28;
    shaftB.sub(shaftA).normalize().multiplyScalar(handleLength).add(shaftA);
    connect(rig.shaft, shaftA, shaftB);
    rig.tool.updateMatrix();
    const shaftDirection = shaftB.clone().sub(shaftA).normalize();
    const localShaftDirection = shaftDirection.clone().applyAxisAngle(UP, relativeHeading);
    rig.diagnostics.armStretch = 1;
    rig.diagnostics.gripError = 0;
    rig.diagnostics.contactY = ty + headElevation + headLowestY(rig);
    rig.diagnostics.activity = pose.activity || (working ? 'rake' : 'walk');
    rig.diagnostics.twoHanded = placement > .95;
    rig.diagnostics.handleLength = handleLength;
    const workingLean = .105 + clamp((bodyDistance - .30) / .35) * .145;
    const breathing = Math.sin(time * 1.9 + rig.id) * .0035;
    const gaitSwing = Math.sin(rig.gaitPhase);
    const leanTarget = lerp(.025 + .020 * rig.moving, workingLean, placement) + rig.lean + breathing;
    rig.leanAmount = lerp(rig.leanAmount, leanTarget, 1 - Math.exp(-dt * 6));
    rig.torsoPivot.rotation.set(rig.leanAmount, -gaitSwing * .028 * rig.moving * (1 - placement) + rig.turn * .09, -rig.sway * 1.9);
    rig.pelvis.rotation.z = -rig.sway;
    for (const [index, arm] of rig.arms.entries()) {
      const extension = clamp((bodyDistance - (rig.child ? .35 : .55)) / (rig.child ? .25 : .31));
      const shoulderHeight = rig.torsoPivot.position.y + rig.shoulderY;
      const gripHeight = arm.side < 0
        ? (shoulderHeight - .11) * rig.scale
        : lerp((rig.torsoPivot.position.y - .015) * rig.scale, (shoulderHeight - lerp(.265, .180, extension)) * rig.scale, placement);
      const shoulder = new THREE.Vector3(arm.side * .136 * rig.girth, rig.shoulderY, .006).applyEuler(rig.torsoPivot.rotation).add(rig.torsoPivot.position);
      let t = clamp((gripHeight - (ty + headElevation - rig.groundY) - shaftA.y) / Math.max(.1, shaftB.y - shaftA.y), .13, .95);
      // A hand can slide along the fixed wooden handle while it is lowered.
      // Clamp its preferred height to the arm's real reach rather than
      // lengthening a child's forearm during the carry-to-work transition.
      const localA = shaftA.clone().applyMatrix4(rig.tool.matrix).divideScalar(rig.scale);
      const localB = shaftB.clone().applyMatrix4(rig.tool.matrix).divideScalar(rig.scale);
      const along = localB.clone().sub(localA), spanSquared = along.lengthSq();
      const fromA = shoulder.clone().sub(localA);
      const projected = fromA.dot(along) / spanSquared;
      const perpendicularSquared = Math.max(0, fromA.lengthSq() - projected * projected * spanSquared);
      const reachSquared = ((rig.armA + rig.armB) * .985) ** 2;
      if (perpendicularSquared < reachSquared) {
        const radius = Math.sqrt((reachSquared - perpendicularSquared) / spanSquared);
        const low = Math.max(.13, projected - radius), high = Math.min(.95, projected + radius);
        if (low <= high) t = clamp(t, low, high);
      }
      const grip = localA.lerp(localB, t);
      arm.end.copy(grip);
      if (arm.side < 0) {
        const freeHand = new THREE.Vector3(-.161 * rig.girth, rig.torsoPivot.position.y - (rig.child ? .044 : .072), .018 - gaitSwing * .067 * rig.moving);
        arm.end.lerpVectors(freeHand, grip, placement);
      }
      arm.start.copy(shoulder);
      const pole = new THREE.Vector3(arm.side * lerp(.158, .190, placement) * rig.girth, shoulder.y - .23, lerp(.029, -.012, placement) - gaitSwing * .035 * (1 - placement) * rig.moving);
      const stretch = solveJoint(arm.start, arm.end, rig.armA, rig.armB, pole, arm.elbow);
      rig.diagnostics.armStretch = Math.max(rig.diagnostics.armStretch, stretch);
      connect(arm.upper, arm.start, arm.elbow); connect(arm.lower, arm.elbow, arm.end);
      rig.elbowCovers.setMatrixAt(index, jointMatrix.compose(arm.elbow, jointRotation, elbowCoverScale));
      arm.hand.position.copy(arm.end);
      const carryQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(.06 - gaitSwing * .16 * rig.moving, 0, -.10));
      const gripQuaternion = new THREE.Quaternion().setFromUnitVectors(UP, localShaftDirection);
      arm.hand.quaternion.copy(arm.side < 0 ? carryQuaternion.slerp(gripQuaternion, placement) : gripQuaternion);
      if (arm.side > 0) arm.hand.rotateY(Math.PI);
    }
    rig.elbowCovers.instanceMatrix.needsUpdate = true;
    const headFollow = 1 - Math.exp(-dt * 5);
    const headYaw = lerp(Math.sin(time * .41 + rig.id * .7) * .105 + rig.turn * .35, clamp(relativeHeading, -.24, .24), placement);
    rig.head.rotation.x = lerp(rig.head.rotation.x, -.45 * rig.leanAmount + .015 + placement * .19, headFollow);
    rig.head.rotation.y = lerp(rig.head.rotation.y, headYaw, headFollow);
    rig.head.rotation.z = lerp(rig.head.rotation.z, -rig.torsoPivot.rotation.z * .65, headFollow);
    const blinkPhase = (time + rig.id * 1.13) % (4.2 + rig.id * .31);
    rig.eyes.scale.y = blinkPhase < .18 ? 1 - Math.sin(blinkPhase / .18 * Math.PI) ** 2 * .95 : 1;
    rig.hat.rotation.z = Math.sin(time * .7 + rig.id) * .004;
    if (rig.ponytail) {
      rig.ponytail.rotation.x = Math.sin(time * 4.7 + rig.id) * .11 * rig.moving + Math.sin(time * .8) * .018;
      rig.ponytail.rotation.z = Math.sin(time * 3.8 + rig.id) * .035 * rig.moving;
    }
  }

  // The world owns the stone's one physical mesh and its whole trajectory. Both
  // palms solve to that exact world-space load, so gait, breathing and turns
  // never introduce a second stone or make a carried stone drift off the hands.
  function updateStoneArms(rig, pose, dt, time) {
    const stone = pose.stone;
    const valid = stone && [stone.x,stone.y,stone.z].every(Number.isFinite);
    const grip = valid ? clamp(Number(stone.grip) || 0) : 0;
    const bend = rig.stoneBend;
    const reach = valid ? Math.max(grip,clamp(Number(stone.bend) || 0)) : 0;
    const gaitSwing = Math.sin(rig.gaitPhase);
    const radius = valid && Number.isFinite(stone.radius) ? clamp(stone.radius,.075,.16) : .118;
    const target = valid ? worldToLocalPoint(stone,rig) : new THREE.Vector3(0,.5,.23);
    // A deep knee bend reaches the low garden stones without lengthening an
    // arm. The planner supplies a continuous bend value with the lift/lower.
    rig.leanAmount = .065 + rig.lean + .965 * bend + .090 * rig.stoneReachAdjust + .014 * rig.moving * (1-bend) + Math.sin(time*1.9+rig.id)*.0018;
    rig.torsoPivot.rotation.set(rig.leanAmount,-gaitSwing*.016*rig.moving*(1-reach)+rig.turn*.055,-rig.sway*1.5);
    rig.pelvis.rotation.z = -rig.sway;
    rig.workBlend = 0;
    rig.diagnostics.armStretch = 1;
    rig.diagnostics.gripError = 0;
    rig.diagnostics.contactY = valid ? stone.y : 0;
    rig.diagnostics.activity = valid ? (stone.carrying ? 'carry-stone' : 'handle-stone') : 'walk';
    rig.diagnostics.twoHanded = grip > .95;
    rig.diagnostics.equipment = 'hands';
    rig.diagnostics.stoneId = valid ? stone.id : null;
    rig.diagnostics.stoneGrip = grip;
    rig.diagnostics.stoneBend = bend;
    rig.diagnostics.stoneHands = [];
    for (const [index,arm] of rig.arms.entries()) {
      arm.start.set(arm.side*.136*rig.girth,rig.shoulderY,.006).applyEuler(rig.torsoPivot.rotation).add(rig.torsoPivot.position);
      const freeHand = new THREE.Vector3(arm.side*.161*rig.girth,rig.torsoPivot.position.y-.068,.017+arm.side*gaitSwing*.063*rig.moving);
      const desired = target.clone().add(new THREE.Vector3(arm.side*radius/rig.scale,.008/rig.scale,-.010/rig.scale));
      arm.end.lerpVectors(freeHand,desired,reach);
      // Empty palms approach and withdraw along the real arm's reachable arc.
      // The exact locked grip is only reached after the gardener has crouched.
      if (grip < .999) {
        const offset = arm.end.clone().sub(arm.start), limit = (rig.armA+rig.armB)*.985;
        if (offset.length() > limit) arm.end.copy(arm.start).add(offset.setLength(limit));
      }
      const pole = new THREE.Vector3(arm.side*(.205+.040*grip)*rig.girth,arm.start.y-.18,target.z*.28-.035);
      const stretch = solveJoint(arm.start,arm.end,rig.armA,rig.armB,pole,arm.elbow);
      rig.diagnostics.armStretch = Math.max(rig.diagnostics.armStretch,stretch);
      connect(arm.upper,arm.start,arm.elbow); connect(arm.lower,arm.elbow,arm.end);
      rig.elbowCovers.setMatrixAt(index,jointMatrix.compose(arm.elbow,jointRotation,elbowCoverScale));
      arm.hand.position.copy(arm.end);
      const relaxed = new THREE.Quaternion().setFromEuler(new THREE.Euler(-arm.side*gaitSwing*.14*rig.moving,0,-arm.side*.08));
      // Slightly inward palms and rounded wrists cup either side of the rock.
      const cupped = new THREE.Quaternion().setFromEuler(new THREE.Euler(.12+grip*.11,-arm.side*.34,-arm.side*.72));
      arm.hand.quaternion.copy(relaxed.slerp(cupped,reach));
      const actualWorld = localToWorldPoint(arm.end,rig);
      const targetWorld = localToWorldPoint(desired,rig);
      if (grip>.95) rig.diagnostics.gripError = Math.max(rig.diagnostics.gripError,actualWorld.distanceTo(targetWorld));
      rig.diagnostics.stoneHands.push({side:arm.side,x:actualWorld.x,y:actualWorld.y,z:actualWorld.z,targetX:targetWorld.x,targetY:targetWorld.y,targetZ:targetWorld.z});
    }
    rig.elbowCovers.instanceMatrix.needsUpdate = true;
    const headLocal = rig.head.position.clone().applyEuler(rig.torsoPivot.rotation).add(rig.torsoPivot.position);
    let pitch = valid ? clamp(Math.atan2(headLocal.y-target.y,target.z-headLocal.z)-rig.leanAmount,-.25,.58) : .025;
    // The broad straw brim stays above a raised rock; a small neck correction
    // lets the father watch his hands without tipping the brim into the load.
    if(rig.headwear==='straw'&&rig.raisedStoneReach>0) pitch=Math.min(pitch,lerp(.58,.72-rig.leanAmount,rig.raisedStoneReach));
    const yaw = valid ? clamp(Math.atan2(target.x-headLocal.x,Math.max(.08,target.z-headLocal.z)),-.28,.28) : Math.sin(time*.41+rig.id*.7)*.10;
    const headFollow = dt > 0 ? 1-Math.exp(-dt*8) : 1;
    rig.head.rotation.x = lerp(rig.head.rotation.x,lerp(.025,pitch,reach),headFollow);
    rig.head.rotation.y = lerp(rig.head.rotation.y,yaw,headFollow);
    rig.head.rotation.z = lerp(rig.head.rotation.z,-rig.torsoPivot.rotation.z*.65,headFollow);
    const blinkPhase = (time+rig.id*1.13)%(4.2+rig.id*.31);
    rig.eyes.scale.y = blinkPhase<.18 ? 1-Math.sin(blinkPhase/.18*Math.PI)**2*.95 : 1;
    rig.hat.rotation.z = Math.sin(time*.7+rig.id)*.003;
    if (rig.ponytail) {
      rig.ponytail.rotation.x = -.15*bend+Math.sin(time*4.7+rig.id)*.070*rig.moving;
      rig.ponytail.rotation.z = Math.sin(time*3.8+rig.id)*.025*rig.moving;
    }
  }

  function update(poses = [], dt = 1 / 60, time = 0, env = {}) {
    dt = clamp(Number(dt) || 0, 0, .10);
    time = Number(time) || 0;
    const byId = new Map(poses.filter(Boolean).map(p => [p.id, p]));
    const nextWet = clamp(Number(env.wetness) || 0);
    if (Math.abs(nextWet - wetness) > .005) {
      wetness = nextWet;
      for (const rig of rigs) {
        rig.clothMaterial.color.copy(rig.clothColor).multiplyScalar(1 - wetness * .22);
        rig.trousersMaterial.color.copy(rig.trousersColor).multiplyScalar(1 - wetness * .18);
        rig.clothMaterial.roughness = .96 - wetness * .16;
        rig.trousersMaterial.roughness = .99 - wetness * .12;
      }
      strawMaterial.color.setHex(0xb69b68).multiplyScalar(1 - wetness * .17);
      strawMaterial.roughness = .91 - wetness * .14;
      woodMaterial.roughness = .84 - wetness * .18;
    }
    for (const rig of rigs) {
      const pose = byId.get(rig.id);
      const visible = !!pose && pose.visible !== false;
      rig.group.visible = visible;
      if (!pose || !visible) { rig.wasVisible = false; continue; }
      const x = Number(pose.x) || 0, z = Number(pose.z) || 0;
      const heading = Number(pose.heading) || 0;
      const groundY = Number(pose.groundY) || 0;
      const hands = pose.equipment === 'hands';
      rig.tool.visible = !hands;
      const stoneDistance = hands && pose.stone ? Math.hypot(pose.stone.x-x,pose.stone.z-z) : 0;
      // A shelf already raises the load toward the hands. Keeping a full ground
      // squat there would bring the face down into the rock, especially number
      // twelve. This changes the person's posture only, never the stone path.
      const raisedReach = hands && pose.stone ? clamp((pose.stone.y-groundY-.12)/.18)*clamp((stoneDistance-.43)/.07) : 0;
      const supportedLoad = Math.max(raisedReach, hands && pose.stone ? clamp((pose.stone.y-groundY-.32)/.12) : 0);
      rig.raisedStoneReach = supportedLoad;
      rig.stoneBend = hands ? clamp(Number(pose.stone?.bend) || 0)*(1-.42*supportedLoad) : 0;
      rig.stoneReachAdjust = hands && pose.stone ? clamp((stoneDistance-.43)/.105)*rig.stoneBend : 0;
      rig.carryWeight = hands && pose.stone?.carrying ? clamp(Number(pose.stone.grip) || 0) : 0;
      if (hands) rig.workBlend = 0;
      const distance = Math.hypot(x - rig.x, z - rig.z);
      const fresh = !rig.initialized || !rig.wasVisible;
      const measuredSpeed = fresh || dt <= 0 ? 0 : distance / dt;
      rig.velocityX = fresh || dt <= 0 ? 0 : (x - rig.x) / dt;
      rig.velocityZ = fresh || dt <= 0 ? 0 : (z - rig.z) / dt;
      const displacementAlongFacing = (x - rig.x) * Math.sin(heading) + (z - rig.z) * Math.cos(heading);
      if (Math.abs(Number(pose.walkSpeed) || 0) > .012) rig.moveSign = Number(pose.walkSpeed) < 0 ? -1 : 1;
      else if (Math.abs(displacementAlongFacing) > .000001) rig.moveSign = displacementAlongFacing < 0 ? -1 : 1;
      rig.x = x; rig.z = z; rig.groundY = groundY;
      rig.stance = Number.isFinite(pose.stanceWidth)
        ? clamp(pose.stanceWidth, .08, .20) / (rig.scale * 2)
        : pose.narrowWalk ? Math.min(rig.footSpread, (rig.child ? .041 : .058) / rig.scale) : rig.footSpread;
      rig.toeOut = pose.narrowWalk ? .025 : .085;
      rig.turn = fresh ? 0 : clamp(wrapAngle(heading - rig.heading), -.5, .5);
      rig.heading = fresh ? heading : rig.heading + wrapAngle(heading - rig.heading) * (1 - Math.exp(-dt * 16));
      rig.group.position.set(x, groundY, z); rig.group.rotation.y = rig.heading;
      if (fresh) {
        initializeFeet(rig);
        const smooth = pose.activity === 'smooth' || pose.activity === 'sweep';
        rig.flip = smooth ? 1 : 0;
        rig.workBlend = pose.working || Number(pose.contact) > .018 ? 1 : 0;
        rig.leanAmount = (rig.workBlend ? .18 : .035) + rig.lean;
        const width = smooth ? rig.smoothWidth : rig.rakeWidth;
        rig.headScaleX = rig.toolScale * (Number.isFinite(pose.toolWidth) ? clamp(pose.toolWidth, .06, .78) / width : 1);
      }
      rig.initialized = true; rig.wasVisible = true;
      updateFeet(rig, pose, dt, time, measuredSpeed);
      if (hands) updateStoneArms(rig,pose,dt,time);
      else { rig.diagnostics.equipment='rake'; rig.diagnostics.stoneId=null; updateToolAndArms(rig, pose, dt, time); }
    }
  }

  function dispose() {
    root.removeFromParent();
    for (const rig of rigs) { rig.elbowCovers.dispose(); rig.kneeCovers.dispose(); }
    geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose());
    Object.values(textures).forEach(t => t.dispose());
  }

  return {
    members: rigs.map(({ id, name, group, height, toolWidth, rakeWidth, smoothWidth, reach, child }) => ({ id, name, group, height, toolWidth, rakeWidth, smoothWidth, reach, child: !!child })),
    update, dispose,
  };
}
