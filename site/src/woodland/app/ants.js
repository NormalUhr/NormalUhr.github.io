import * as THREE from 'three';

// World units: an adult worker is 0.64 units from abdomen to closed mandibles.
// Navigation owns the ground-level root. This module owns the articulated body
// and contact feet, and never mutates a worker or a stone passed by its caller.
const TAU = Math.PI * 2;
const UP = new THREE.Vector3(0, 1, 0);
const IDENTITY = new THREE.Quaternion();
const MOUTH = new THREE.Vector3(.115, -.018, 0);
const LEG = [
  { x: .073, z: .036, cx: .012, restX: .159, restZ: .180, upper: .118, lower: .159, bend: 1.35 },
  { x: .005, z: .041, cx: -.002, restX: -.004, restZ: .208, upper: .126, lower: .170, bend: -.83 },
  { x: -.066, z: .030, cx: -.014, restX: -.175, restZ: .184, upper: .137, lower: .179, bend: -1.38 }
];

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function angleDelta(a, b) { return Math.atan2(Math.sin(b - a), Math.cos(b - a)); }
function ease(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
function finite(v, fallback = 0) { return Number.isFinite(v) ? v : fallback; }
function random(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let n = value;
    n = Math.imul(n ^ n >>> 15, n | 1);
    n ^= n + Math.imul(n ^ n >>> 7, n | 61);
    return ((n ^ n >>> 14) >>> 0) / 4294967296;
  };
}

function catmull(p0, p1, p2, p3, t) {
  return .5 * ((2 * p1) + (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
}

// A closed longitudinal shell, with anatomical cross-sections rather than a
// row of intersecting spheres. The sutures are shallow geometric depressions.
function shellGeometry(profile, radial, subdivisions, sutures = false, head = false) {
  const positions = [], uv = [], colors = [], indices = [];
  const rings = (profile.length - 1) * subdivisions;
  for (let ring = 0; ring <= rings; ring++) {
    const u = ring / rings;
    const section = Math.min(profile.length - 2, Math.floor(u * (profile.length - 1)));
    const t = u * (profile.length - 1) - section;
    const p0 = profile[Math.max(0, section - 1)], p1 = profile[section];
    const p2 = profile[section + 1], p3 = profile[Math.min(profile.length - 1, section + 2)];
    const x = p1[0] + (p2[0] - p1[0]) * t;
    let radiusY = Math.max(.0004, catmull(p0[1], p1[1], p2[1], p3[1], t));
    let radiusZ = Math.max(.0004, catmull(p0[2], p1[2], p2[2], p3[2], t));
    const centerY = catmull(p0[3] || 0, p1[3] || 0, p2[3] || 0, p3[3] || 0, t);
    let groove = 0;
    if (sutures) {
      for (const band of [.29, .51, .71]) groove += Math.exp(-Math.pow((u - band) / .021, 2));
      radiusY *= 1 - .032 * groove;
      radiusZ *= 1 - .029 * groove;
    }
    for (let j = 0; j <= radial; j++) {
      const theta = j / radial * TAU, c = Math.cos(theta), s = Math.sin(theta);
      const sectionY = head ? Math.sign(c) * Math.pow(Math.abs(c), .83) : c;
      const minuteRidge = 1 + .007 * Math.sin(theta * 7 + x * 29);
      positions.push(x, centerY + sectionY * radiusY * minuteRidge, s * radiusZ * minuteRidge);
      uv.push(u * 2, j / radial);
      const ventral = .92 + .08 * Math.max(0, c);
      const shade = ventral * (1 - groove * .18);
      colors.push(shade, shade, shade);
      if (ring < rings && j < radial) {
        const a = ring * (radial + 1) + j, b = a + radial + 1;
        indices.push(a, a + 1, b, b, a + 1, b + 1);
      }
    }
  }
  for (let end = 0; end < 2; end++) {
    const point = profile[end ? profile.length - 1 : 0], center = positions.length / 3;
    positions.push(point[0], point[3] || 0, 0); uv.push(end * 2, .5); colors.push(.92, .92, .92);
    const offset = end ? rings * (radial + 1) : 0;
    for (let j = 0; j < radial; j++) {
      if (end) indices.push(center, offset + j, offset + j + 1);
      else indices.push(center, offset + j + 1, offset + j);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function cuticleTexture() {
  const edge = 64, data = new Uint8Array(edge * edge * 4), rnd = random(722913);
  for (let y = 0; y < edge; y++) for (let x = 0; x < edge; x++) {
    const offset = (y * edge + x) * 4;
    const pore = rnd() > .976 ? -32 : 0;
    const value = clamp(143 + (rnd() - .5) * 32 + Math.sin(y * .72 + Math.sin(x * .37)) * 8 + pore, 0, 255);
    data[offset] = data[offset + 1] = data[offset + 2] = value;
    data[offset + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, edge, edge, THREE.RGBAFormat);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.needsUpdate = true;
  return texture;
}

export function createAntColony(scene, ants, { heightAt = () => 0, groundHeightAt = heightAt,
  normalAt, quality = 'high' } = {}) {
  const group = new THREE.Group();
  group.name = 'Articulated worker colony';
  scene.add(group);
  const high = quality !== 'low', n = ants.length;
  const texture = cuticleTexture();
  const materials = {
    thorax: new THREE.MeshStandardMaterial({ color: 0x543522, roughness: .55, metalness: 0,
      bumpMap: texture, bumpScale: .0009, vertexColors: true }),
    head: new THREE.MeshStandardMaterial({ color: 0x38271d, roughness: .51, metalness: 0,
      bumpMap: texture, bumpScale: .00065, vertexColors: true }),
    abdomen: new THREE.MeshStandardMaterial({ color: 0x291e17, roughness: .47, metalness: 0,
      bumpMap: texture, bumpScale: .0007, vertexColors: true }),
    joints: new THREE.MeshStandardMaterial({ color: 0x4a2b18, roughness: .63, metalness: 0 }),
    legs: new THREE.MeshStandardMaterial({ color: 0x4a2d1c, roughness: .57, metalness: 0 }),
    eyes: new THREE.MeshStandardMaterial({ color: 0x100d0b, roughness: .22, metalness: 0 }),
    jaws: new THREE.MeshStandardMaterial({ color: 0x3d2413, roughness: .50, metalness: 0 }),
    antennae: new THREE.MeshStandardMaterial({ color: 0x422914, roughness: .65, metalness: 0 }),
    setae: new THREE.MeshStandardMaterial({ color: 0x99805d, roughness: .80, metalness: 0 })
  };
  const bodyRadial = high ? 24 : 12;
  const geometries = {
    head: shellGeometry([
      [-.066, .001, .002], [-.056, .034, .041], [-.032, .051, .066],
      [.006, .054, .069], [.037, .048, .060], [.060, .035, .042], [.073, .008, .013]
    ], bodyRadial, high ? 5 : 3, false, true),
    thorax: shellGeometry([
      [-.106, .009, .014, -.010], [-.084, .030, .031, -.001],
      [-.047, .033, .032, .002], [-.008, .032, .035, -.002],
      [.037, .046, .047, .008], [.080, .042, .046, .009], [.110, .014, .022, .009]
    ], bodyRadial, high ? 4 : 3),
    abdomen: shellGeometry([
      [-.109, .001, .001, -.006], [-.090, .035, .041, -.001],
      [-.050, .065, .071, .006], [.004, .073, .078, .008],
      [.053, .059, .064, .002], [.082, .030, .035, -.006], [.094, .003, .004, -.008]
    ], bodyRadial, high ? 6 : 3, true),
    node: new THREE.SphereGeometry(1, high ? 12 : 8, high ? 8 : 5),
    eye: new THREE.SphereGeometry(1, high ? 14 : 8, high ? 9 : 5),
    joint: new THREE.SphereGeometry(1, high ? 7 : 5, high ? 5 : 3),
    limb: new THREE.CylinderGeometry(.65, 1, 1, high ? 7 : 5, 1),
    fine: new THREE.CylinderGeometry(.52, 1, 1, high ? 6 : 4, 1),
    hair: new THREE.ConeGeometry(1, 1, 4, 1)
  };
  const meshes = {}, slots = {};
  function batch(name, geometry, material, count, shadows = true) {
    const mesh = new THREE.InstancedMesh(geometry, material, n * count);
    mesh.name = `Worker ${name}`;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.castShadow = shadows;
    mesh.receiveShadow = shadows;
    const color = new THREE.Color();
    ants.forEach((ant, index) => {
      const variation = .87 + random((Number(ant.id) || index + 1) * 7919)() * .21;
      color.setRGB(variation, variation * (.985 + index % 3 * .007), variation * .965);
      for (let slot = 0; slot < count; slot++) mesh.setColorAt(index * count + slot, color);
    });
    meshes[name] = mesh; slots[name] = count; group.add(mesh);
    return mesh;
  }
  batch('thorax', geometries.thorax, materials.thorax, 1);
  batch('head', geometries.head, materials.head, 1);
  batch('abdomen', geometries.abdomen, materials.abdomen, 1);
  batch('waist', geometries.node, materials.joints, 3);
  batch('eyes', geometries.eye, materials.eyes, 2);
  batch('legs', geometries.limb, materials.legs, 24);
  batch('joints', geometries.joint, materials.joints, 18);
  batch('jaws', geometries.fine, materials.jaws, 4);
  batch('antennae', geometries.fine, materials.antennae, 8);
  batch('claws', geometries.fine, materials.jaws, high ? 18 : 12, false);
  if (high) batch('setae', geometries.hair, materials.setae, 12, false);

  const rigs = new Map();
  const stats = { antCount: n, drawCalls: Object.keys(meshes).length,
    instances: Object.values(slots).reduce((a, b) => a + b, 0) * n,
    minSupport: 6, maxExtension: 0, totalSteps: 0, wetness: 0 };
  const scratch = {
    position: new THREE.Vector3(), scale: new THREE.Vector3(), direction: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(), euler: new THREE.Euler(), local: new THREE.Matrix4(),
    world: new THREE.Matrix4(), normal: new THREE.Vector3(), forward: new THREE.Vector3(),
    side: new THREE.Vector3(), upper: new THREE.Vector3(), pole: new THREE.Vector3(),
    a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3(), d: new THREE.Vector3(),
    base: new THREE.Vector3(), basis: new THREE.Matrix4(), yaw: new THREE.Quaternion(),
    supportNormal: new THREE.Vector3(), triangleA: new THREE.Vector3(), triangleB: new THREE.Vector3(),
    unitScale: new THREE.Vector3(1, 1, 1)
  };

  function ground(x, z) { return finite(heightAt(x, z)); }
  function terrainNormal(x, z, result) {
    const normal = normalAt?.(x, z);
    if (normal && Number.isFinite(normal.x)) result.set(normal.x, normal.y, normal.z);
    else if (normal && Number.isFinite(normal[0])) result.set(normal[0], normal[1], normal[2]);
    else {
      const h = .045;
      result.set(ground(x - h, z) - ground(x + h, z), 2 * h,
        ground(x, z - h) - ground(x, z + h));
    }
    if (!Number.isFinite(result.lengthSq()) || result.lengthSq() < 1e-8) result.copy(UP);
    if (result.y < .15) result.y = .15;
    return result.normalize();
  }

  function writeLocal(name, index, parent, x = 0, y = 0, z = 0,
    sx = 1, sy = 1, sz = 1, pitch = 0, yaw = 0) {
    scratch.position.set(x, y, z); scratch.scale.set(sx, sy, sz);
    if (pitch || yaw) scratch.quaternion.setFromEuler(scratch.euler.set(0, yaw, pitch, 'YXZ'));
    else scratch.quaternion.copy(IDENTITY);
    scratch.local.compose(scratch.position, scratch.quaternion, scratch.scale);
    scratch.world.multiplyMatrices(parent, scratch.local);
    meshes[name].setMatrixAt(index, scratch.world);
  }

  function writeSegment(name, index, a, b, radius) {
    scratch.direction.subVectors(b, a);
    const length = Math.max(.00001, scratch.direction.length());
    scratch.direction.multiplyScalar(1 / length);
    scratch.quaternion.setFromUnitVectors(UP, scratch.direction);
    scratch.position.addVectors(a, b).multiplyScalar(.5);
    scratch.scale.set(radius, length, radius);
    scratch.world.compose(scratch.position, scratch.quaternion, scratch.scale);
    meshes[name].setMatrixAt(index, scratch.world);
  }

  function writeJoint(index, p, radius) {
    scratch.scale.setScalar(radius);
    scratch.world.compose(p, IDENTITY, scratch.scale);
    meshes.joints.setMatrixAt(index, scratch.world);
  }

  function createRig(ant, index) {
    const size = clamp(finite(ant.size, 1), .35, 2);
    const root = new THREE.Vector3(finite(ant.x), finite(ant.y, ground(ant.x, ant.z)), finite(ant.z));
    const normal = terrainNormal(root.x, root.z, new THREE.Vector3());
    const heading = finite(ant.angle), c = Math.cos(heading), s = Math.sin(heading);
    const feet = [];
    for (let leg = 0; leg < 3; leg++) for (const side of [-1, 1]) {
      const def = LEG[leg], x = root.x + (c * def.restX - s * side * def.restZ) * size;
      const z = root.z + (s * def.restX + c * side * def.restZ) * size;
      const p = new THREE.Vector3(x, ground(x, z) + .003 * size, z);
      feet.push({ leg, side, group: (leg + (side === 1 ? 1 : 0)) % 2,
        phase: 'stance', p, from: p.clone(), to: p.clone(), progress: 0,
        duration: .1, stepHeight: .038 * size, stepSerial: 0,
        hip: new THREE.Vector3(), coxa: new THREE.Vector3(), knee: new THREE.Vector3(),
        ankle: new THREE.Vector3(), bodyOffset: new THREE.Vector3(), supportAnkle: new THREE.Vector3(),
        extension: 0, support: true });
    }
    const rig = { ant, index, size, feet, previousRoot: root.clone(), previousAngle: heading,
      root: root.clone(), angle: heading, normal, bodyMatrix: new THREE.Matrix4(),
      headMatrix: new THREE.Matrix4(), abdomenMatrix: new THREE.Matrix4(),
      forward: new THREE.Vector3(c, 0, s), side: new THREE.Vector3(-s, 0, c),
      velocity: new THREE.Vector3(), angular: 0, motion: 0, load: 0,
      gait: random(index * 331 + 72)() * TAU, phase: random(index * 87 + 915)() * TAU,
      clock: 0, distance: 0, idleTime: 0, nextGroup: index % 2, hasStepped: false,
      headPitch: 0, appliedHeadPitch: 0, flex: 0, rain: 0, maxExtension: 0, minSupport: 6, mouth: new THREE.Vector3(),
      bodyHeightAdjustment: 0, gripClearanceError: 0 };
    rigs.set(ant, rig);
    return rig;
  }

  // Matrices use local +x forward, +y up, +z right. The ant.angle convention is
  // clockwise in the x/z plane, opposite Three's positive rotation about +y.
  function carryHeadPitch(rig, angle) {
    const stone = rig.ant.hold;
    rig.gripClearanceError = 0;
    if (!stone || !(stone.radius > 0)) return rig.headPitch;
    const radius = stone.radius, e = rig.bodyMatrix.elements;
    const yaw = -rig.flex * .50, cy = Math.cos(yaw), sy = Math.sin(yaw);
    const forwardX = Math.cos(angle) * radius * .65, forwardZ = Math.sin(angle) * radius * .65;
    // Evaluate the actual mandible tip around its neck joint. Raising the
    // stone to clear a mound must raise this articulated head with it; a
    // separate vertical clamp on the stone used to leave a visible air gap.
    const gapAt = pitch => {
      const c = Math.cos(pitch), s = Math.sin(pitch);
      const dx = (.071 + MOUTH.x) * c - MOUTH.y * s;
      const x = .123 + dx * cy, y = .012 + (.071 + MOUTH.x) * s + MOUTH.y * c, z = -dx * sy;
      const worldX = e[0] * x + e[4] * y + e[8] * z + e[12] + forwardX;
      const worldY = e[1] * x + e[5] * y + e[9] * z + e[13] + radius * .30;
      const worldZ = e[2] * x + e[6] * y + e[10] * z + e[14] + forwardZ;
      return finite(groundHeightAt(worldX, worldZ)) + radius * .76 + .051 - worldY;
    };
    let pitch = rig.headPitch, gap = gapAt(pitch);
    if (gap > 0) {
      let low = pitch, best = pitch, bestGap = gap;
      const maximum = 1.28;
      for (let high = Math.min(maximum, low + .16); high <= maximum + .000001;
        high = Math.min(maximum, high + .16)) {
        const candidateGap = gapAt(high);
        if (candidateGap < bestGap) { best = high; bestGap = candidateGap; }
        if (candidateGap <= 0) {
          // The first crossing is the smallest neck lift that clears the
          // ground. The continuous terrain and bracket keep its path smooth.
          let upper = high;
          for (let iteration = 0; iteration < 12; iteration++) {
            const middle = (low + upper) * .5;
            if (gapAt(middle) > 0) low = middle; else upper = middle;
          }
          best = upper; break;
        }
        if (high >= maximum) break;
        low = high;
      }
      pitch = best;
    }
    if (rig.ant.state === 'pickup') {
      const t = ease(clamp(finite(rig.ant.phaseTime) / Math.max(.01, finite(rig.ant.phaseDuration, 1)), 0, 1));
      const neutral = -.095 - rig.load * .018;
      pitch = Math.max(rig.headPitch, neutral + (pitch - neutral) * t);
    }
    rig.gripClearanceError = Math.max(0, gapAt(pitch) - .006);
    return pitch;
  }

  function composePose(rig, root = rig.root, angle = rig.angle) {
    scratch.forward.set(Math.cos(angle), 0, Math.sin(angle));
    scratch.forward.addScaledVector(rig.normal, -scratch.forward.dot(rig.normal)).normalize();
    scratch.side.crossVectors(scratch.forward, rig.normal).normalize();
    scratch.upper.crossVectors(scratch.side, scratch.forward).normalize();
    rig.forward.copy(scratch.forward); rig.side.copy(scratch.side);
    scratch.basis.makeBasis(scratch.forward, scratch.upper, scratch.side);
    scratch.quaternion.setFromRotationMatrix(scratch.basis);
    const breath = Math.sin(rig.clock * 2.3 + rig.phase) * .0009;
    const heave = .130 - rig.load * .010 - rig.rain * .004 +
      Math.sin(rig.gait * 2) * .006 * rig.motion + breath;
    scratch.position.copy(root).addScaledVector(scratch.upper, heave * rig.size);
    // A root can cross a pebble crest faster than a support tripod can lift.
    // Fit body height to the actual six reach spheres; the thorax crouches or
    // descends with its feet instead of stretching rigid limb segments. This
    // continuous min/max constraint does not move any locked contact or root.
    const desiredY = scratch.position.y;
    let minimumY = -Infinity, maximumY = Infinity;
    for (const foot of rig.feet) {
      const def = LEG[foot.leg], s = rig.size;
      const lx = (def.x + def.cx) * s, ly = -.029 * s, lz = foot.side * (def.z + .024) * s;
      const ox = scratch.forward.x * lx + scratch.upper.x * ly + scratch.side.x * lz;
      const oy = scratch.forward.y * lx + scratch.upper.y * ly + scratch.side.y * lz;
      const oz = scratch.forward.z * lx + scratch.upper.z * ly + scratch.side.z * lz;
      const ax = foot.p.x - scratch.forward.x * .020 * s + rig.normal.x * .013 * s;
      const ay = foot.p.y - scratch.forward.y * .020 * s + rig.normal.y * .013 * s;
      const az = foot.p.z - scratch.forward.z * .020 * s + rig.normal.z * .013 * s;
      foot.bodyOffset.set(ox, oy, oz); foot.supportAnkle.set(ax, ay, az);
      const dx = ax - scratch.position.x - ox, dz = az - scratch.position.z - oz;
      const reach = (def.upper + def.lower) * s * .975;
      const vertical = Math.sqrt(Math.max(.00001, reach * reach - dx * dx - dz * dz));
      minimumY = Math.max(minimumY, ay - vertical - oy);
      maximumY = Math.min(maximumY, ay + vertical - oy);
    }
    scratch.position.y = minimumY <= maximumY ? clamp(desiredY, minimumY, maximumY) : (minimumY + maximumY) * .5;
    // A tiny horizontal lean resolves the remaining reach at a sharp ridge or
    // turn. It moves the torso within its support, never a grounded toe. The
    // reserve keeps both rigid IK segments away from their singular straight pose.
    for (let pass = 0; pass < 12; pass++) {
      let excess = 0;
      for (const foot of rig.feet) {
        const def = LEG[foot.leg], reach = (def.upper + def.lower) * rig.size * .975;
        const folded = (Math.abs(def.lower - def.upper) + .004) * rig.size;
        scratch.a.copy(foot.supportAnkle).sub(foot.bodyOffset).sub(scratch.position);
        const distance = scratch.a.length();
        if (distance > reach) {
          excess = Math.max(excess, distance - reach);
          scratch.position.addScaledVector(scratch.a, (distance - reach) / distance);
        } else if (distance < folded) {
          excess = Math.max(excess, folded - distance);
          if (distance > .00001) scratch.position.addScaledVector(scratch.a, -(folded - distance) / distance);
          else scratch.position.addScaledVector(rig.normal, folded);
        }
      }
      if (excess < .000001) break;
    }
    rig.bodyHeightAdjustment = scratch.position.y - desiredY;
    scratch.scale.setScalar(rig.size);
    rig.bodyMatrix.compose(scratch.position, scratch.quaternion, scratch.scale);
    rig.appliedHeadPitch = carryHeadPitch(rig, angle);
    scratch.position.set(.123, .012, 0);
    scratch.quaternion.setFromEuler(scratch.euler.set(0, -rig.flex * .50, rig.appliedHeadPitch, 'YXZ'));
    scratch.local.compose(scratch.position, scratch.quaternion, scratch.unitScale);
    rig.headMatrix.multiplyMatrices(rig.bodyMatrix, scratch.local);
    scratch.local.makeTranslation(.071, 0, 0);
    rig.headMatrix.multiply(scratch.local);
    scratch.position.set(-.222, .012, 0);
    scratch.quaternion.setFromEuler(scratch.euler.set(0, rig.flex, -.014 * rig.load, 'YXZ'));
    scratch.local.compose(scratch.position, scratch.quaternion, scratch.unitScale);
    rig.abdomenMatrix.multiplyMatrices(rig.bodyMatrix, scratch.local);
    rig.mouth.copy(MOUTH).applyMatrix4(rig.headMatrix);
  }

  function hipPoints(rig, foot) {
    const def = LEG[foot.leg];
    foot.hip.set(def.x, -.021, foot.side * def.z).applyMatrix4(rig.bodyMatrix);
    foot.coxa.set(def.x + def.cx, -.029, foot.side * (def.z + .024)).applyMatrix4(rig.bodyMatrix);
    // The terminal tarsus lies partly along the substrate, ending exactly at
    // the persistent world-space contact. It is never reprojected while planted.
    foot.ankle.copy(foot.p).addScaledVector(rig.forward, -.020 * rig.size)
      .addScaledVector(rig.normal, .013 * rig.size);
  }

  function chooseDuration(rig, group, effort, stationary) {
    const stride = (.170 - rig.load * .021) * rig.size;
    const cycle = stride / Math.max(.14 * rig.size, effort);
    let duration = stationary ? .13 : clamp(cycle * (.29 - .055 * rig.load), .017, .135);
    // Finish the swing before its support tripod reaches full extension. This
    // is particularly important when the root pivots or moves sideways.
    if (!stationary && effort > .005) {
      let margin = Infinity;
      for (const foot of rig.feet) if (foot.group !== group) {
        hipPoints(rig, foot);
        const def = LEG[foot.leg], reach = (def.upper + def.lower) * rig.size;
        const d = foot.coxa.distanceTo(foot.ankle);
        margin = Math.min(margin, Math.max(.002, reach * .93 - d));
      }
      duration = Math.min(duration, clamp(margin / effort * .63, .013, .135));
    }
    return { duration, cycle };
  }

  function startStep(rig, group, effort, stationary = false) {
    const { duration, cycle } = chooseDuration(rig, group, effort, stationary);
    const prediction = stationary ? 0 : duration + cycle * .29;
    const angle = rig.angle + clamp(rig.angular * prediction, -.35, .35);
    const c = Math.cos(angle), s = Math.sin(angle);
    const vx = rig.velocity.x, vz = rig.velocity.z, speed = Math.hypot(vx, vz);
    const lead = stationary ? 0 : Math.min(speed * prediction, .097 * rig.size);
    const px = rig.root.x + (speed > .00001 ? vx / speed * lead : 0);
    const pz = rig.root.z + (speed > .00001 ? vz / speed * lead : 0);
    for (const foot of rig.feet) if (foot.group === group) {
      const def = LEG[foot.leg], spread = (def.restZ + rig.load * .010) * foot.side;
      let x = px + (def.restX * c - spread * s) * rig.size;
      let z = pz + (def.restX * s + spread * c) * rig.size;
      let y = ground(x, z) + .003 * rig.size;
      const desiredX = x, desiredZ = z;
      // Search nearby footholds when a small stone lies between the nominal
      // feet. A worker can narrow its stance onto the stone's actual surface.
      const rootY = rig.root.y + rig.velocity.y * Math.min(prediction, .07);
      const hipX = px + ((def.x + def.cx) * c - foot.side * (def.z + .024) * s) * rig.size;
      const hipZ = pz + ((def.x + def.cx) * s + foot.side * (def.z + .024) * c) * rig.size;
      const hipY = rootY + (.101 - rig.load * .010) * rig.size;
      const reach = (def.upper + def.lower) * rig.size;
      const scoreAt = (tx, ty, tz) => {
        const distance = Math.hypot(tx - hipX - c * .020 * rig.size,
          ty + .013 * rig.size - hipY, tz - hipZ - s * .020 * rig.size);
        return Math.hypot(tx - desiredX, tz - desiredZ) / rig.size +
          Math.max(0, distance / reach - .84) * 16 +
          Math.max(0, (Math.abs(def.lower - def.upper) + .014) * rig.size - distance) / rig.size * 16;
      };
      let best = scoreAt(x, y, z);
      if (best > .12) for (const narrow of [.82, .63, .44, .30]) for (const fore of [1, .72, .44]) {
        const lx = def.restX * fore, lz = spread * narrow;
        const tx = px + (lx * c - lz * s) * rig.size, tz = pz + (lx * s + lz * c) * rig.size;
        const ty = ground(tx, tz) + .003 * rig.size, score = scoreAt(tx, ty, tz);
        if (score < best) { best = score; x = tx; y = ty; z = tz; }
      }
      foot.from.copy(foot.p); foot.to.set(x, y, z);
      const mx = (foot.from.x + x) * .5, mz = (foot.from.z + z) * .5;
      const obstacle = Math.max(0, ground(mx, mz) - (foot.from.y + foot.to.y) * .5);
      foot.progress = 0; foot.duration = duration;
      foot.stepHeight = (.039 - rig.load * .009 + rig.rain * .003) * rig.size + obstacle;
      foot.phase = 'swing'; foot.support = false; foot.stepSerial++;
      stats.totalSteps++;
    }
    rig.nextGroup = 1 - group; rig.hasStepped = true;
  }

  function stepRig(rig, ant, dt, effort, wetness, rain) {
    rig.clock += dt;
    const response = 1 - Math.exp(-dt * 12);
    rig.motion += (clamp(effort / (.65 * rig.size), 0, 1) - rig.motion) * response;
    const load = ant.hold ? clamp((ant.hold.radius || .17) / (.18 * rig.size), .5, 1) : 0;
    rig.load += (load - rig.load) * (1 - Math.exp(-dt * 8));
    rig.rain += (rain - rig.rain) * (1 - Math.exp(-dt * 4));
    const working = /pickup|pick|place|alignSource|alignDest/.test(String(ant.state || ''));
    const headTarget = (working ? -.095 : -.018) - rig.load * .018;
    rig.headPitch += (headTarget - rig.headPitch) * (1 - Math.exp(-dt * 10));
    rig.flex += (clamp(rig.angular * .028, -.095, .095) * (1 - rig.load * .60) - rig.flex) * response;
    terrainNormal(rig.root.x, rig.root.z, scratch.normal);
    // A pebble's point normal is too abrupt for an insect spanning both sides.
    // Use the plane under its support tripod as the dominant body inclination.
    scratch.supportNormal.set(0, 0, 0);
    for (const group of [0, 1]) {
      const tripod = rig.feet.filter(foot => foot.group === group && foot.phase === 'stance');
      if (tripod.length !== 3) continue;
      scratch.triangleA.subVectors(tripod[1].p, tripod[0].p);
      scratch.triangleB.subVectors(tripod[2].p, tripod[0].p);
      scratch.triangleA.cross(scratch.triangleB);
      if (scratch.triangleA.y < 0) scratch.triangleA.negate();
      if (scratch.triangleA.lengthSq() > 1e-9) scratch.supportNormal.add(scratch.triangleA.normalize());
    }
    if (scratch.supportNormal.lengthSq() > 1e-9) scratch.normal.multiplyScalar(.28)
      .addScaledVector(scratch.supportNormal.normalize(), .72).normalize();
    rig.normal.lerp(scratch.normal, 1 - Math.exp(-dt * 14)).normalize();
    rig.gait += effort * dt / (.17 * rig.size) * TAU;
    rig.distance += effort * dt;
    rig.idleTime = effort < .012 ? rig.idleTime + dt : 0;

    for (const foot of rig.feet) if (foot.phase === 'swing') {
      foot.progress = Math.min(1, foot.progress + dt / foot.duration);
      const t = ease(foot.progress), arc = Math.sin(Math.PI * foot.progress);
      foot.p.lerpVectors(foot.from, foot.to, t);
      foot.p.y += foot.stepHeight * arc * arc;
      if (foot.progress >= 1) {
        foot.p.copy(foot.to); foot.phase = 'stance'; foot.support = true;
      }
    }
    composePose(rig);
    // Preserve the lifted pose as the next step's starting point, so the neck
    // settles smoothly after crossing the crest or putting its load down.
    rig.headPitch = rig.appliedHeadPitch;
    if (!rig.feet.some(foot => foot.phase === 'swing')) {
      let most = .855, group = -1;
      for (const foot of rig.feet) {
        hipPoints(rig, foot);
        const def = LEG[foot.leg];
        const extension = foot.coxa.distanceTo(foot.ankle) / ((def.upper + def.lower) * rig.size);
        if (extension > most) { most = extension; group = foot.group; }
      }
      const halfStride = (.170 - rig.load * .021) * rig.size * .5;
      if (effort > .012 && (!rig.hasStepped || rig.distance >= halfStride || group >= 0)) {
        // A stretched newly landed tripod must not immediately swing again:
        // that would strand the other three locked feet behind a moving root.
        // Alternate the support groups, using strain only to start the next
        // step early. chooseDuration still protects the current support reach.
        startStep(rig, rig.nextGroup, effort);
        rig.distance = Math.max(0, rig.distance - halfStride);
      } else if (rig.idleTime > .24 && group >= 0) {
        startStep(rig, group, 0, true); rig.distance = 0;
      }
    }
  }

  function solveLeg(rig, foot) {
    hipPoints(rig, foot);
    const def = LEG[foot.leg], a = def.upper * rig.size, b = def.lower * rig.size;
    scratch.direction.subVectors(foot.ankle, foot.coxa);
    const actual = scratch.direction.length();
    foot.extension = actual / (a + b);
    rig.maxExtension = Math.max(rig.maxExtension, foot.extension);
    const d = clamp(actual, Math.abs(a - b) + .00001, a + b - .00001);
    scratch.direction.multiplyScalar(1 / Math.max(actual, .00001));
    const along = (a * a - b * b + d * d) / (2 * d);
    const bend = Math.sqrt(Math.max(0, a * a - along * along));
    scratch.pole.copy(rig.normal).multiplyScalar(.28)
      .addScaledVector(rig.side, foot.side * .48).addScaledVector(rig.forward, def.bend);
    scratch.pole.addScaledVector(scratch.direction, -scratch.pole.dot(scratch.direction));
    if (scratch.pole.lengthSq() < 1e-9) scratch.pole.copy(rig.normal);
    scratch.pole.normalize();
    foot.knee.copy(foot.coxa).addScaledVector(scratch.direction, along).addScaledVector(scratch.pole, bend);
  }

  function renderRig(rig) {
    const i = rig.index, size = rig.size;
    composePose(rig);
    writeLocal('thorax', i, rig.bodyMatrix);
    writeLocal('head', i, rig.headMatrix);
    writeLocal('abdomen', i, rig.abdomenMatrix);
    writeLocal('waist', i * 3, rig.bodyMatrix, -.119, .003, 0, .023, .037, .021, -.10);
    writeLocal('waist', i * 3 + 1, rig.bodyMatrix, -.142, -.013, 0, .036, .012, .014);
    writeLocal('waist', i * 3 + 2, rig.bodyMatrix, .118, .011, 0, .021, .022, .023);
    for (let side = -1; side <= 1; side += 2) {
      const eye = side === -1 ? 0 : 1;
      writeLocal('eyes', i * 2 + eye, rig.headMatrix, -.005, .012, side * .063, .017, .019, .010, 0, side * .12);
      const sweep = Math.sin(rig.clock * (rig.motion > .1 ? 5.3 : 3.1) + rig.phase + side * 1.2) * .016;
      const feel = Math.sin(rig.clock * 2.2 + rig.phase + side) * .010;
      // Scape, elbow and a segmented funiculus are full 3D geometry. The distal
      // antenna bends toward the soil while a loaded worker keeps it wider.
      const points = [
        [.038, .030, side * .039], [.106 + sweep, .052 + feel, side * (.093 + rig.load * .012)],
        [.143 + sweep * .7, .031 + feel, side * (.116 + rig.load * .012)],
        [.167 + sweep * .5, .006 + feel, side * (.125 + rig.load * .012)],
        [.183 + sweep * .3, -.008 + feel, side * (.122 + rig.load * .012)]
      ];
      for (let section = 0; section < 4; section++) {
        scratch.a.fromArray(points[section]).applyMatrix4(rig.headMatrix);
        scratch.b.fromArray(points[section + 1]).applyMatrix4(rig.headMatrix);
        writeSegment('antennae', i * 8 + eye * 4 + section, scratch.a, scratch.b,
          (section === 0 ? .0055 : .0037 - section * .00040) * size);
      }
      const jawOpen = .003 + (1 - rig.load) * (.005 + Math.sin(rig.clock * 1.7 + rig.phase) * .002);
      const jaws = [[.062, -.018, side * .038], [.095, -.024, side * (.027 + jawOpen)],
        [.115, -.018, side * (.010 + jawOpen * .35)]];
      for (let section = 0; section < 2; section++) {
        scratch.a.fromArray(jaws[section]).applyMatrix4(rig.headMatrix);
        scratch.b.fromArray(jaws[section + 1]).applyMatrix4(rig.headMatrix);
        writeSegment('jaws', i * 4 + eye * 2 + section, scratch.a, scratch.b, (.009 - section * .0026) * size);
      }
      if (high) for (let tooth = 0; tooth < 3; tooth++) {
        const x = .090 + tooth * .007, z = side * (.030 - tooth * .004 + jawOpen * .4);
        scratch.a.set(x, -.023, z).applyMatrix4(rig.headMatrix);
        scratch.b.set(x + .007, -.020, z - side * .009).applyMatrix4(rig.headMatrix);
        writeSegment('claws', i * slots.claws + 12 + eye * 3 + tooth, scratch.a, scratch.b, .0025 * size);
      }
    }
    let support = 0;
    for (let footIndex = 0; footIndex < rig.feet.length; footIndex++) {
      const foot = rig.feet[footIndex];
      solveLeg(rig, foot);
      if (foot.phase === 'stance') support++;
      const offset = i * 24 + footIndex * 4;
      writeSegment('legs', offset, foot.hip, foot.coxa, .0110 * size);
      writeSegment('legs', offset + 1, foot.coxa, foot.knee, .0086 * size);
      writeSegment('legs', offset + 2, foot.knee, foot.ankle, .0064 * size);
      writeSegment('legs', offset + 3, foot.ankle, foot.p, .0036 * size);
      writeJoint(i * 18 + footIndex * 3, foot.coxa, .0104 * size);
      writeJoint(i * 18 + footIndex * 3 + 1, foot.knee, .0076 * size);
      writeJoint(i * 18 + footIndex * 3 + 2, foot.ankle, .0043 * size);
      for (let claw = 0; claw < 2; claw++) {
        scratch.a.copy(foot.p).addScaledVector(rig.forward, -.003 * size)
          .addScaledVector(rig.normal, .003 * size);
        scratch.b.copy(foot.p).addScaledVector(rig.forward, .009 * size)
          .addScaledVector(rig.side, (claw ? 1 : -1) * .003 * size);
        writeSegment('claws', i * slots.claws + footIndex * 2 + claw, scratch.a, scratch.b, .0019 * size);
      }
    }
    rig.minSupport = Math.min(rig.minSupport, support);
    stats.minSupport = Math.min(stats.minSupport, support);
    stats.maxExtension = Math.max(stats.maxExtension, rig.maxExtension);
    if (high) for (let hair = 0; hair < 12; hair++) {
      const ring = hair % 4, theta = (hair < 8 ? .24 : -.24) + Math.floor(hair / 4) * 1.6;
      const x = -.070 + ring * .038, taper = Math.sqrt(Math.max(.13, 1 - Math.pow(x / .115, 2)));
      scratch.a.set(x, Math.cos(theta) * .069 * taper + .008, Math.sin(theta) * .074 * taper).applyMatrix4(rig.abdomenMatrix);
      scratch.b.set(x - .005, Math.cos(theta) * (.069 * taper + .009) + .008,
        Math.sin(theta) * (.074 * taper + .009)).applyMatrix4(rig.abdomenMatrix);
      writeSegment('setae', i * 12 + hair, scratch.a, scratch.b, .00070 * size);
    }
  }

  ants.forEach(createRig);
  // Choose reachable initial contacts before the first rendered frame. A
  // nominal footprint must not straddle a stone pile or a sharp soil ridge.
  for (const rig of rigs.values()) {
    composePose(rig);
    for (const group of [0, 1]) {
      startStep(rig, group, 0, true);
      for (const foot of rig.feet) if (foot.group === group) {
        foot.p.copy(foot.to); foot.from.copy(foot.to); foot.phase = 'stance';
        foot.progress = 0; foot.support = true; foot.stepSerial = 0;
      }
    }
    rig.hasStepped = false; rig.nextGroup = rig.index % 2;
  }
  stats.totalSteps = 0;
  let disposed = false;
  function update(dt = 0, time = 0, env = {}) {
    if (disposed) return;
    dt = Math.max(0, finite(dt));
    const wetness = clamp(finite(env.wetness), 0, 1), rain = clamp(finite(env.rain), 0, 1);
    stats.wetness += (wetness - stats.wetness) * (1 - Math.exp(-dt * 1.7));
    materials.thorax.roughness = .55 - stats.wetness * .12;
    materials.head.roughness = .51 - stats.wetness * .12;
    materials.abdomen.roughness = .47 - stats.wetness * .11;
    materials.legs.roughness = .57 - stats.wetness * .10;
    for (const ant of ants) {
      const rig = rigs.get(ant);
      if (!rig) continue;
      scratch.base.set(finite(ant.x), finite(ant.y, ground(ant.x, ant.z)), finite(ant.z));
      const x = scratch.base.x, y = scratch.base.y, z = scratch.base.z, angle = finite(ant.angle);
      const da = angleDelta(rig.previousAngle, angle);
      if (dt > 0) {
        rig.velocity.set((x - rig.previousRoot.x) / dt, (y - rig.previousRoot.y) / dt,
          (z - rig.previousRoot.z) / dt);
        rig.angular = da / dt;
        const effort = Math.hypot(rig.velocity.x, rig.velocity.z) + Math.abs(rig.velocity.y) * 1.1 +
          Math.abs(rig.angular) * .17 * rig.size;
        // Substep a moving root without writing intermediate values into ant.
        // Large display frames consequently do not skip a planted support phase.
        const steps = Math.max(1, Math.ceil(dt / (1 / 150))), h = dt / steps;
        for (let step = 1; step <= steps; step++) {
          const t = step / steps;
          rig.root.set(rig.previousRoot.x + (x - rig.previousRoot.x) * t,
            rig.previousRoot.y + (y - rig.previousRoot.y) * t,
            rig.previousRoot.z + (z - rig.previousRoot.z) * t);
          rig.angle = rig.previousAngle + da * t;
          stepRig(rig, ant, h, effort, wetness, rain);
        }
      } else {
        rig.root.set(x, y, z); rig.angle = angle;
      }
      rig.previousRoot.set(x, y, z); rig.previousAngle = angle;
      renderRig(rig);
    }
    for (const mesh of Object.values(meshes)) mesh.instanceMatrix.needsUpdate = true;
  }

  function mouthPosition(ant, result = new THREE.Vector3()) {
    const rig = rigs.get(ant);
    if (!rig) return result.set(finite(ant.x), finite(ant.y, ground(ant.x, ant.z)) + .124,
      finite(ant.z));
    // Also works between navigation and render updates. Current logical root,
    // same smoothed head/stance pose: no stale one-frame mouth attachment.
    scratch.base.set(finite(ant.x), finite(ant.y, ground(ant.x, ant.z)), finite(ant.z));
    composePose(rig, scratch.base, finite(ant.angle));
    return result.copy(rig.mouth);
  }

  function dispose() {
    if (disposed) return;
    disposed = true; scene.remove(group);
    for (const mesh of Object.values(meshes)) mesh.dispose();
    for (const geometry of Object.values(geometries)) geometry.dispose();
    for (const material of Object.values(materials)) material.dispose();
    texture.dispose(); rigs.clear();
  }

  update(0, 0, {});
  return { update, mouthPosition, dispose, stats, rigs, group, meshes };
}
