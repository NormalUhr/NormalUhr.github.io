import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Architectural lamps, in metres, independent of the scene's exposure.
 * Fixed fittings are batched by finish; cabin fittings inherit the real car's
 * transform. Point lights only supplement nearby surfaces and never cast shadows.
 */
export function createNightLighting(THREE, scene, {
  architecture, lifts, clippingPlanes = [], lowPower = false,
} = {}) {
  if (!architecture?.floors?.length || !architecture.central?.shafts || !lifts?.cabins) {
    throw new Error('Night lighting requires the built galleries and actual lift cabins');
  }
  const TAU = Math.PI * 2, DEG = Math.PI / 180;
  const group = new THREE.Group();
  group.name = 'Warm gallery and lift lighting';
  scene.add(group);
  const materials = [], meshes = [], lights = [], cabinGroups = [];
  const fixtures = { gallery: [], fascia: [], vertical: [], headers: [], cabins: [] };
  const lightSpecs = [];
  const makeMaterial = (name, color, roughness, metalness) => {
    const material = new THREE.MeshStandardMaterial({
      color, roughness, metalness, clippingPlanes, clipShadows: true,
    });
    material.name = 'Night lamps / ' + name;
    materials.push(material);
    return material;
  };
  const housing = makeMaterial('blackened bronze', '#443f35', .86, .20);
  const brass = makeMaterial('aged brass edges', '#ac9061', .81, .25);
  const opal = makeMaterial('warm ivory diffuser', '#e9d7b2', .84, 0);
  opal.emissive.set('#ffc178');

  const staticBuckets = new Map();
  const boxTemplate = new THREE.BoxGeometry(1, 1, 1).toNonIndexed();
  const beamTemplate = new THREE.CylinderGeometry(1, 1, 1, 6, 1).toNonIndexed();
  const polar = (radius, angle, y = 0) => new THREE.Vector3(Math.sin(angle) * radius, y, -Math.cos(angle) * radius);
  let triangles = 0;
  function put(buckets, material, geometry) {
    if (geometry.index) {
      const indexed = geometry;
      geometry = geometry.toNonIndexed();
      indexed.dispose();
    }
    for (const name of Object.keys(geometry.attributes)) {
      if (!['position', 'normal', 'uv'].includes(name)) geometry.deleteAttribute(name);
    }
    if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
    if (!geometry.getAttribute('uv')) {
      geometry.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(geometry.getAttribute('position').count * 2), 2));
    }
    if (!buckets.has(material)) buckets.set(material, []);
    buckets.get(material).push(geometry);
  }
  function box(buckets, material, centre, size, yaw = 0) {
    const geometry = boxTemplate.clone().scale(size[0], size[1], size[2]);
    if (yaw) geometry.rotateY(yaw);
    geometry.translate(centre.x, centre.y, centre.z);
    put(buckets, material, geometry);
  }
  function radialBox(buckets, material, radius, angle, y, width, height, depth) {
    box(buckets, material, polar(radius, angle, y), [width, height, depth], -angle);
  }
  function beam(buckets, material, from, to, radius) {
    const delta = to.clone().sub(from), length = delta.length();
    if (length < 1e-8) return;
    const geometry = beamTemplate.clone().scale(radius, length, radius);
    geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize()));
    geometry.translate((from.x + to.x) / 2, (from.y + to.y) / 2, (from.z + to.z) / 2);
    put(buckets, material, geometry);
  }
  function cylinder(buckets, material, centre, top, bottom, height, sides = 10) {
    const geometry = new THREE.CylinderGeometry(top, bottom, height, sides, 1);
    geometry.translate(centre.x, centre.y, centre.z);
    put(buckets, material, geometry);
  }
  // A real, closed band segment: correctly wound front, back, end and cap faces.
  // Adjacent bays have a physical seam; housing and diffuser are separate shells.
  function band(buckets, material, inner, outer, bottom, top, start, end) {
    const positions = [], normals = [], uvs = [];
    function triangle(a, b, c, n) {
      for (const p of [a, b, c]) {
        positions.push(p.x, p.y, p.z); normals.push(n.x, n.y, n.z); uvs.push(p.x, p.z);
      }
    }
    function quad(a, b, c, d, normal) {
      if (b.clone().sub(a).cross(c.clone().sub(a)).dot(normal) < 0) {
        triangle(a, c, b, normal); triangle(a, d, c, normal);
      } else {
        triangle(a, b, c, normal); triangle(a, c, d, normal);
      }
    }
    for (let i = 0; i < 8; i++) {
      const a = start + (end - start) * i / 8, b = start + (end - start) * (i + 1) / 8, m = (a + b) / 2;
      const ia = polar(inner, a, top), ib = polar(inner, b, top), oa = polar(outer, a, top), ob = polar(outer, b, top);
      const ial = ia.clone().setY(bottom), ibl = ib.clone().setY(bottom), oal = oa.clone().setY(bottom), obl = ob.clone().setY(bottom);
      quad(ia, oa, ob, ib, new THREE.Vector3(0, 1, 0));
      quad(ial, ibl, obl, oal, new THREE.Vector3(0, -1, 0));
      quad(oa, oal, obl, ob, polar(1, m));
      quad(ia, ib, ibl, ial, polar(-1, m));
      if (i === 0) quad(ia, ial, oal, oa, new THREE.Vector3(-Math.cos(a), 0, -Math.sin(a)));
      if (i === 7) quad(ib, ob, obl, ibl, new THREE.Vector3(Math.cos(b), 0, Math.sin(b)));
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    put(buckets, material, geometry);
  }
  function flush(buckets, parent, name) {
    for (const [material, pieces] of buckets) {
      const geometry = mergeGeometries(pieces, false);
      if (!geometry) throw new Error('Night lamp geometry merge failed: ' + name);
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
      const mesh = new THREE.Mesh(geometry, material);
      mesh.name = name + ' / ' + material.name;
      mesh.castShadow = false;
      mesh.receiveShadow = material !== opal;
      parent.add(mesh); meshes.push(mesh);
      triangles += geometry.getAttribute('position').count / 3;
      for (const piece of pieces) piece.dispose();
    }
    buckets.clear();
  }
  function point(parent, position, peak, distance, label) {
    const light = new THREE.PointLight('#ffd0a0', 0, distance, 2);
    light.name = label;
    light.position.copy(position);
    light.castShadow = false;
    parent.add(light);
    lights.push(light);
    lightSpecs.push({ light, nightIntensity: peak, distance, label });
    return light;
  }

  // One substantial shaded lantern at each existing masonry rib. The shade
  // hangs in front of the tall bookcases, including their 4.83 m pointed crowns;
  // the high bracket clears those crowns and the next slab. R30.52+ remains
  // outside the entire moving-bridge sweep and above the walking/stair lanes.
  for (const floor of architecture.floors) {
    for (let bay = 0; bay < 12; bay++) {
      const angle = bay * TAU / 12, y = floor.y, radius = 30.80;
      radialBox(staticBuckets, housing, 32.095, angle, y + 5.075, .18, .30, .06);
      beam(staticBuckets, housing, polar(32.10, angle, y + 5.18), polar(radius, angle, y + 5.16), .031);
      beam(staticBuckets, housing, polar(32.10, angle, y + 5.02), polar(31.58, angle, y + 5.172), .025);
      beam(staticBuckets, brass, polar(radius, angle, y + 5.16), polar(radius, angle, y + 5.015), .031);
      cylinder(staticBuckets, opal, polar(radius, angle, y + 4.64), .21, .21, .48);
      cylinder(staticBuckets, brass, polar(radius, angle, y + 4.955), .08, .28, .14);
      cylinder(staticBuckets, brass, polar(radius, angle, y + 4.355), .25, .20, .09);
      for (let rod = 0; rod < 4; rod++) {
        const offset = polar(.224, angle + rod * TAU / 4);
        const from = polar(radius, angle, y + 4.39).add(offset);
        beam(staticBuckets, housing, from, from.clone().setY(y + 4.89), .017);
      }
      fixtures.gallery.push({
        level: floor.level, angle, position: polar(radius, angle, y + 4.64),
        mountingPoint: polar(32.125, angle, y + 5.075),
        bottomY: y + 4.31, topY: y + 5.225, innerRadius: 30.52,
      });
    }
  }

  // A segmented warm line below each fixed platform lip. Nothing rises into
  // the bridge landing: maximum R4.507 and top y-.212, inside the R4.6 collar.
  for (const level of architecture.central.levels) {
    for (let bay = 0; bay < 12; bay++) {
      const a = bay * TAU / 12 + .006, b = (bay + 1) * TAU / 12 - .006, y = level.y;
      band(staticBuckets, housing, 4.399, 4.495, y - .392, y - .212, a, b);
      band(staticBuckets, opal, 4.486, 4.507, y - .326, y - .255, a + .003, b - .003);
      band(staticBuckets, brass, 4.487, 4.505, y - .249, y - .233, a, b);
    }
    fixtures.fascia.push({ level: level.level, y: level.y, innerRadius: 4.399, outerRadius: 4.507, bottomY: level.y - .392, topY: level.y - .212, bays: 12 });
  }

  // Slender diffusers follow the real tapered posts, stopping between floors.
  // They stay on the outside of the four diagonal posts, away from both doors.
  const postRadius = y => .24 - .07 * (y + .5) / 42.2;
  for (const level of architecture.central.levels) {
    for (const degrees of [45, 135, 225, 315]) {
      const angle = degrees * DEG, fromY = level.y + .70, toY = level.y + 5.18;
      const backFrom = polar(2.45 + postRadius(fromY) + .033, angle, fromY);
      const backTo = polar(2.45 + postRadius(toY) + .033, angle, toY);
      beam(staticBuckets, housing, backFrom, backTo, .045);
      const glowFrom = polar(2.45 + postRadius(fromY) + .086, angle, fromY + .075);
      const glowTo = polar(2.45 + postRadius(toY) + .086, angle, toY - .075);
      beam(staticBuckets, opal, glowFrom, glowTo, .021);
      for (const p of [backFrom, backTo]) cylinder(staticBuckets, brass, p, .056, .056, .055, 8);
      fixtures.vertical.push({ level: level.level, angle, from: glowFrom, to: glowTo, postRadiusAt: postRadius });
    }
    for (const shaft of architecture.central.shafts) {
      const y = level.y;
      // The header's front face is z=-1.05. Backplate touches that face;
      // diffuser faces the lobby, above a person's head and clear of the gate.
      box(staticBuckets, housing, new THREE.Vector3(shaft.x, y + 2.33, -1.095), [1.34, .085, .09]);
      box(staticBuckets, opal, new THREE.Vector3(shaft.x, y + 2.324, -1.151), [1.22, .040, .022]);
      box(staticBuckets, brass, new THREE.Vector3(shaft.x, y + 2.377, -1.119), [1.39, .024, .12]);
      fixtures.headers.push({ level: level.level, position: new THREE.Vector3(shaft.x, y + 2.324, -1.151), bottomY: y + 2.2875 });
    }
  }
  flush(staticBuckets, group, 'Fixed lamps');

  // Add diffuse local illumination to the odd galleries; the original app
  // already lights levels 0/2/4/6. Keep colour and intensity stable under zoom.
  const galleryLevels = lowPower ? [1, 5] : [1, 3, 5];
  for (const level of galleryLevels) {
    const floor = architecture.floors.find(entry => entry.level === level);
    if (!floor) continue;
    const angle = (level - 1) / 2 * TAU / 3;
    point(group, polar(30.50, angle, floor.y + 4.22), 47, 16, 'Gallery lantern pool L' + level);
  }
  if (!lowPower && architecture.central.levels.length > 3) {
    const middle = architecture.central.levels[3];
    point(group, new THREE.Vector3(architecture.central.shafts[0].x, middle.y + 2.24, -1.24), 28, 12, 'Central lift lobby pool');
  }

  lifts.cabins.forEach((cabin, index) => {
    const moving = new THREE.Group();
    moving.name = 'Cabin ' + (index + 1) + ' ceiling lamps';
    cabin.group.add(moving); cabinGroups.push(moving);
    const buckets = new Map();
    for (const x of [-.56, .56]) {
      box(buckets, housing, new THREE.Vector3(x, 2.095, 0), [.17, .08, 1.20]);
      box(buckets, opal, new THREE.Vector3(x, 2.052, 0), [.12, .025, 1.08]);
      for (const z of [-.592, .592]) box(buckets, brass, new THREE.Vector3(x, 2.095, z), [.18, .070, .030]);
      fixtures.cabins.push({ cabinIndex: index, parent: moving, position: new THREE.Vector3(x, 2.052, 0), bottomY: 2.0395 });
    }
    flush(buckets, moving, 'Cabin ' + (index + 1) + ' lamps');
    point(moving, new THREE.Vector3(0, 1.87, 0), 7, 3.6, 'Cabin ' + (index + 1) + ' warm ceiling pool');
  });
  boxTemplate.dispose(); beamTemplate.dispose();

  const staticMeshes = meshes.filter(mesh => mesh.parent === group);
  const metrics = {
    galleryLanterns: fixtures.gallery.length,
    galleryLanternsPerFloor: 12,
    fasciaLevels: fixtures.fascia.length,
    fasciaSegments: fixtures.fascia.length * 12,
    verticalStrips: fixtures.vertical.length,
    doorHeaderLights: fixtures.headers.length,
    movingCabinLights: fixtures.cabins.length,
    dynamicLights: lights.length,
    shadowLights: 0,
    fixedDrawCalls: staticMeshes.length,
    movingDrawCalls: meshes.length - staticMeshes.length,
    drawCalls: meshes.length,
    triangles,
    materials: materials.length,
    textures: 0,
    lowPower: !!lowPower,
    maximumCoreFasciaRadius: 4.507,
    bridgeCollarRadialClearance: .093,
    galleryMinimumFixtureHeight: 4.31,
    cabinMinimumFixtureHeight: 2.0395,
  };
  let currentNight = 0, disposed = false;
  function setNight(value = 0) {
    // A malformed input retains the last valid state instead of flashing day.
    if (disposed) return;
    if (Number.isFinite(value)) currentNight = Math.min(1, Math.max(0, value));
    opal.emissiveIntensity = .025 + currentNight * 1.325;
    for (const spec of lightSpecs) spec.light.intensity = spec.nightIntensity * currentNight;
  }
  setNight(0);
  return {
    group, materials, meshes, staticMeshes, lights, fixtures, cabinGroups, lightSpecs, metrics,
    setNight,
    dispose() {
      if (disposed) return;
      disposed = true;
      group.removeFromParent();
      for (const cabin of cabinGroups) cabin.removeFromParent();
      for (const mesh of meshes) mesh.geometry.dispose();
      for (const material of materials) material.dispose();
      for (const light of lights) light.dispose();
    },
  };
}
