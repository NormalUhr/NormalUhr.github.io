import * as THREE from 'three';

// This forest is a fixed part of the scenery, not a camera-facing panorama or
// a succession of distance-switched LODs. Nothing in it fades with the camera.
const TAU = Math.PI * 2;
const UP = new THREE.Vector3(0, 1, 0);
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const mix = (a, b, t) => a + (b - a) * t;
function randomSource(seed) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function noiseSource(seed) {
  const hash = (x, z) => {
    let h = Math.imul(x, 374761393) ^ Math.imul(z, 668265263) ^ seed;
    h = Math.imul(h ^ h >>> 13, 1274126177);
    return ((h ^ h >>> 16) >>> 0) / 4294967295;
  };
  return (x, z) => {
    const ix = Math.floor(x), iz = Math.floor(z), fx = x - ix, fz = z - iz;
    const u = fx * fx * (3 - 2 * fx), v = fz * fz * (3 - 2 * fz);
    return mix(mix(hash(ix, iz), hash(ix + 1, iz), u),
      mix(hash(ix, iz + 1), hash(ix + 1, iz + 1), u), v);
  };
}

// Four triangles make a folded leaf rather than a rectangular alpha card. The
// clusters have volume from every direction; random folds and vertex colors
// supply small highlights without thousands of separate draw calls.
function crownGeometry(seed, count, conifer = false, sizeBoost = 1) {
  const rnd = randomSource(seed), positions = [], normals = [], uvs = [], colors = [], indices = [];
  const rotation = new THREE.Quaternion(), axis = new THREE.Vector3(), point = new THREE.Vector3();
  const normal = new THREE.Vector3(), center = new THREE.Vector3();
  const euler = new THREE.Euler();
  const lobes = Array.from({length: 7}, (_, i) => {
    const a = i * 2.399963 + rnd() * .5;
    return new THREE.Vector3(Math.sin(a) * (.30 + rnd() * .34), (rnd() - .5) * .65,
      Math.cos(a) * (.30 + rnd() * .34));
  });
  for (let leaf = 0; leaf < count; leaf++) {
    const a = rnd() * TAU, v = rnd() * 2 - 1, rr = Math.sqrt(1 - v * v);
    if (conifer) {
      const tier = Math.floor(rnd() * 7), level = (tier + rnd() * .46) / 7;
      const spread = (.96 - level * .76) * (.46 + Math.sqrt(rnd()) * .56);
      center.set(Math.sin(a) * spread, -.93 + level * 1.88, Math.cos(a) * spread);
      rotation.setFromEuler(euler.set(-.10 + rnd() * .65, -a, (rnd() - .5) * .45));
    } else {
      const lobe = lobes[leaf % lobes.length], radius = .33 + Math.cbrt(rnd()) * .31;
      center.set(lobe.x + Math.sin(a) * rr * radius, lobe.y + v * radius * .85,
        lobe.z + Math.cos(a) * rr * radius);
      rotation.setFromEuler(euler.set((rnd() - .5) * 1.8, rnd() * TAU, (rnd() - .5) * 1.35));
    }
    const length = (conifer ? .33 + rnd() * .27 : .30 + rnd() * .25) * sizeBoost;
    const width = length * (conifer ? .40 + rnd() * .18 : .49 + rnd() * .20);
    const fold = length * (.07 + rnd() * .13), start = positions.length / 3;
    const shade = .63 + rnd() * .32 + Math.max(0, center.y) * .10;
    for (const [x, y, z, u, uv] of [
      [0, 0, -length * .45, .5, 0], [-width * .5, 0, 0, 0, .45],
      [0, 0, length * .55, .5, 1], [width * .5, 0, 0, 1, .45], [0, fold, 0, .5, .45]
    ]) {
      point.set(x, y, z).applyQuaternion(rotation).add(center); positions.push(point.x, point.y, point.z);
      axis.set(x * -.28, 1, z * -.18).normalize(); normal.copy(axis).applyQuaternion(rotation);
      normals.push(normal.x, normal.y, normal.z); uvs.push(u, uv);
      colors.push(shade * (conifer ? .89 : .93), shade, shade * (conifer ? .91 : .79));
    }
    indices.push(start, start + 1, start + 4, start + 1, start + 2, start + 4,
      start + 2, start + 3, start + 4, start + 3, start, start + 4);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices); geometry.computeBoundingSphere();
  return geometry;
}

function trunkGeometry() {
  const geometry = new THREE.CylinderGeometry(.34, 1, 1, 8, 5, true);
  geometry.translate(0, .5, 0);
  const positions = geometry.attributes.position;
  for (let i = 0; i < positions.count; i++) {
    const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i), a = Math.atan2(z, x);
    const flare = 1 + .23 * Math.exp(-y * 13), ridge = 1 + Math.sin(a * 5 + y * 2.8) * .044;
    positions.setXYZ(i, x * flare * ridge + Math.sin(y * 2.1) * .07,
      y, z * flare * ridge + y * y * .10);
  }
  geometry.computeVertexNormals(); return geometry;
}

/**
 * Render-only woodland surrounding the original collidable clearing.
 * makeMaterial shares habitat weather uniforms; heightAt is the actual outer
 * scenery surface. All geometry, instances and tree locations are persistent.
 */
export function createForestBackdrop(parent, {
  seed, quality = 'high', heightAt, makeMaterial, textures, startView
}) {
  const group = new THREE.Group(); group.name = 'Layered 360 degree woodland'; parent.add(group);
  group.userData.nonInteractive = true;
  const low = quality === 'low', rnd = randomSource(seed), noise = noiseSource(seed ^ 0x45b192);
  const wood = makeMaterial({
    map: textures.bark, normalMap: textures.barkNormal, normalScale: new THREE.Vector2(.65, .65),
    roughnessMap: textures.barkARM, roughness: .96, color: '#bcbbaa'
  }, {snow: .22, wet: .8});
  const broadleaf = makeMaterial({
    map: textures.leaf, color: '#718746', vertexColors: true, roughness: .93, side: THREE.DoubleSide
  }, {wind: .023, snow: .7, wet: .7});
  const evergreen = makeMaterial({
    map: textures.leaf, color: '#4f6a45', vertexColors: true, roughness: .96, side: THREE.DoubleSide
  }, {wind: .013, snow: .85, wet: .65});
  const transform = new THREE.Object3D(), segment = new THREE.Vector3(), meshes = [];
  const statistics = {trees: 0, startViewTrees: 0, trunks: 0, branches: 0, canopyClusters: 0, shrubs: 0, triangles: 0, drawCalls: 0, layers: []};
  // The start view looks over the clearing from among the trees at its edge. A
  // tree with a crown on that line of sight is not planted, so the view is open
  // and no part of the forest has to fade for it.
  const local = new THREE.Matrix4(), near = new THREE.Vector3(), far = new THREE.Vector3();
  function onSightLine(crown, geometry) {
    transform.position.set(crown.x, crown.y, crown.z);
    transform.rotation.set(crown.rx || 0, crown.angle || 0, crown.rz || 0);
    transform.scale.set(crown.sx, crown.sy, crown.sz); transform.updateMatrix();
    local.copy(transform.matrix).invert();
    near.copy(startView.from).applyMatrix4(local); far.copy(startView.to).applyMatrix4(local).sub(near);
    const {center, radius} = geometry.boundingSphere;
    const t = clamp((center.x - near.x) * far.x + (center.y - near.y) * far.y + (center.z - near.z) * far.z, 0, far.lengthSq()) / far.lengthSq();
    return near.addScaledVector(far, t).distanceTo(center) < radius;
  }
  function instances(geometry, material, entries, name) {
    const mesh = new THREE.InstancedMesh(geometry, material, entries.length);
    mesh.name = name; mesh.castShadow = false; mesh.receiveShadow = false;
    mesh.userData.nonInteractive = true;
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i]; transform.position.set(entry.x, entry.y, entry.z);
      if (entry.end) {
        segment.copy(entry.end).sub(transform.position);
        transform.quaternion.setFromUnitVectors(UP, segment.clone().normalize());
        transform.scale.set(entry.r, segment.length(), entry.r);
      } else {
        transform.rotation.set(entry.rx || 0, entry.angle || 0, entry.rz || 0);
        transform.scale.set(entry.sx, entry.sy, entry.sz);
      }
      transform.updateMatrix(); mesh.setMatrixAt(i, transform.matrix);
      if (entry.color) mesh.setColorAt(i, entry.color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere(); mesh.boundingSphere.radius += 1;
    group.add(mesh); meshes.push(mesh);
    statistics.triangles += geometry.index.count / 3 * entries.length;
    statistics.drawCalls++; return mesh;
  }
  const occupied = [], trunkEntries = [], branchEntries = [], shrubs = [];
  const layers = [
    {name: 'Clearing edge', count: low ? 29 : 46, inner: 1.04, outer: 1.95, spacing: 3.25,
      height: [10.5, 17.5], crown: [2.55, 4.0], leaves: low ? 128 : 180},
    {name: 'Middle woodland', count: low ? 83 : 134, inner: 1.62, outer: 3.45, spacing: 4.15,
      height: [14, 24], crown: [3.4, 5.1], leaves: low ? 61 : 90},
    {name: 'Distant forest', count: low ? 120 : 182, inner: 2.98, outer: 5.05, spacing: 5.5,
      height: [18, 30], crown: [4.1, 6.25], leaves: low ? 37 : 55}
  ];
  for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
    const layer = layers[layerIndex], trees = [], leaves = [], needles = [];
    for (let tries = 0; trees.length < layer.count && tries < layer.count * 350; tries++) {
      // Cartesian rejection sampling plus broad density patches makes groves
      // and openings. No regular angular spacing or aligned concentric rows.
      const x = (rnd() * 2 - 1) * 22 * (layer.outer + .18);
      const z = (rnd() * 2 - 1) * 18 * (layer.outer + .18);
      const radius = Math.hypot(x / 22, z / 18), angle = Math.atan2(z / 18, x / 22);
      const irregularity = .12 * Math.sin(angle * 3 + .71) + (noise(x * .04 + 19, z * .04 + 29) - .5) * .32;
      if (radius < layer.inner + irregularity || radius > layer.outer + irregularity) continue;
      if (Math.abs(x) < 19.8 && Math.abs(z) < 15.8) continue;
      if (rnd() > .38 + noise(x * .073 + 4, z * .073 + 9) * .62) continue;
      const spacing = layer.spacing * (.79 + rnd() * .33);
      if (occupied.some(p => (p.x - x) ** 2 + (p.z - z) ** 2 < Math.min(p.spacing, spacing) ** 2)) continue;
      const tree = {x, z, spacing}; occupied.push(tree); trees.push(tree);
    }
    const boost = low ? 1.12 : 1;
    const broadleafCrown = crownGeometry(seed + 917 + layerIndex * 37, layer.leaves, false, boost);
    const evergreenCrown = crownGeometry(seed + 1881 + layerIndex * 41, Math.round(layer.leaves * 1.1), true, boost);
    let leftOut = 0;
    for (const tree of trees) {
      const {x, z} = tree, y = heightAt(x, z), typeChoice = rnd();
      const conifer = typeChoice > .81, slender = !conifer && typeChoice > .60;
      const height = mix(...layer.height, rnd()) * (slender ? 1.08 : 1);
      const radius = (.24 + rnd() * .31) * (height / 17) * (conifer ? .78 : 1);
      const crown = mix(...layer.crown, rnd()) * (slender ? .68 : 1);
      const leanX = (rnd() - .5) * .8, leanZ = (rnd() - .5) * .8;
      const tint = .65 + rnd() * .30;
      const trunk = {x, y: y - .12, z, sx: radius, sy: height, sz: radius,
        rx: leanZ / height, rz: -leanX / height, angle: rnd() * TAU,
        color: new THREE.Color().setRGB(tint * .98, tint, tint * .91)};
      const crowns = [], branches = [];
      if (conifer) {
        crowns.push({x: x + leanX, y: y + height * .72, z: z + leanZ,
          sx: crown, sy: height * .43, sz: crown * (.83 + rnd() * .25), angle: rnd() * TAU,
          color: new THREE.Color().setRGB(.82 + rnd() * .17, .87 + rnd() * .13, .78 + rnd() * .19)});
      } else {
        const count = layerIndex === 0 ? 3 : (rnd() < .57 ? 2 : 3);
        for (let branch = 0; branch < count; branch++) {
          const angle = branch * 2.399963 + rnd() * .75;
          const spread = crown * (.39 + rnd() * .32), crownY = y + height * (.76 + rnd() * .16);
          const crownX = x + leanX + Math.sin(angle) * spread;
          const crownZ = z + leanZ + Math.cos(angle) * spread;
          const reach = crown * (.66 + rnd() * .27);
          crowns.push({x: crownX, y: crownY, z: crownZ,
            sx: reach, sy: reach * (slender ? 1.17 : .74 + rnd() * .18), sz: reach * (.86 + rnd() * .23),
            angle: rnd() * TAU, rx: (rnd() - .5) * .22, rz: (rnd() - .5) * .22,
            color: new THREE.Color().setRGB(.72 + rnd() * .26, .82 + rnd() * .18, .65 + rnd() * .30)});
          if (layerIndex < 2) {
            branches.push({x: x + leanX * .5, y: y + height * (.49 + branch * .065), z: z + leanZ * .5,
              end: new THREE.Vector3(crownX, crownY, crownZ), r: radius * (.20 + rnd() * .13)});
          }
        }
      }
      // A tree left out still draws its random numbers, so every other tree stays where it was.
      if (crowns.some(c => onSightLine(c, conifer ? evergreenCrown : broadleafCrown))) leftOut++;
      else {
        trunkEntries.push(trunk); (conifer ? needles : leaves).push(...crowns); branchEntries.push(...branches);
      }
      // Irregular low vegetation fills the gap between grass and high crowns.
      // It is outside the original terrain rectangle and never affects paths.
      if (layerIndex === 0 || (layerIndex === 1 && rnd() < .24)) {
        for (let bush = 0; bush < (layerIndex === 0 ? 2 : 1); bush++) {
          const bx = x + (rnd() - .5) * 4, bz = z + (rnd() - .5) * 4;
          if (Math.abs(bx) < 19 && Math.abs(bz) < 15) continue;
          const sapling = rnd() < .40, by = heightAt(bx, bz);
          const r = sapling ? 1.2 + rnd() * .95 : .68 + rnd() * .88;
          const foliageY = by + r * (sapling ? 1.27 : .60);
          shrubs.push({x: bx, y: foliageY, z: bz,
            sx: r, sy: r * (sapling ? 1.15 : .65), sz: r * (.8 + rnd() * .4), angle: rnd() * TAU,
            color: new THREE.Color().setRGB(.74 + rnd() * .22, .88 + rnd() * .12, .66 + rnd() * .25)});
          if (sapling) trunkEntries.push({x: bx, y: by - .03, z: bz,
            sx: .07 + rnd() * .06, sy: foliageY - by + r * .3, sz: .09 + rnd() * .04,
            angle: rnd() * TAU, color: new THREE.Color('#b5b7a1')});
        }
      }
    }
    instances(broadleafCrown, broadleaf, leaves, `${layer.name}: overlapping broadleaf crowns`);
    instances(evergreenCrown, evergreen, needles, `${layer.name}: evergreen spires`);
    statistics.trees += trees.length - leftOut; statistics.startViewTrees += leftOut;
    statistics.canopyClusters += leaves.length + needles.length;
    statistics.layers.push({name: layer.name, trees: trees.length - leftOut, canopyClusters: leaves.length + needles.length});
  }
  instances(trunkGeometry(), wood, trunkEntries, 'Forest trunks with varied height and lean');
  const branchGeometry = new THREE.CylinderGeometry(.16, 1, 1, 5, 2, true); branchGeometry.translate(0, .5, 0);
  instances(branchGeometry, wood, branchEntries, 'Visible branching structure beneath the crowns');
  instances(crownGeometry(seed + 4481, low ? 61 : 90, false, low ? 1.1 : 1), broadleaf, shrubs,
    'Patchy understorey beyond the clearing');
  statistics.trunks = trunkEntries.length; statistics.branches = branchEntries.length; statistics.shrubs = shrubs.length;
  const broadPalette = ['#87a25a', '#718746', '#b18443', '#857651'].map(c => new THREE.Color(c));
  const evergreenPalette = ['#607d4d', '#4f6a45', '#5c714b', '#627263'].map(c => new THREE.Color(c));
  function update(season = 1) {
    const s = ((season % 4) + 4) % 4, i = Math.floor(s), t = s - i;
    broadleaf.color.copy(broadPalette[i]).lerp(broadPalette[(i + 1) % 4], t);
    evergreen.color.copy(evergreenPalette[i]).lerp(evergreenPalette[(i + 1) % 4], t);
  }
  update();
  return {
    group, meshes, statistics, update, materials: {wood, broadleaf, evergreen},
    extents: {minX: -116, maxX: 116, minZ: -96, maxZ: 96}
  };
}
