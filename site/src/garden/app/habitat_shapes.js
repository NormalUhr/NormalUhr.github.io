import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import {rng, fbm, TAU} from './math.js';

export const UP = new THREE.Vector3(0, 1, 0);
const obj = new THREE.Object3D(), point = new THREE.Vector3(), end = new THREE.Vector3();

export function rockGeometry(seed, detail = 2) {
  const g = new THREE.IcosahedronGeometry(1, detail), p = g.attributes.position;
  const colors = [], r = rng(seed), ox = r() * 21, oz = r() * 21;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const n = fbm(x * 2.9 + ox, z * 2.6 + y * .7 + oz);
    const shape = .82 + n * .3 + Math.sin(x * 7.1 + y * 5.3 + seed) * .038;
    const yy = Math.max(-.64, y * shape + Math.sin(z * 6 + seed) * .035);
    p.setXYZ(i, x * shape + .055 * y, yy, z * shape);
    const shade = .71 + fbm(x * 5 + ox, z * 5 + y * 2 + oz) * .28;
    const moss = Math.max(0, y - .45) * Math.max(0, n - .48) * .8;
    colors.push(shade * (1 - moss * .35), shade * (1 - moss * .12), shade * (1 - moss * .5));
  }
  g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  g.computeVertexNormals(); return g;
}

export function mossGeometry(seed, radius = 1) {
  const g = new THREE.SphereGeometry(radius, 24, 10, 0, TAU, 0, Math.PI / 2), p = g.attributes.position, colors = [];
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const n = fbm(x * 4 + seed, z * 4 - seed);
    const r = 1 + (n - .5) * .23;
    p.setXYZ(i, x * r, y * .10 * (.72 + n * .55), z * r);
    const s = .7 + n * .45; colors.push(s * .96, s, s * .85);
  }
  g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); g.computeVertexNormals(); return g;
}

export function createBatch(root) {
  const batches = new Map(), geoms = new Set();
  function add(g, mat, position = [0, 0, 0], scale = [1, 1, 1], rotation = [0, 0, 0]) {
    obj.position.set(...position); obj.scale.set(...scale); obj.rotation.set(...rotation); obj.updateMatrix();
    const c = g.clone(); c.applyMatrix4(obj.matrix);
    if (!batches.has(mat)) batches.set(mat, []);
    batches.get(mat).push(c); geoms.add(g); return c;
  }
  const box = new THREE.BoxGeometry(1, 1, 1);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 10, 1, false);
  function cube(mat, x, y, z, sx, sy, sz, ry = 0) {add(box, mat, [x, y, z], [sx, sy, sz], [0, ry, 0]);}
  function beam(mat, a, b, width, depth = width) {
    point.set(...a); end.set(...b); const length = point.distanceTo(end);
    obj.position.copy(point).add(end).multiplyScalar(.5); obj.quaternion.setFromUnitVectors(UP, end.sub(point).normalize()); obj.scale.set(width, length, depth); obj.updateMatrix();
    const c = box.clone().applyMatrix4(obj.matrix); if (!batches.has(mat)) batches.set(mat, []); batches.get(mat).push(c); geoms.add(box);
  }
  function pole(mat, a, b, radius) {
    point.set(...a); end.set(...b); const length = point.distanceTo(end);
    obj.position.copy(point).add(end).multiplyScalar(.5); obj.quaternion.setFromUnitVectors(UP, end.sub(point).normalize()); obj.scale.set(radius, length, radius); obj.updateMatrix();
    const c = cylinder.clone().applyMatrix4(obj.matrix); if (!batches.has(mat)) batches.set(mat, []); batches.get(mat).push(c); geoms.add(cylinder);
  }
  function finish() {
    const meshes = [];
    for (const [mat, list] of batches) {
      const color = list.some(g => g.hasAttribute('color'));
      // Unified attributes permit natural stones, lathe shapes and box details
      // to share one material and one draw call.
      const nonIndexed = list.map(g => {
        const n = g.index ? g.toNonIndexed() : g;
        if (color && !n.hasAttribute('color')) {
          const colors = new Float32Array(n.attributes.position.count * 3); colors.fill(1);
          n.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        }
        if (!n.hasAttribute('uv')) n.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(n.attributes.position.count * 2), 2));
        return n;
      });
      const combined = mergeGeometries(nonIndexed, false);
      if (!combined) throw new Error('Garden geometry batching failed: ' + mat.name);
      combined.computeBoundingSphere();
      const mesh = new THREE.Mesh(combined, mat); mesh.name = '庭院 · ' + (mat.name || mat.uuid.slice(0, 5));
      mesh.castShadow = true; mesh.receiveShadow = true; root.add(mesh); meshes.push(mesh);
      for (const g of new Set([...list, ...nonIndexed])) g.dispose();
    }
    geoms.forEach(g => g.dispose()); box.dispose(); cylinder.dispose(); return meshes;
  }
  return {add, cube, beam, pole, finish};
}
