import * as THREE from 'three';
import { TAU } from './math.js';
import { rockGeometry } from './habitat_shapes.js';

// These are the same twelve seeded rocks that used to be baked into the garden
// mesh. Each has one enduring mesh: a move only changes its transform. Geometry,
// material, scale and the original ring positions stay stable across old saves.
export function createHourStoneView(root, material, random) {
  const group = new THREE.Group();
  group.name = '时石 · 从木台到庭中';
  root.add(group);
  const slots = [], stock = [], meshes = new Map();
  const transformed = new THREE.Vector3();
  const transform = new THREE.Matrix4();
  for (let i = 0; i < 12; i++) {
    const id = i || 12, angle = i * TAU / 12, north = i === 0;
    const geometry = rockGeometry(81019 + i * 97, 2);
    const scale = [north ? .127 : .118, north ? .172 : .086, north ? .125 : .105];
    // Keep these two random draws, and their order, identical to 1.3. The same
    // generator is also used later by the fixed stones around the sand frame.
    const rotation = [.05 * (random() - .5), angle + .2, .04 * (random() - .5)];
    const slot = { id, x: Math.sin(angle) * 3.25, y: north ? .072 : .031, z: -Math.cos(angle) * 3.25, rotation, scale, r: north ? .17 : .14 };
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `时石-${id}`;
    mesh.userData.hourStoneId = id;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.set(slot.x, slot.y, slot.z);
    mesh.rotation.set(...rotation);
    mesh.scale.set(...scale);
    group.add(mesh);
    meshes.set(id, mesh);
    transform.compose(new THREE.Vector3(), mesh.quaternion, mesh.scale);
    let minimumY = Infinity, maximumY = -Infinity;
    const p = geometry.attributes.position;
    for (let j = 0; j < p.count; j++) {
      transformed.fromBufferAttribute(p, j).applyMatrix4(transform);
      minimumY = Math.min(minimumY, transformed.y);
      maximumY = Math.max(maximumY, transformed.y);
    }
    slot.bottomOffset = minimumY;
    slot.topOffset = maximumY;
    slots.push(slot);
    stock.push({ ...slot, x: -.8 + (id - 1) * .30, y: .28 - minimumY, z: -4.48, rotation: [...rotation], scale: [...scale], supportY: .28 });
  }
  slots.sort((a, b) => a.id - b.id);
  stock.sort((a, b) => a.id - b.id);

  function set(records) {
    if (!Array.isArray(records)) return;
    for (const state of records) {
      const mesh = meshes.get(state?.id);
      if (!mesh || ![state.x, state.y, state.z].every(Number.isFinite)) continue;
      mesh.position.set(state.x, state.y, state.z);
      if (Array.isArray(state.rotation) && state.rotation.length === 3 && state.rotation.every(Number.isFinite)) mesh.rotation.set(...state.rotation);
      else if ([state.rx, state.ry, state.rz].every(Number.isFinite)) mesh.rotation.set(state.rx, state.ry, state.rz);
    }
  }
  return { group, slots, stock, meshes, set };
}
