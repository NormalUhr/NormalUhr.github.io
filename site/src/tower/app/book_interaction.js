import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { polar } from './mechanism.js';
import { angleOf, floorY, innerRadius, segmentPoint, walkRadius } from './navigation.js';
import { bookPickupRoute } from './book_tasks.js';
import { isCameraHitVisible } from './camera_collision.js';
import { words } from './words.js';

const point = (angle, radius, y, tangent = 0) => ({
  x: Math.sin(angle) * radius + Math.cos(angle) * tangent,
  y, z: -Math.cos(angle) * radius + Math.sin(angle) * tangent,
});

/** Three reachable return trays per floor, mounted to the actual positive
 * cabinet/bookcase wings. The service routes avoid fixed stairs and islands. */
export function createBookReturnStations(interiors) {
  return (interiors?.stations || []).filter(s => [1, 3, 5].includes(s.index)).map(station => {
    const { level, angle } = station;
    const position = point(angle, 30.37, floorY(level), 3.59), a = angleOf(position);
    return { id: `book-return-${level}-${station.index}`, level, angle: a,
      position, approach: [polar(walkRadius(level), a, floorY(level)), { ...position }],
      shelfPosition: point(angle, 30.77, level * 6 + .985, 3.59),
      shelfYaw: -angle, facing: Math.PI - angle, mountingAngle: angle, tangent: 3.59,
      label: words.places.returnDesk, sourceStationId: station.id, source: 'mounted wing return tray' };
  });
}

export function createBookReturnTrays(THREE, scene, { stations = [], clippingPlanes = [] } = {}) {
  const group = new THREE.Group(); group.name = 'Small library return trays'; scene.add(group);
  const materials = [
    new THREE.MeshStandardMaterial({ color: '#77533a', roughness: .90, metalness: 0, clippingPlanes, clipShadows: true }),
    new THREE.MeshStandardMaterial({ color: '#45362d', roughness: .94, metalness: 0, clippingPlanes, clipShadows: true }),
    new THREE.MeshStandardMaterial({ color: '#b39461', roughness: .87, metalness: .10, clippingPlanes, clipShadows: true }),
  ];
  materials.forEach((m, i) => { m.name = `Book returns / ${['oak trays', 'walnut brackets', 'aged brass edges'][i]}`; });
  const buckets = materials.map(() => []), meshes = [], mountingSamples = [];
  function box(material, station, radius, height, tangent, width, h, depth, tilt = 0) {
    const p = point(station.mountingAngle, radius, station.level * 6 + height, tangent);
    const g = new THREE.BoxGeometry(width, h, depth).toNonIndexed();
    if (tilt) g.rotateX(tilt);
    g.rotateY(-station.mountingAngle).translate(p.x, p.y, p.z); buckets[material].push(g);
  }
  for (const s of stations) {
    const t = s.tangent;
    box(0, s, 30.77, .95, t, .74, .07, .66);
    for (const sign of [-1, 1]) box(1, s, 30.79, 1.015, t + sign * .385, .045, .10, .70);
    box(1, s, 31.105, 1.015, t, .80, .10, .04);
    box(2, s, 30.43, .96, t, .28, .045, .018);
    // Back plates penetrate the old cabinet/bookcase face. Angled brackets
    // physically join them to the underside of the projecting tray.
    for (const sign of [-1, 1]) {
      box(1, s, 31.10, .72, t + sign * .26, .07, .52, .12);
      box(1, s, 30.885, .715, t + sign * .26, .062, .56, .062, -.79);
      mountingSamples.push(point(s.mountingAngle, 31.10, s.level * 6 + .68, t + sign * .26));
    }
  }
  let triangles = 0;
  for (let i = 0; i < buckets.length; i++) {
    if (!buckets[i].length) continue;
    const geometry = mergeGeometries(buckets[i], false);
    geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, materials[i]);
    mesh.name = materials[i].name; mesh.castShadow = true; mesh.receiveShadow = true;
    group.add(mesh); meshes.push(mesh); triangles += geometry.attributes.position.count / 3;
    for (const g of buckets[i]) g.dispose();
  }
  let disposed = false;
  return { group, materials, meshes, stations, mountingSamples, collisionRoots: [group],
    metrics: { trays: stations.length, drawCalls: meshes.length, triangles, textures: 0, extraLights: 0 },
    dispose() { if (disposed) return; disposed = true; group.removeFromParent(); for (const mesh of meshes) mesh.geometry.dispose(); for (const m of materials) m.dispose(); } };
}

/** Hit the visible scene first; only then validate the actual gallery and its
 * pickup/return spurs. No invisible picking plane and no off-click relocation. */
export function createBookInteraction(THREE, { camera, element, collision, scene, architecture, interiors, blockers = [] } = {}) {
  const ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), normal = new THREE.Vector3();
  const transform = new THREE.Matrix4(), instance = new THREE.Matrix4();
  const origin = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), down = new THREE.Vector3(0, -1, 0);
  const physicalMaterials = new Set();
  (scene || architecture?.group)?.traverse(object => {
    if (object.material) for (const m of Array.isArray(object.material) ? object.material : [object.material]) physicalMaterials.add(m);
  });
  if (!scene) interiors?.group?.traverse(object => { if (object.material) physicalMaterials.add(object.material); });
  const materialSnapshots = [];
  const stats = { attempts: 0, accepted: 0, rejected: {}, clearanceQueries: 0 };
  function fail(reason) { stats.rejected[reason] = (stats.rejected[reason] || 0) + 1; return { ok: false, reason }; }
  function normalY(hit) {
    const geometry = hit.object.geometry, positions = geometry?.attributes.position, index = geometry?.index;
    const offset = hit.faceIndex * 3;
    if (!positions || !Number.isInteger(offset) || offset < 0) return -1;
    transform.copy(hit.object.matrixWorld);
    if (hit.instanceId !== undefined) { hit.object.getMatrixAt(hit.instanceId, instance); transform.multiply(instance); }
    for (const [vector, i] of [[a, 0], [b, 1], [c, 2]]) vector.fromBufferAttribute(positions, index ? index.getX(offset + i) : offset + i).applyMatrix4(transform);
    return normal.subVectors(b, a).cross(c.sub(a)).normalize().y;
  }
  function isFloor(hit, level) {
    return !!hit && normalY(hit) > .985 && Math.abs(hit.point.y - level * 6) <= .045 &&
      (isDescendant(hit.object, architecture?.group) || isDescendant(hit.object, interiors?.group));
  }
  function isDescendant(object, parent) { for (let p = object; p; p = p.parent) if (p === parent) return true; return false; }
  function query(p, direction, far) { stats.clearanceQueries++; return collision.nearest(p, direction, 0, far); }
  function supported(p) {
    origin.set(p.x, floorY(p.level) + .30, p.z);
    return isFloor(query(origin, down, .39), p.level);
  }
  function emptyColumn(p, radius, height) {
    for (let i = 0; i < 9; i++) {
      const theta = (i - 1) * Math.PI / 4, r = i === 0 ? 0 : radius;
      const q = { x: p.x + Math.cos(theta) * r, z: p.z + Math.sin(theta) * r, level: p.level };
      if (!supported(q)) return false;
      origin.set(q.x, floorY(p.level) + .085, q.z);
      if (query(origin, up, height - .085)) return false;
    }
    return true;
  }
  function clearSegment(segment, level) {
    const steps = Math.max(1, Math.ceil(segment.length / .28));
    for (let i = 0; i <= steps; i++) if (!emptyColumn({ ...segmentPoint(segment, segment.length * i / steps), level }, .30, 1.90)) return false;
    return true;
  }
  function withPhysicalSurfaces(fn) {
    // Cutaway removes pixels, not furniture or walking floors. The synchronous
    // safety query includes those real surfaces, then restores the SAME arrays
    // before the browser can render. The initial visible hit is never bypassed.
    materialSnapshots.length = 0;
    for (const m of physicalMaterials) if (m.clippingPlanes?.length) { materialSnapshots.push([m, m.clippingPlanes]); m.clippingPlanes = []; }
    try { return fn(); } finally { for (const [m, planes] of materialSnapshots) m.clippingPlanes = planes; materialSnapshots.length = 0; }
  }
  function validate(position, level) {
    if (!Number.isInteger(level) || level < 0 || level > 6 || ![position?.x, position?.y, position?.z].every(Number.isFinite)) return 'not-gallery';
    const r = Math.hypot(position.x, position.z), angle = angleOf(position);
    if (r < innerRadius(level) + .80 || r > 29.20) return 'gallery-space';
    if (!emptyColumn({ ...position, level }, .55, 2.05)) return 'blocked';
    for (const sign of [-1, 1]) {
      const actor = { level, age: 'adult', position: polar(walkRadius(level), angle + sign * .14, floorY(level)) };
      const pickup = bookPickupRoute(actor, position);
      // The unbroken ring is existing circulation. Validate the actual two new
      // spur segments on both sides, including the final fixed pickup stance.
      // The .30m envelope also covers the child's route .09m further inward;
      // both retain a .20m body radius without duplicating all scene queries.
      for (const segment of pickup.path.slice(-2)) if (!clearSegment(segment, level)) return 'approach-blocked';
      const home = polar(walkRadius(level), angle, floorY(level));
      const steps = Math.max(1, Math.ceil(Math.abs(Math.hypot(pickup.stance.x, pickup.stance.z) - walkRadius(level)) / .28));
      for (let i = 0; i <= steps; i++) {
        const u = i / steps, p = { x: pickup.stance.x + (home.x - pickup.stance.x) * u,
          z: pickup.stance.z + (home.z - pickup.stance.z) * u, level };
        if (!emptyColumn(p, .30, 1.90)) return 'approach-blocked';
      }
    }
    return null;
  }
  function pickRay(rayOrigin, direction, far = 900) {
    stats.attempts++;
    const hit = collision.nearest(rayOrigin, direction, .015, far);
    if (!hit) return fail('not-gallery');
    const level = Math.round(hit.point.y / 6);
    if (!isFloor(hit, level)) return fail('not-gallery');
    if (blockers.length) {
      ray.set(rayOrigin, direction.clone().normalize()); ray.near = .015; ray.far = hit.distance;
      if (ray.intersectObjects(blockers, false).some(isCameraHitVisible)) return fail('occupied');
    }
    const position = { x: hit.point.x, y: floorY(level), z: hit.point.z };
    const reason = withPhysicalSurfaces(() => validate(position, level));
    if (reason) return fail(reason);
    stats.accepted++; return { ok: true, level, position };
  }
  function pick(x, y) {
    const rect = element?.getBoundingClientRect(), activeCamera = typeof camera === 'function' ? camera() : camera;
    if (!rect || !activeCamera || !(rect.width > 0 && rect.height > 0) || !Number.isFinite(x + y) ||
      x < rect.left || x > rect.left + rect.width || y < rect.top || y > rect.top + rect.height) return fail('not-gallery');
    pointer.set((x - rect.left) / rect.width * 2 - 1, -(y - rect.top) / rect.height * 2 + 1);
    activeCamera.updateMatrixWorld(); ray.setFromCamera(pointer, activeCamera);
    return pickRay(ray.ray.origin, ray.ray.direction, activeCamera.far);
  }
  return { pick, pickRay, stats, validatePosition(position, level) { return withPhysicalSurfaces(() => validate(position, level)); } };
}
