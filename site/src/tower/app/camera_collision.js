/** Material/ancestor visibility used by both accelerated and native reference hits. */
export function isCameraHitVisible(hit) {
  if (!hit?.object) return false;
  for (let object = hit.object; object; object = object.parent) if (!object.visible) return false;
  const material = hit.material || (Array.isArray(hit.object.material)
    ? hit.object.material[hit.face?.materialIndex ?? 0] : hit.object.material);
  return visibleMaterialAt(material, hit.point.x, hit.point.y, hit.point.z);
}

function visibleMaterialAt(material, x, y, z) {
  if (!material || material.visible === false) return false;
  const planes = material.clippingPlanes;
  if (!planes?.length) return true;
  if (material.clipIntersection) {
    for (const p of planes) if (p.normal.x * x + p.normal.y * y + p.normal.z * z + p.constant >= 0) return true;
    return false;
  }
  for (const p of planes) if (p.normal.x * x + p.normal.y * y + p.normal.z * z + p.constant < 0) return false;
  return true;
}

/**
 * Exact triangles in a sparse uniform grid. A 3 m cell keeps short camera rays
 * local, even though architecture is merged into a few very large draw meshes.
 *
 * Static matrices/topology are captured once; call rebuild() after changing them.
 * Material clipping/side/visibility remains live. Every InstancedMesh is treated
 * dynamically, including retracting gallery balustrades in a static root.
 * Dynamic transforms are refreshed per query and use native triangle raycasts.
 */
export function createCameraCollision(THREE, { staticRoots = [], dynamicRoots = [], cellSize = 3 } = {}) {
  cellSize = Math.max(.5, cellSize);
  const roots = list => list.filter(Boolean).map(root => root.group || root);
  staticRoots = roots(staticRoots); dynamicRoots = roots(dynamicRoots);
  const raycaster = new THREE.Raycaster(), proxy = new THREE.Mesh();
  const pointA = new THREE.Vector3(), pointB = new THREE.Vector3(), pointC = new THREE.Vector3();
  const instanceMatrix = new THREE.Matrix4();
  const dynamicHits = [], visibleCache = new Map();
  let vertices = new Float64Array(0), triangleBatch = new Uint32Array(0), faceIndex = new Uint32Array(0), stamps = new Uint32Array(0);
  let cells = new Map(), batches = [], dynamicMeshes = [], serial = 0, disposed = false;
  let minX = 0, minY = 0, minZ = 0, maxX = 0, maxY = 0, maxZ = 0, nx = 0, ny = 0, nz = 0;
  const stats = { buildMilliseconds: 0, triangles: 0, staticMeshes: 0, dynamicMeshes: 0, dynamicInstances: 0,
    occupiedCells: 0, cellReferences: 0, typedArrayBytes: 0, estimatedMemoryMiB: 0, queries: 0,
    lastQuery: { cells: 0, triangles: 0, dynamicCandidates: 0 } };
  const now = () => globalThis.performance?.now() ?? Date.now();

  function objectVisible(object) {
    if (visibleCache.has(object)) return visibleCache.get(object);
    let result = (object.layers.mask & raycaster.layers.mask) !== 0;
    for (let p = object; result && p; p = p.parent) if (p.visible === false) result = false;
    visibleCache.set(object, result); return result;
  }
  function materialFor(batch) {
    return Array.isArray(batch.object.material) ? batch.object.material[batch.materialIndex] : batch.object.material;
  }
  function rebuild() {
    const began = now(), seen = new Set(), descriptors = [], staticObjects = new Set();
    dynamicMeshes = []; batches = []; cells = new Map(); disposed = false;
    const dynamicSet = new Set();
    for (const root of dynamicRoots) { root.updateWorldMatrix(true, true); root.traverse(object => dynamicSet.add(object)); }
    const addDynamic = object => {
      if (seen.has(object) || !object.isMesh) return;
      seen.add(object); dynamicMeshes.push(object);
      if (!object.geometry.boundingSphere) object.geometry.computeBoundingSphere();
      if (!object.geometry.boundingBox) object.geometry.computeBoundingBox();
    };
    let triangleCount = 0;
    for (const root of staticRoots) {
      root.updateWorldMatrix(true, true);
      root.traverse(object => {
        if (seen.has(object) || !object.isMesh) return;
        if (dynamicSet.has(object) || object.isInstancedMesh || object.isSkinnedMesh || object.morphTargetInfluences?.length) { addDynamic(object); return; }
        seen.add(object);
        const geometry = object.geometry, position = geometry?.attributes.position;
        if (!position) return;
        const size = geometry.index?.count ?? position.count, draw = geometry.drawRange;
        const ranges = Array.isArray(object.material) ? geometry.groups : [{ start: 0, count: size, materialIndex: 0 }];
        for (const range of ranges) {
          const start = Math.max(draw.start, range.start), end = Math.min(size, draw.start + draw.count, range.start + range.count);
          const count = Math.max(0, Math.floor((end - start) / 3));
          if (!count) continue;
          const batch = { object, materialIndex: range.materialIndex ?? 0, orientation: object.matrixWorld.determinant() < 0 ? -1 : 1 };
          const batchIndex = batches.length; batches.push(batch);
          descriptors.push({ object, geometry, position, start, count, batchIndex });
          triangleCount += count; staticObjects.add(object);
        }
      });
    }
    for (const root of dynamicRoots) root.traverse(addDynamic);
    vertices = new Float64Array(triangleCount * 9);
    triangleBatch = new Uint32Array(triangleCount); faceIndex = new Uint32Array(triangleCount); stamps = new Uint32Array(triangleCount);
    const bounds = new Float64Array(triangleCount * 6);
    let lowX = Infinity, lowY = Infinity, lowZ = Infinity, highX = -Infinity, highY = -Infinity, highZ = -Infinity, triangle = 0;
    for (const d of descriptors) {
      const index = d.geometry.index;
      for (let n = 0; n < d.count; n++, triangle++) {
        const offset = d.start + n * 3;
        pointA.fromBufferAttribute(d.position, index ? index.getX(offset) : offset).applyMatrix4(d.object.matrixWorld);
        pointB.fromBufferAttribute(d.position, index ? index.getX(offset + 1) : offset + 1).applyMatrix4(d.object.matrixWorld);
        pointC.fromBufferAttribute(d.position, index ? index.getX(offset + 2) : offset + 2).applyMatrix4(d.object.matrixWorld);
        const v = triangle * 9, b = triangle * 6;
        vertices[v] = pointA.x; vertices[v + 1] = pointA.y; vertices[v + 2] = pointA.z;
        vertices[v + 3] = pointB.x - pointA.x; vertices[v + 4] = pointB.y - pointA.y; vertices[v + 5] = pointB.z - pointA.z;
        vertices[v + 6] = pointC.x - pointA.x; vertices[v + 7] = pointC.y - pointA.y; vertices[v + 8] = pointC.z - pointA.z;
        bounds[b] = Math.min(pointA.x, pointB.x, pointC.x); bounds[b + 1] = Math.min(pointA.y, pointB.y, pointC.y); bounds[b + 2] = Math.min(pointA.z, pointB.z, pointC.z);
        bounds[b + 3] = Math.max(pointA.x, pointB.x, pointC.x); bounds[b + 4] = Math.max(pointA.y, pointB.y, pointC.y); bounds[b + 5] = Math.max(pointA.z, pointB.z, pointC.z);
        lowX = Math.min(lowX, bounds[b]); lowY = Math.min(lowY, bounds[b + 1]); lowZ = Math.min(lowZ, bounds[b + 2]);
        highX = Math.max(highX, bounds[b + 3]); highY = Math.max(highY, bounds[b + 4]); highZ = Math.max(highZ, bounds[b + 5]);
        triangleBatch[triangle] = d.batchIndex; faceIndex[triangle] = Math.floor(offset / 3);
      }
    }
    if (triangleCount) {
      minX = Math.floor((lowX - 1e-7) / cellSize) * cellSize;
      minY = Math.floor((lowY - 1e-7) / cellSize) * cellSize;
      minZ = Math.floor((lowZ - 1e-7) / cellSize) * cellSize;
      nx = Math.max(1, Math.ceil((highX + 1e-7 - minX) / cellSize));
      ny = Math.max(1, Math.ceil((highY + 1e-7 - minY) / cellSize));
      nz = Math.max(1, Math.ceil((highZ + 1e-7 - minZ) / cellSize));
      maxX = minX + nx * cellSize; maxY = minY + ny * cellSize; maxZ = minZ + nz * cellSize;
      for (let t = 0; t < triangleCount; t++) {
        const b = t * 6;
        const x0 = Math.max(0, Math.floor((bounds[b] - 1e-7 - minX) / cellSize));
        const y0 = Math.max(0, Math.floor((bounds[b + 1] - 1e-7 - minY) / cellSize));
        const z0 = Math.max(0, Math.floor((bounds[b + 2] - 1e-7 - minZ) / cellSize));
        const x1 = Math.min(nx - 1, Math.floor((bounds[b + 3] + 1e-7 - minX) / cellSize));
        const y1 = Math.min(ny - 1, Math.floor((bounds[b + 4] + 1e-7 - minY) / cellSize));
        const z1 = Math.min(nz - 1, Math.floor((bounds[b + 5] + 1e-7 - minZ) / cellSize));
        for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          const key = x + nx * (y + ny * z);
          let cell = cells.get(key); if (!cell) { cell = []; cells.set(key, cell); } cell.push(t);
        }
      }
    } else { nx = ny = nz = 0; }
    let references = 0;
    for (const [key, list] of cells) { const packed = new Uint32Array(list); references += packed.length; cells.set(key, packed); }
    const bytes = vertices.byteLength + triangleBatch.byteLength + faceIndex.byteLength + stamps.byteLength + references * 4;
    Object.assign(stats, { buildMilliseconds: now() - began, triangles: triangleCount, staticMeshes: staticObjects.size,
      dynamicMeshes: dynamicMeshes.length, dynamicInstances: dynamicMeshes.reduce((sum, object) => sum + (object.isInstancedMesh ? object.count : 0), 0),
      occupiedCells: cells.size, cellReferences: references, typedArrayBytes: bytes,
      estimatedMemoryMiB: (bytes + cells.size * 64 + batches.length * 128) / 1048576 });
    serial = 0; return stats;
  }

  function nearest(origin, unitDirection, near = 0, far = Infinity) {
    if (disposed || far < near || near < 0 || Number.isNaN(far)) return null;
    raycaster.ray.origin.copy(origin); raycaster.ray.direction.copy(unitDirection);
    const length = raycaster.ray.direction.length();
    if (!(length > 0)) return null;
    if (Math.abs(length - 1) > 1e-10) raycaster.ray.direction.multiplyScalar(1 / length);
    const ox = origin.x, oy = origin.y, oz = origin.z;
    const dx = raycaster.ray.direction.x, dy = raycaster.ray.direction.y, dz = raycaster.ray.direction.z;
    raycaster.near = near; raycaster.far = far;
    visibleCache.clear();
    let bestDistance = far, bestObject = null, bestMaterial = null, bestFace = 0, bestInstance;
    let visitedCells = 0, visitedTriangles = 0, dynamicCandidates = 0;
    serial = (serial + 1) >>> 0; if (serial === 0) { stamps.fill(0); serial = 1; }

    // Clip the DDA traversal to the scene bounds and the finite camera segment.
    let enter = near, exit = far, misses = !vertices.length;
    for (const [o, d, min, max] of [[ox, dx, minX, maxX], [oy, dy, minY, maxY], [oz, dz, minZ, maxZ]]) {
      if (Math.abs(d) < 1e-15) { if (o < min || o > max) misses = true; }
      else {
        let a = (min - o) / d, b = (max - o) / d;
        if (a > b) [a, b] = [b, a]; enter = Math.max(enter, a); exit = Math.min(exit, b);
        if (enter > exit) misses = true;
      }
    }
    if (!misses) {
      const start = Math.min(exit, enter + 1e-9);
      let x = Math.max(0, Math.min(nx - 1, Math.floor((ox + dx * start - minX) / cellSize)));
      let y = Math.max(0, Math.min(ny - 1, Math.floor((oy + dy * start - minY) / cellSize)));
      let z = Math.max(0, Math.min(nz - 1, Math.floor((oz + dz * start - minZ) / cellSize)));
      const sx = Math.sign(dx), sy = Math.sign(dy), sz = Math.sign(dz);
      const tx = dx ? cellSize / Math.abs(dx) : Infinity, ty = dy ? cellSize / Math.abs(dy) : Infinity, tz = dz ? cellSize / Math.abs(dz) : Infinity;
      let nextX = dx ? (minX + (x + (sx > 0 ? 1 : 0)) * cellSize - ox) / dx : Infinity;
      let nextY = dy ? (minY + (y + (sy > 0 ? 1 : 0)) * cellSize - oy) / dy : Infinity;
      let nextZ = dz ? (minZ + (z + (sz > 0 ? 1 : 0)) * cellSize - oz) / dz : Infinity;
      let cellStart = enter;
      while (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz && cellStart <= Math.min(exit, bestDistance)) {
        const cell = cells.get(x + nx * (y + ny * z)); visitedCells++;
        if (cell) for (let i = 0; i < cell.length; i++) {
          const t = cell[i]; if (stamps[t] === serial) continue; stamps[t] = serial; visitedTriangles++;
          const batch = batches[triangleBatch[t]];
          if (!objectVisible(batch.object)) continue;
          const material = materialFor(batch); if (!material || material.visible === false) continue;
          const v = t * 9;
          const ax = vertices[v], ay = vertices[v + 1], az = vertices[v + 2];
          const e1x = vertices[v + 3], e1y = vertices[v + 4], e1z = vertices[v + 5];
          const e2x = vertices[v + 6], e2y = vertices[v + 7], e2z = vertices[v + 8];
          const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
          const determinant = e1x * px + e1y * py + e1z * pz;
          if (determinant === 0) continue;
          const facing = determinant * batch.orientation;
          if (material.side === THREE.FrontSide && facing < 0) continue;
          if (material.side === THREE.BackSide && facing > 0) continue;
          const inv = 1 / determinant, vx = ox - ax, vy = oy - ay, vz = oz - az;
          const u = (vx * px + vy * py + vz * pz) * inv;
          if (u < -1e-10 || u > 1 + 1e-10) continue;
          const qx = vy * e1z - vz * e1y, qy = vz * e1x - vx * e1z, qz = vx * e1y - vy * e1x;
          const w = (dx * qx + dy * qy + dz * qz) * inv;
          if (w < -1e-10 || u + w > 1 + 1e-10) continue;
          const hitDistance = (e2x * qx + e2y * qy + e2z * qz) * inv;
          if (hitDistance < near || hitDistance > bestDistance) continue;
          const hx = ox + dx * hitDistance, hy = oy + dy * hitDistance, hz = oz + dz * hitDistance;
          if (!visibleMaterialAt(material, hx, hy, hz)) continue;
          bestDistance = hitDistance; bestObject = batch.object; bestMaterial = material; bestFace = faceIndex[t]; bestInstance = undefined;
        }
        const next = Math.min(nextX, nextY, nextZ);
        if (!Number.isFinite(next)) break;
        cellStart = next;
        if (nextX <= next + 1e-10) { x += sx; nextX += tx; }
        if (nextY <= next + 1e-10) { y += sy; nextY += ty; }
        if (nextZ <= next + 1e-10) { z += sz; nextZ += tz; }
      }
    }

    const acceptDynamic = (object, instanceId = undefined) => {
      for (const hit of dynamicHits) {
        hit.object = object;
        const material = Array.isArray(object.material) ? object.material[hit.face?.materialIndex ?? 0] : object.material;
        if (hit.distance <= bestDistance && visibleMaterialAt(material, hit.point.x, hit.point.y, hit.point.z)) {
          bestDistance = hit.distance; bestObject = object; bestMaterial = material; bestFace = hit.faceIndex; bestInstance = instanceId;
        }
      }
      dynamicHits.length = 0;
    };
    for (const object of dynamicMeshes) {
      if (!objectVisible(object)) continue;
      object.updateWorldMatrix(true, false);
      raycaster.far = bestDistance;
      if (!object.isInstancedMesh) {
        object.raycast(raycaster, dynamicHits); dynamicCandidates++; acceptDynamic(object); continue;
      }
      // Native InstancedMesh.raycast creates a Mesh test for every instance even
      // when the ray is nowhere near its small bars. Read the *current* matrices,
      // reject each with an exact conservative sphere, then raycast only survivors.
      // This is valid even when an instance is moved outside its original ring.
      const sphere = object.geometry.boundingSphere, center = sphere.center, elements = object.instanceMatrix.array;
      const world = object.matrixWorld.elements;
      const worldScale2 = Math.max(world[0] ** 2 + world[1] ** 2 + world[2] ** 2,
        world[4] ** 2 + world[5] ** 2 + world[6] ** 2, world[8] ** 2 + world[9] ** 2 + world[10] ** 2);
      proxy.geometry = object.geometry; proxy.material = object.material;
      for (let i = 0; i < object.count; i++) {
        const j = i * 16;
        const cx = elements[j] * center.x + elements[j + 4] * center.y + elements[j + 8] * center.z + elements[j + 12];
        const cy = elements[j + 1] * center.x + elements[j + 5] * center.y + elements[j + 9] * center.z + elements[j + 13];
        const cz = elements[j + 2] * center.x + elements[j + 6] * center.y + elements[j + 10] * center.z + elements[j + 14];
        const vx = world[0] * cx + world[4] * cy + world[8] * cz + world[12] - ox;
        const vy = world[1] * cx + world[5] * cy + world[9] * cz + world[13] - oy;
        const vz = world[2] * cx + world[6] * cy + world[10] * cz + world[14] - oz;
        const scale2 = Math.max(elements[j] ** 2 + elements[j + 1] ** 2 + elements[j + 2] ** 2,
          elements[j + 4] ** 2 + elements[j + 5] ** 2 + elements[j + 6] ** 2,
          elements[j + 8] ** 2 + elements[j + 9] ** 2 + elements[j + 10] ** 2);
        const projected = Math.max(near, Math.min(bestDistance, vx * dx + vy * dy + vz * dz));
        const ex = vx - dx * projected, ey = vy - dy * projected, ez = vz - dz * projected;
        if (ex * ex + ey * ey + ez * ez > sphere.radius * sphere.radius * scale2 * worldScale2 + 1e-10) continue;
        object.getMatrixAt(i, instanceMatrix); proxy.matrixWorld.multiplyMatrices(object.matrixWorld, instanceMatrix);
        raycaster.far = bestDistance; proxy.raycast(raycaster, dynamicHits); dynamicCandidates++;
        acceptDynamic(object, i);
      }
    }
    stats.queries++; Object.assign(stats.lastQuery, { cells: visitedCells, triangles: visitedTriangles, dynamicCandidates });
    return bestObject ? { distance: bestDistance, point: new THREE.Vector3(ox + dx * bestDistance, oy + dy * bestDistance, oz + dz * bestDistance),
      object: bestObject, material: bestMaterial, faceIndex: bestFace, instanceId: bestInstance } : null;
  }
  function dispose() {
    vertices = new Float64Array(0); triangleBatch = faceIndex = stamps = new Uint32Array(0);
    cells.clear(); batches = []; dynamicMeshes = []; visibleCache.clear(); disposed = true;
  }
  rebuild();
  return { nearest, rebuild, dispose, stats, isVisible: isCameraHitVisible };
}
