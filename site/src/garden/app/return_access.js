import { createSandField } from './sand_field.js';
import { SurfaceIndex } from './surface_index.js';
import { Navigation } from './navigation.js';

// A new mark can cut the already prepared exit behind the keeper. That mark
// is a prerequisite of the current visit even if the deepest clock fragment
// is elsewhere. Planning only returns its local position; the normal worker
// still has to walk there, lower the rake and physically clear it.
const caches = new WeakMap();
const LOCAL_RADIUS = .60, TRAIL_RADIUS = .40, CELL = .45;
const point = p => ({ x: p.x, z: p.z });
const distance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);
const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
const valid = p => p && Number.isFinite(p.x) && Number.isFinite(p.z);

function segmentDistance(p, a, b) {
  const dx = b.x - a.x, dz = b.z - a.z, length = dx * dx + dz * dz;
  const t = length ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.z - a.z) * dz) / length)) : 0;
  return Math.hypot(p.x - a.x - dx * t, p.z - a.z - dz * t);
}

// Match the scheduler's bounded raw-edge admission, including disconnected
// pawprints. A proof must not clear a whole animal visit to justify one job.
function localPoints(group, centre) {
  const out = [];
  for (const damage of group) {
    const local = [];
    const append = (a, b) => {
      const last = local.at(-1);
      if (!last || distance(last, a) > 1e-7) local.push({ ...point(a), break: true });
      if (distance(a, b) > 1e-8) local.push(point(b));
    };
    for (let i = 0; i < damage.points.length; i++) {
      const b = damage.points[i], a = i && !b.break ? damage.points[i - 1] : b;
      const n = Math.max(1, Math.ceil(distance(a, b) / .08));
      for (let k = 0; k < n; k++) {
        const p = { x: a.x + (b.x - a.x) * k / n, z: a.z + (b.z - a.z) * k / n };
        const q = { x: a.x + (b.x - a.x) * (k + 1) / n, z: a.z + (b.z - a.z) * (k + 1) / n };
        if (segmentDistance(centre, p, q) < LOCAL_RADIUS) append(p, q);
      }
    }
    local.forEach((p, i) => out.push({ ...p, ...(i === 0 ? { break: true } : {}) }));
  }
  return out;
}

function candidatesFor(world, w) {
  const grid = new Map();
  for (const p of w.accessTrail) {
    if (!valid(p)) continue;
    const key = `${Math.floor(p.x / CELL)}:${Math.floor(p.z / CELL)}`;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(p);
  }
  const onTrail = p => {
    const x = Math.floor(p.x / CELL), z = Math.floor(p.z / CELL);
    for (let dz = -1; dz <= 1; dz++) for (let dx = -1; dx <= 1; dx++)
      if ((grid.get(`${x + dx}:${z + dz}`) || []).some(q => distance(p, q) < TRAIL_RADIUS)) return true;
    return false;
  };
  const records = [...world.damages, ...(world.activeMark ? [world.activeMark] : [])], candidates = [];
  const groups = new Map();
  for (const damage of records) {
    const key = damage.sourceKey || damage;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(damage);
  }
  for (const [key, group] of groups) {
    const centres = [];
    for (const damage of group) for (let i = 0; i < damage.points.length; i++) {
      const b = damage.points[i], a = i && !b.break ? damage.points[i - 1] : b;
      if (!valid(a) || !valid(b)) continue;
      const n = Math.max(1, Math.ceil(distance(a, b) / .08));
      for (let k = 0; k <= n; k++) {
        const p = { x: a.x + (b.x - a.x) * k / n, z: a.z + (b.z - a.z) * k / n };
        if (!onTrail(p) || centres.some(q => distance(p, q) < .45)) continue;
        centres.push(p);
        candidates.push({ damage, group, key, point: p, distance: distance(w, p) });
      }
    }
  }
  return candidates.sort((a, b) => a.distance - b.distance);
}

function descriptor(world, candidate) {
  if (!candidate) return null;
  const index = world.damages.indexOf(candidate.damage), active = world.activeMark === candidate.damage;
  if (index < 0 && !active) return null;
  const ready = !active && world.damageReady(candidate.damage);
  return { index, point: point(candidate.point), ready, waiting: !ready };
}

function scratchFor(cache, world, field) {
  if (!cache.scratch || cache.scratch.width !== field.width || cache.scratch.height !== field.height) {
    cache.scratch = createSandField({ width: field.width, height: field.height });
    cache.index = new SurfaceIndex(cache.scratch);
  }
  if (cache.base !== field.base) { cache.scratch.base.set(field.base); cache.base = field.base; }
  return {
    sand: cache.scratch, surfaceIndex: cache.index, nav: world.nav,
    surfaceFree: world.surfaceFree, feetVisible: world.feetVisible,
  };
}

/**
 * Return {index, point, ready, waiting} for a causally necessary local exit
 * obstruction, or null. An active pointer has index -1 and ready false.
 * The caller waits for an unfinished input, or admits only points within the
 * scheduler's .60 m scope around point. No live sand, queue or body is edited.
 */
export function findReturnBlocker(world, w) {
  if (!w?.accessTrail?.length || (!world.damages.length && !world.activeMark)) return null;
  const field = world.sand.field || world.sand;
  if (!field?.heights || !field.base || !Number.isSafeInteger(field.revision)) return null;
  let cache = caches.get(world);
  if (!cache) { cache = {}; caches.set(world, cache); }
  const inputs = [field, field.revision, w, w.x, w.z, w.accessTrail, w.accessTrail.length,
    world.damages, world.damages.length, world.activeMark, world.stats.visitorSegments, world.stats.animalIncidents];
  for (const d of [...world.damages, ...(world.activeMark ? [world.activeMark] : [])])
    inputs.push(d, d.points, d.points.length, d.lastTouchedAt, d.sourceKey);
  if (cache.inputs && same(cache.inputs, inputs)) return descriptor(world, cache.result);
  cache.inputs = inputs; cache.result = null;
  const home = world.habitat.home.threshold || { x: -3.2, z: -4.12 };
  if (distance(w, home) < .025 || world.travelRoute(w, home).length) return null;
  const candidates = candidatesFor(world, w);
  if (!candidates.length) return null;
  const shadow = scratchFor(cache, world, field), sand = shadow.sand;
  if (Object.keys(sand.bounds).some(key => sand.bounds[key] !== field.bounds?.[key])) return null;
  const tasks = new Map();
  const taskFor = candidate => {
    if (!tasks.has(candidate)) {
      const points = localPoints(candidate.group, candidate.point);
      const damage = { ...candidate.damage, points };
      tasks.set(candidate, world.prepareClean(points, damage, w));
    }
    return tasks.get(candidate);
  };
  const opensReturn = selected => {
    // Exact float samples, without save quantization. Only this private copy
    // receives the hypothetical brush; the real world is never rolled back.
    sand.heights.set(field.heights);
    for (const candidate of selected) {
      const task = taskFor(candidate), path = task.path;
      for (let i = 1; i < path.length; i++) {
        if (path[i].working === false) continue;
        const a = path[i - 1], b = path[i], n = Math.max(1, Math.ceil(distance(a, b) / .0165));
        for (let k = 0; k < n; k++) sand.stroke(
          a.x + (b.x - a.x) * k / n, a.z + (b.z - a.z) * k / n,
          a.x + (b.x - a.x) * (k + 1) / n, a.z + (b.z - a.z) * (k + 1) / n,
          { kind: 'smooth', width: task.cleanWidth || .38, strength: .98 });
      }
    }
    cache.index.allDirty(sand.revision);
    const navigator = new Navigation(world.habitat.obstacles, {
      surface: (x, z) => shadow.surfaceFree(x, z), cell: .16, fineCell: .05, surfaceStep: .005,
    });
    const route = navigator.route(w, home);
    if (!route.length) return false;
    let a = w;
    for (const b of route) { if (!shadow.feetVisible(a, b)) return false; a = b; }
    return world.peopleRoute(w, route);
  };
  // Usually a single recent stroke crossed the lane. More than one crossing
  // can be in series, so retain a minimal set of necessary local clearances
  // rather than requiring each mark alone to reopen the whole exit.
  if (opensReturn([candidates[0]])) cache.result = candidates[0];
  else if (candidates.length > 1 && opensReturn(candidates)) {
    let necessary = candidates;
    for (let i = necessary.length - 1; i >= 0; i--) {
      const without = necessary.filter((_, j) => j !== i);
      if (without.length && opensReturn(without)) necessary = without;
    }
    cache.result = necessary[0];
  }
  return descriptor(world, cache.result);
}
