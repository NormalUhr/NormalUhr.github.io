import { exitStaysOpen, findWorkAnchor, unfinishedWorkAccess } from './work_planner.js';

const distance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);
const point = p => ({ x: p.x, z: p.z, m: p.m });

function segmentDistance(p, a, b) {
  const dx = b.x - a.x, dz = b.z - a.z, length = dx * dx + dz * dz;
  const t = length ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.z - a.z) * dz) / length)) : 0;
  return Math.hypot(p.x - a.x - t * dx, p.z - a.z - t * dz);
}

function reaches(body, a, b, minimum, maximum) {
  return distance(body, a) <= maximum && distance(body, b) <= maximum &&
    segmentDistance(body, a, b) >= minimum;
}

/** Prefer work already within the planted worker's reach, then nearby work. */
export function orderRestoreRuns(world, w, runs) {
  if (!w || runs.length < 2) return runs;
  const minimum = world.rakeFootClearance(), maximum = w.id >= 4 ? .575 : .835;
  const planted = world.walkNav.free(w.x, w.z);
  const tool = Number.isFinite(w.toolX) && Number.isFinite(w.toolZ) ? { x: w.toolX, z: w.toolZ } : w;
  const scores = new Map();
  for (const run of runs) {
    let sameStance = planted;
    for (let i = run.i; sameStance && i < run.j; i++) sameStance = reaches(w, world.points[i], world.points[i + 1], minimum, maximum);
    const a = world.points[run.i], b = world.points[run.j];
    scores.set(run, { sameStance, tool: Math.min(distance(tool, a), distance(tool, b)),
      body: Math.min(distance(w, a), distance(w, b)) });
  }
  return [...runs].sort((a, b) => {
    const sa = scores.get(a), sb = scores.get(b);
    return Number(sb.sameStance) - Number(sa.sameStance) ||
      (sa.sameStance ? sa.tool - sb.tool : sa.body - sb.body) || a.i - b.i;
  });
}

/**
 * Before opening sand for an awkward residual gap, use a smaller safe stroke.
 * The topology proof preserves access to individual missing edges; those edges
 * need not all share one stance. Only a real clear route permits this fallback.
 */
export function findClearRestoreFragment(world, w, task) {
  if (!w || task.kind !== 'restore' || task.path.length < 3) return null;
  const tool = Number.isFinite(w.toolX) && Number.isFinite(w.toolZ) ? { x: w.toolX, z: w.toolZ } : w;
  const candidates = [], seen = new Set(), last = task.path.length - 1;
  for (let span = Math.ceil(last / 2); span >= 1; span = span === 1 ? 0 : Math.ceil(span / 2)) {
    for (const [from, to] of [[0, span], [last - span, last]]) {
      const key = `${from}:${to}`; if (seen.has(key)) continue; seen.add(key);
      const path = task.path.slice(from, to + 1).map((p, i) => ({ ...p, working: i > 0 }));
      candidates.push({ ...task, path, anchor: undefined });
    }
  }
  candidates.sort((a, b) => (b.path.length - a.path.length) ||
    Math.min(distance(tool, a.path[0]), distance(tool, a.path.at(-1))) -
    Math.min(distance(tool, b.path[0]), distance(tool, b.path.at(-1))));
  for (const candidate of candidates) {
    const anchor = findWorkAnchor(world, w, candidate, { clearOnly: true });
    if (anchor) return extendRestoreStroke(world, w, candidate, anchor);
  }
  return null;
}

/**
 * Extend an already selected, safe repair into one reachable continuous arc.
 * The historical small runs still choose the dependency. Only the actual
 * missing neighbours belonging to this same worker can join its brush stroke.
 * The complete stroke must leave home and all other unfinished work reachable.
 */
export function extendRestoreStroke(world, w, task, anchor, { maximumLength = 1.30 } = {}) {
  if (!w || task.kind !== 'restore' || !anchor || !task.path?.length || !w.repairs) return { ...task, anchor };
  const step = world.points[1].m - world.points[0].m;
  const end = Math.min(world.coverage.length, Math.floor(Math.min(world.minute, world.recordedMinute) / step));
  const from = Math.min(task.path[0].m, task.path.at(-1).m), to = Math.max(task.path[0].m, task.path.at(-1).m);
  const first = Math.round(from / step), last = Math.round(to / step);
  // Restored legacy tasks may have fractional endpoints. Leave those exact
  // saved strokes alone rather than replacing their partially worked path.
  if (Math.abs(first * step - from) > 1e-7 || Math.abs(last * step - to) > 1e-7) return { ...task, anchor };
  const minimum = world.rakeFootClearance(), maximum = w.id >= 4 ? .575 : .835;
  const others = world.workers.filter(worker => worker !== w);
  const missing = i => i >= 0 && i < end && !world.coverage[i] && w.repairs[i] !== undefined &&
    !others.some(worker => worker.repairs?.[i] !== undefined) &&
    !world.isDirtyStroke(world.points[i], world.points[i + 1]);
  const eligible = i => missing(i) && reaches(anchor, world.points[i], world.points[i + 1], minimum, maximum);
  let length = 0;
  for (let i = first; i < last; i++) {
    if (!eligible(i)) return { ...task, anchor };
    length += distance(world.points[i], world.points[i + 1]);
  }
  let left = first, right = last;
  // Grow both ends near the original target. Geometry, not old sweep ranks,
  // limits this one planted stroke; the full dependency check below protects
  // any branch that still needs its entrance.
  while (left > 0 || right < end) {
    const canLeft = eligible(left - 1), canRight = eligible(right);
    if (!canLeft && !canRight) break;
    const takeLeft = canLeft && (!canRight || first - left <= right - last);
    const index = takeLeft ? left - 1 : right, extra = distance(world.points[index], world.points[index + 1]);
    if (length + extra > maximumLength + 1e-7) break;
    length += extra;
    if (takeLeft) left--; else right++;
  }
  const tool = Number.isFinite(w.toolX) && Number.isFinite(w.toolZ) ? { x: w.toolX, z: w.toolZ } : w;
  const make = (a, b, body = anchor) => {
    let path = world.points.slice(a, b + 1).map(point);
    // Start at the nearer endpoint. Reversing a repair never changes the clock
    // tip; coverage still follows the same physically painted segment range.
    if (distance(tool, path.at(-1)) + .025 < distance(tool, path[0])) path.reverse();
    path = path.map((p, i) => ({ ...p, working: i > 0 }));
    return { ...task, path, anchor: body };
  };
  // A stance selected for a tiny seed can sit at one end of an otherwise
  // simple arc. Once, consider the whole short gap from a better clear stance.
  // It may trade a small walk for eliminating several full raise/lower cycles;
  // it may never open another passage or join across intact sand.
  let wholeLeft = first, wholeRight = last, wholeLength = 0;
  for (let i = first; i < last; i++) wholeLength += distance(world.points[i], world.points[i + 1]);
  while (wholeLength <= maximumLength && missing(wholeLeft - 1)) {
    wholeLength += distance(world.points[wholeLeft - 1], world.points[wholeLeft]); wholeLeft--;
  }
  while (wholeLength <= maximumLength && missing(wholeRight)) {
    wholeLength += distance(world.points[wholeRight], world.points[wholeRight + 1]); wholeRight++;
  }
  if (wholeLength <= maximumLength && (wholeLeft < first || wholeRight > last)) {
    const complete = make(wholeLeft, wholeRight), body = findWorkAnchor(world, w, complete, { clearOnly: true });
    if (body) {
      const oldWalk = world.routeLength(w, anchor.route || []), newWalk = world.routeLength(w, body.route || []);
      const savedLifts = Math.max(1, Math.ceil(wholeLength / Math.max(.01, length)) - 1);
      if (newWalk <= oldWalk + Math.min(.90, savedLifts * .45 + .08)) return { ...complete, anchor: body };
    }
  }
  // A long arc may seal another worksite even though its seed was safe. Try
  // shorter geometric spans, then retain the proven seed without opening sand.
  for (let attempt = 0; attempt < 4 && (left < first || right > last); attempt++) {
    const extended = make(left, right);
    if (exitStaysOpen(world, anchor, extended, task.keepAccess || []) && unfinishedWorkAccess(world, w, extended).open) return extended;
    left = first - Math.floor((first - left) / 2);
    right = last + Math.floor((right - last) / 2);
  }
  return make(first, last);
}
