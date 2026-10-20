/*
 * A work itinerary only: no scene, clock, randomness or sand writes.
 * Occupancy is formed by exact capsule/rectangle intersections. Every input
 * stroke, including a tiny isolated paw dab, contributes its complete brush
 * support; this is coverage consolidation, never point decimation.
 */
// Adjacent rows remain within the smoothing blade's full-pressure core. The
// former .18m spacing used less than half of a .38m blade and repeatedly swept
// the same strip. .25/.38 is still below the core diameter ratio of .68.
const CELL = .25;
const MARK_SUPPORT = .060;
const BRUSH_CLEARANCE = .195;
const TAU = Math.PI * 2;
const BOUNDS = { minX: -5.40, maxX: 5.40, minZ: -3.91, maxZ: 3.91 };
const point = p => ({ x: p.x, z: p.z });
const distance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const mixPoint = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t });
function segmentDistance(p, a, b) {
  const dx = b.x - a.x, dz = b.z - a.z, length2 = dx * dx + dz * dz;
  const t = length2 ? clamp(((p.x - a.x) * dx + (p.z - a.z) * dz) / length2, 0, 1) : 0;
  return Math.hypot(p.x - a.x - t * dx, p.z - a.z - t * dz);
}
function clipRectangle(a, b, box) {
  let lo = 0, hi = 1;
  for (const [p, d, min, max] of [[a.x, b.x - a.x, box.minX, box.maxX], [a.z, b.z - a.z, box.minZ, box.maxZ]]) {
    if (Math.abs(d) < 1e-12) { if (p < min || p > max) return null; }
    else { const t0 = (min - p) / d, t1 = (max - p) / d; lo = Math.max(lo, Math.min(t0, t1)); hi = Math.min(hi, Math.max(t0, t1)); if (hi < lo) return null; }
  }
  return [mixPoint(a, b, lo), mixPoint(a, b, hi)];
}
function rectangleDistance(a, b, box) {
  if (clipRectangle(a, b, box)) return 0;
  let best = Infinity;
  for (const p of [a, b]) best = Math.min(best, Math.hypot(p.x - clamp(p.x, box.minX, box.maxX), p.z - clamp(p.z, box.minZ, box.maxZ)));
  for (const x of [box.minX, box.maxX]) for (const z of [box.minZ, box.maxZ]) best = Math.min(best, segmentDistance({ x, z }, a, b));
  return best;
}
function clearSegment(a, b, obstacles, brushClearance = BRUSH_CLEARANCE) {
  return obstacles.every(o => segmentDistance(o, a, b) >= o.r + brushClearance - 1e-8);
}
function clipStones(a, b, obstacles, brushClearance = BRUSH_CLEARANCE) {
  const bounded = clipRectangle(a, b, BOUNDS); if (!bounded) return [];
  [a, b] = bounded;
  const dx = b.x - a.x, dz = b.z - a.z, aa = dx * dx + dz * dz;
  if (aa < 1e-12) return [];
  const blocked = [];
  for (const o of obstacles) {
    const ox = a.x - o.x, oz = a.z - o.z, radius = o.r + brushClearance;
    const bb = ox * dx + oz * dz, cc = ox * ox + oz * oz - radius * radius, disc = bb * bb - aa * cc;
    if (disc <= 0) continue;
    const root = Math.sqrt(disc), lo = Math.max(0, (-bb - root) / aa), hi = Math.min(1, (-bb + root) / aa);
    if (hi > lo) blocked.push([lo, hi]);
  }
  blocked.sort((x, y) => x[0] - y[0]);
  const pieces = []; let from = 0;
  for (const [lo, hi] of blocked) {
    if (lo > from + 1e-7) pieces.push([mixPoint(a, b, from), mixPoint(a, b, lo)]);
    from = Math.max(from, hi);
  }
  if (from < 1 - 1e-7) pieces.push([mixPoint(a, b, from), point(b)]);
  return pieces;
}
function addAngle(intervals, angle, pad) {
  angle = ((angle % TAU) + TAU) % TAU;
  const lo = angle - pad, hi = angle + pad;
  if (lo < 0) { intervals.push([0, hi], [lo + TAU, TAU]); }
  else if (hi > TAU) intervals.push([lo, TAU], [0, hi - TAU]);
  else intervals.push([lo, hi]);
}

function planMaintenancePath(points, obstacles = [], { width = .38, rowPhase = 0 } = {}) {
  const cell = CELL * width / .38, brushClearance = width * .5 + .005;
  if (!Array.isArray(points) || !points.length) return [];
  if (!points.every(p => p && Number.isFinite(p.x) && Number.isFinite(p.z))) throw new TypeError('Maintenance points must have finite coordinates.');
  const stones = obstacles.filter(o => o && Number.isFinite(o.x) && Number.isFinite(o.z) && Number.isFinite(o.r) && o.r > 0);
  const segments = points.map((p, i) => [i && !p.break ? points[i - 1] : p, p]);
  // Align the sweep with the main trail, so a diagonal line does not become
  // scores of short horizontal strokes. All later coordinates stay in metres.
  const mean = points.reduce((s, p) => ({ x: s.x + p.x / points.length, z: s.z + p.z / points.length }), { x: 0, z: 0 });
  let xx = 0, zz = 0, xz = 0;
  for (const p of points) { const x = p.x - mean.x, z = p.z - mean.z; xx += x * x; zz += z * z; xz += x * z; }
  const angle = .5 * Math.atan2(2 * xz, xx - zz), cs = Math.cos(angle), sn = Math.sin(angle);
  const offset = cell * rowPhase;
  const local = p => ({ x: (p.x - mean.x) * cs + (p.z - mean.z) * sn, z: -(p.x - mean.x) * sn + (p.z - mean.z) * cs + offset });
  const world = p => ({ x: mean.x + p.x * cs - (p.z - offset) * sn, z: mean.z + p.x * sn + (p.z - offset) * cs });
  const rows = new Map();
  for (const [first, second] of segments) {
    const a = local(first), b = local(second);
    const x0 = Math.floor((Math.min(a.x, b.x) - MARK_SUPPORT) / cell), x1 = Math.floor((Math.max(a.x, b.x) + MARK_SUPPORT) / cell);
    const z0 = Math.floor((Math.min(a.z, b.z) - MARK_SUPPORT) / cell), z1 = Math.floor((Math.max(a.z, b.z) + MARK_SUPPORT) / cell);
    for (let iz = z0; iz <= z1; iz++) for (let ix = x0; ix <= x1; ix++) {
      const box = { minX: ix * cell, maxX: (ix + 1) * cell, minZ: iz * cell, maxZ: (iz + 1) * cell };
      if (rectangleDistance(a, b, box) > MARK_SUPPORT + 1e-9) continue;
      if (!rows.has(iz)) rows.set(iz, new Set()); rows.get(iz).add(ix);
    }
  }
  const strokes = [];
  for (const iz of [...rows.keys()].sort((a, b) => a - b)) {
    const columns = [...rows.get(iz)].sort((a, b) => a - b);
    for (let start = 0; start < columns.length;) {
      let end = start;
      // A single empty cell is narrower than the rake. Sweeping across it is
      // less work and preserves continuous motion between neighbouring prints.
      while (end + 1 < columns.length && columns[end + 1] <= columns[end] + 2) end++;
      const a = world({ x: columns[start] * cell - .025, z: (iz + .5) * cell });
      const b = world({ x: (columns[end] + 1) * cell + .025, z: (iz + .5) * cell });
      for (const piece of clipStones(a, b, stones, brushClearance)) strokes.push(piece);
      start = end + 1;
    }
  }
  // Broad straight passes stop outside each stone. A tangential sweep reaches
  // the nearby paw impressions from the outside, keeping the actual blade clear.
  for (const stone of stones) {
    const intervals = [], radius = stone.r + brushClearance + .006;
    for (const [a, b] of segments) {
      if (segmentDistance(stone, a, b) > stone.r + .31) continue;
      const dx = b.x - a.x, dz = b.z - a.z, length2 = dx * dx + dz * dz;
      const t = length2 ? clamp(((stone.x - a.x) * dx + (stone.z - a.z) * dz) / length2, 0, 1) : 0;
      for (const p of [a, b, mixPoint(a, b, t)]) {
        const d = distance(p, stone); if (d > stone.r + .38) continue;
        const pad = Math.asin(Math.min(.9, .115 / Math.max(.13, d))) + .07;
        addAngle(intervals, Math.atan2(p.z - stone.z, p.x - stone.x), pad);
      }
    }
    intervals.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const interval of intervals) {
      if (merged.length && interval[0] <= merged.at(-1)[1] + .06) merged.at(-1)[1] = Math.max(merged.at(-1)[1], interval[1]);
      else merged.push([...interval]);
    }
    for (const [lo, hi] of merged) {
      const count = Math.max(1, Math.ceil((hi - lo) * radius / .055)); let run = [];
      for (let i = 0; i <= count; i++) {
        const t = lo + (hi - lo) * i / count, p = { x: stone.x + Math.cos(t) * radius, z: stone.z + Math.sin(t) * radius };
        const inside = p.x >= BOUNDS.minX && p.x <= BOUNDS.maxX && p.z >= BOUNDS.minZ && p.z <= BOUNDS.maxZ;
        if (!inside || !clearSegment(p, p, stones, brushClearance) || run.length && !clearSegment(run.at(-1), p, stones, brushClearance)) {
          if (run.length > 1) strokes.push(run); run = inside && clearSegment(p, p, stones, brushClearance) ? [p] : [];
        } else run.push(p);
      }
      if (run.length > 1) strokes.push(run);
    }
  }
  // Join nearby passes when the whole blade can travel between them. Longer
  // gaps and stones retain an explicit lift for the existing walking planner.
  const out = []; let cursor = points[0];
  while (strokes.length) {
    let best = 0, reverse = false, bestDistance = Infinity;
    for (let i = 0; i < strokes.length; i++) for (const rev of [false, true]) {
      const d = distance(cursor, rev ? strokes[i].at(-1) : strokes[i][0]);
      if (d < bestDistance) { bestDistance = d; best = i; reverse = rev; }
    }
    const stroke = strokes.splice(best, 1)[0]; if (reverse) stroke.reverse();
    const join = out.length && bestDistance < .56 && clearSegment(cursor, stroke[0], stones, brushClearance);
    if (!join || bestDistance > 1e-8) out.push({ ...stroke[0], ...(!join ? { break: true } : {}) });
    for (let i = 1; i < stroke.length; i++) out.push(point(stroke[i]));
    cursor = stroke.at(-1);
  }
  return out;
}

function workLength(points) {
  let length = 0;
  for (let i = 1; i < points.length; i++) if (!points[i].break) length += distance(points[i - 1], points[i]);
  return length;
}

function itineraryCost(points) {
  // Estimate the existing physical speeds and tool transitions. A slightly
  // shorter drawing path is not better if it requires several extra walks.
  let seconds = 0;
  for (let i = 1; i < points.length; i++) {
    const d = distance(points[i - 1], points[i]);
    seconds += points[i].break ? d / .46 + 1.05 : d / .33;
  }
  return seconds;
}

/** Return raw world-space points with pen-lift `break` markers for makePath. */
export function animalMaintenancePath(points, obstacles = [], { width = .38 } = {}) {
  const boundaryAligned = planMaintenancePath(points, obstacles, { width });
  if (boundaryAligned.length < 2) return boundaryAligned;
  // A narrow trail centred on its principal axis otherwise straddles two rows
  // for its entire length. Try a row centred there too, preserving the same
  // complete capsule/cell coverage and the same blade clearance around stones.
  const centreAligned = planMaintenancePath(points, obstacles, { width, rowPhase: .5 });
  return centreAligned.length > 1 && itineraryCost(centreAligned) < itineraryCost(boundaryAligned)
    ? centreAligned : boundaryAligned;
}

/** Consolidate repeated manual passes only when the actual work shrinks. */
export function consolidateManualPath(points, obstacles = [], { width = .38 } = {}) {
  if (!Array.isArray(points) || points.length < 4) return null;
  const originalLength = workLength(points);
  if (originalLength < .9) return null;
  const candidate = animalMaintenancePath(points, obstacles, { width });
  // Ordinary small marks and single, purposeful arcs keep their exact local
  // itinerary. Repeatedly drawing over one place must not command an equal
  // number of identical smoothing passes. The raw input itself is untouched.
  return candidate.length > 1 && workLength(candidate) <= originalLength * .65 ? candidate : null;
}
