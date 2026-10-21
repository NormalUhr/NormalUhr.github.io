import { TAU, FLOOR_HEIGHT, polar, shortest, wrap, clamp } from './mechanism.js';
import { words } from './words.js';

// These coordinates are shared with the built geometry, not a screen-space nav mesh.
export const FLOOR_OFFSET = .025;
export const CORE_WALK_RADIUS = 3.55;
export const STAIR_RADIUS = 30.7;
export const STAIR_ANGLE = Math.PI / 6;
export const floorY = level => level * FLOOR_HEIGHT + FLOOR_OFFSET;
export const innerRadius = level => level < 4 ? 22 : 27;
export const walkRadius = level => innerRadius(level) + 1.4;
export const angleOf = p => wrap(Math.atan2(p.x, -p.z));
export const distance = (a, b) => Math.hypot(b.x - a.x, b.z - a.z);
export const copyPoint = p => ({ x: p.x, y: p.y, z: p.z });

export function line(from, to, support = { type: 'floor', level: Math.round(to.y / 6) }, speed = 1) {
  return { kind: 'line', from: copyPoint(from), to: copyPoint(to), length: distance(from, to), progress: 0, support, speed };
}

export function arc(radius, fromAngle, delta, level, support = { type: 'floor', level }, speed = 1) {
  return { kind: 'arc', radius, fromAngle, delta, level, length: Math.abs(delta) * radius, progress: 0, support, speed };
}

export function segmentPoint(segment, at = segment.progress) {
  const t = segment.length > 1e-8 ? clamp(at / segment.length) : 1;
  if (segment.kind === 'line') return {
    x: segment.from.x + (segment.to.x - segment.from.x) * t,
    y: segment.from.y + (segment.to.y - segment.from.y) * t,
    z: segment.from.z + (segment.to.z - segment.from.z) * t,
  };
  const a = segment.fromAngle + segment.delta * t;
  const p = polar(segment.radius, a, floorY(segment.level));
  if (segment.kind === 'stairs') {
    const fraction = clamp((a - segment.baseAngle) / STAIR_ANGLE);
    p.y = floorY(segment.lowerLevel) + Math.min(30, Math.ceil(fraction * 30 - 1e-8)) * .2;
  }
  return p;
}

export function appendLine(path, from, to, support, speed = 1) {
  if (distance(from, to) > .00001) path.push(line(from, to, support, speed));
  return copyPoint(to);
}

export function appendRing(path, from, level, targetAngle, radius = walkRadius(level), supportType = 'floor') {
  const a = angleOf(from), delta = shortest(a, targetAngle);
  // Opposing walkers have fixed, separate lanes. There is no reciprocal avoidance
  // steering that can turn a crowd into a group of orbiting characters.
  const laneRadius = radius + (delta >= 0 ? .30 : -.30);
  const support = { type: supportType, level, direction: Math.sign(delta) || 1 };
  let p = appendLine(path, from, polar(laneRadius, a, floorY(level)), support);
  if (Math.abs(delta) > 1e-7) path.push(arc(laneRadius, a, delta, level, support));
  p = polar(laneRadius, targetAngle, floorY(level));
  return appendLine(path, p, polar(radius, targetAngle, floorY(level)), support);
}

function chooseStair(angle, direction, finalAngle, last) {
  let best = 0, score = Infinity;
  for (let i = 0; i < 3; i++) {
    const base = i * TAU / 3;
    const entrance = base + (direction > 0 ? -Math.PI / 60 : STAIR_ANGLE + Math.PI / 45);
    const exit = base + (direction > 0 ? STAIR_ANGLE + Math.PI / 45 : -Math.PI / 60);
    const s = Math.abs(shortest(angle, entrance)) + (last ? .72 * Math.abs(shortest(exit, finalAngle)) : 0);
    if (s < score) { score = s; best = i; }
  }
  return best;
}

/** Continuous, supported route through the three actual fixed stair stacks. */
export function outerRoute(from, fromLevel, toLevel, toAngle) {
  const path = [];
  let p = copyPoint(from), level = fromLevel;
  p = appendLine(path, p, polar(walkRadius(level), angleOf(p), floorY(level)), { type: 'floor', level });
  while (level !== toLevel) {
    const direction = Math.sign(toLevel - level), next = level + direction;
    const stack = chooseStair(angleOf(p), direction, toAngle, next === toLevel);
    const base = stack * TAU / 3, low = Math.min(level, next);
    const r = STAIR_RADIUS + (direction > 0 ? -.28 : .28);
    // Separate radial approaches as well as the stairs themselves. Otherwise two
    // residents can meet face to face on the long radial access to a stair stack.
    const laneAngle = direction * .012;
    const start = base - Math.PI / 60 - laneAngle;
    const end = base + STAIR_ANGLE + Math.PI / 45 + laneAngle;
    const entrance = direction > 0 ? start : end;
    p = appendRing(path, p, level, entrance);
    p = appendLine(path, p, polar(r, entrance, floorY(level)), { type: 'floor', level });
    const approachEnd = direction > 0 ? base : base + STAIR_ANGLE;
    path.push(arc(r, entrance, approachEnd - entrance, level, { type: 'floor', level }));
    const stair = arc(r, approachEnd, direction * STAIR_ANGLE, level,
      { type: 'stairs', id: `stair-${low}-${stack}`, lowerLevel: low, stack, baseAngle: base, direction }, .74);
    stair.kind = 'stairs'; stair.lowerLevel = low; stair.baseAngle = base;
    path.push(stair);
    const stairEnd = direction > 0 ? base + STAIR_ANGLE : base;
    const departureEnd = direction > 0 ? end : start;
    path.push(arc(r, stairEnd, departureEnd - stairEnd, next, { type: 'floor', level: next }));
    p = polar(r, departureEnd, floorY(next));
    p = appendLine(path, p, polar(walkRadius(next), departureEnd, floorY(next)), { type: 'floor', level: next });
    level = next;
  }
  appendRing(path, p, level, toAngle);
  return path;
}

export function coreRoute(from, level, target) {
  const path = [];
  const a = angleOf(from), b = angleOf(target), support = { type: 'core', level };
  let p = appendLine(path, from, polar(CORE_WALK_RADIUS, a, floorY(level)), support, .86);
  p = appendRing(path, p, level, b, CORE_WALK_RADIUS, 'core');
  appendLine(path, p, { ...target, y: floorY(level) }, support, .86);
  return path;
}

export function fallbackDestinations() {
  const sites = [];
  const angles = [55, 90, 175, 210, 295, 330];
  for (let level = 0; level < 7; level++) for (let i = 0; i < 12; i++) {
    const a = angles[i % 6] * Math.PI / 180 + (i < 6 ? -.023 : .023);
    const position = polar(29.15, a, floorY(level));
    sites.push({ id: `quiet-${level}-${i}`, level, angle: a, radius: 29.15,
      position, activity: 'observe', facing: a + Math.PI,
      label: words.places.viewpoint, capacity: 1,
      approach: [polar(walkRadius(level), a, floorY(level)), position] });
  }
  return sites;
}
