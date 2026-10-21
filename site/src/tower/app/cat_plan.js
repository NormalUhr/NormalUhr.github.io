import { GARDEN_PLAN, gardenPolar } from './garden_plan.js';
import { clamp, polar, shortest } from './mechanism.js';
import { angleOf, appendLine, arc, floorY, segmentPoint } from './navigation.js';
import { createOutdoorRouting, gardenStepHeight, outdoorSegmentPoint } from './outdoor_routes.js';
import { words } from './words.js';

const clone = value => JSON.parse(JSON.stringify(value));
export const pointAt = (segment, at = segment.progress) => outdoorSegmentPoint(segment, at) || segmentPoint(segment, at);
const finitePoint = p => !!p && [p.x, p.y, p.z].every(Number.isFinite);
const flatDistance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);
const distance3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
export const CAT_RADIUS = .22;
/** The cat's indoor walks keep to this ring, inside the window seats and clear of the gallery traffic. */
const WINDOW_RING = 24.45;

/**
 * The cat's small world on the ground level is two window seats indoors, the porch of the first garden
 * entrance and the edge of the water court. Indoors it walks the ring by the windows. Outside it uses
 * the garden's own routes, so it never cuts through walls, water or steps.
 */
export function createCatPlan(plan = GARDEN_PLAN) {
  const spots = [42.5, 162.5].map((degrees, index) => {
    const angle = degrees * Math.PI / 180, position = polar(29.2, angle, floorY(0));
    return { id: `cat-window-${index}`, level: 0, angle, position, outdoor: false, approach: [polar(WINDOW_RING, angle, floorY(0)), position],
      facing: Math.PI - angle, support: { type: 'floor', level: 0 }, label: words.cat.window };
  });
  const entrance = plan.entrances[0], doorNode = plan.nodes.find(node => node.id === entrance.doorBottomNode);
  const porch = gardenPolar(35.65, entrance.angle, -.67, .98), poolNode = plan.nodes.find(node => node.id === 'court-pool-0');
  const water = gardenPolar(11.4, entrance.angle + 7 * Math.PI / 180, plan.dimensions.courtSurfaceY);
  const outdoorSpots = [
    { id: 'cat-porch', level: 0, nodeId: doorNode.id, zone: 'outer', outdoor: true, position: porch,
      approach: [{ ...doorNode.position }, gardenPolar(35.65, entrance.angle, -.67), porch], angle: entrance.angle,
      facing: Math.PI - entrance.angle, label: words.cat.porch },
    { id: 'cat-water', level: 0, nodeId: poolNode.id, zone: 'court', outdoor: true, position: water,
      approach: [{ ...poolNode.position }, water], angle: angleOf(water), facing: -angleOf(water), label: words.cat.water }];
  const garden = createOutdoorRouting({ ...plan, destinations: outdoorSpots });
  spots.push(...garden.destinations.map(spot => ({ ...spot, support: { type: 'garden', level: 0, zone: spot.zone, surfaceY: spot.position.y - .025 } })));
  const byId = new Map(spots.map(spot => [spot.id, spot]));
  function indoorRoute(from, spot) {
    const path = [], angle = angleOf(from), target = spot.angle;
    let p = appendLine(path, from, polar(WINDOW_RING, angle, floorY(0)), { type: 'floor', level: 0 }, .7);
    const turn = shortest(angle, target);
    if (Math.abs(turn) > 1e-7) path.push(arc(WINDOW_RING, angle, turn, 0, { type: 'floor', level: 0 }, .82));
    p = polar(WINDOW_RING, target, floorY(0));
    appendLine(path, p, spot.position, { type: 'floor', level: 0 }, .62);
    return path;
  }
  function route(from, origin, spot) {
    if (!origin.outdoor && !spot.outdoor) return indoorRoute(from, spot);
    if (origin.outdoor && spot.outdoor) {
      if (origin.zone === spot.zone) return garden.between(from, origin, spot).path;
      // From the porch to the water court, or back, the cat comes in to the first window seat and goes out again.
      const inside = spots[0];
      return [...garden.toGallery(from, origin, inside).path, ...route(inside.position, inside, spot)];
    }
    if (origin.outdoor) return garden.toGallery(from, origin, spot).path;
    const path = [], ring = polar(WINDOW_RING, angleOf(from), floorY(0));
    appendLine(path, from, ring, { type: 'floor', level: 0 }, .65);
    return [...path, ...garden.fromGallery(ring, 0, spot).path];
  }
  /** The height under a paw, on the actual tread wherever the point lies on an entrance's steps. */
  function sampleGround(x, z, support) {
    for (const gate of plan.entrances) {
      const radial = x * Math.sin(gate.angle) - z * Math.cos(gate.angle), across = x * Math.cos(gate.angle) + z * Math.sin(gate.angle);
      for (const steps of [gate.courtSteps, gate.doorSteps, gate.porchSteps]) {
        if (Math.abs(across) < (steps === gate.porchSteps ? 2.86 : 1.4) && radial >= steps.innerRadius - .36 && radial <= steps.outerRadius + .36)
          return gardenStepHeight(steps, x, z) + .025;
      }
    }
    return support?.steps ? gardenStepHeight(support.steps, x, z) + .025 : support?.type === 'garden' ? (support.surfaceY ?? 0) + .025 : floorY(support?.level || 0);
  }
  return { version: 1, spots, byId, route, sampleGround, signature: `cat-v1:${garden.signature}`, radius: CAT_RADIUS };
}

/** The part of a route already walked, reversed, so the cat can turn back the way it came. */
function walkedBack(path, index) {
  const back = [];
  for (let i = index; i >= 0; i--) {
    const segment = path[i], done = i === index ? segment.progress : segment.length;
    if (!(done > 1e-6)) continue;
    const reversed = clone(segment), share = done / Math.max(1e-8, segment.length);
    if (segment.kind === 'line') { reversed.from = pointAt(segment, done); reversed.to = pointAt(segment, 0); }
    else if (segment.kind === 'gardenStairs') {
      reversed.fromRadius = segment.fromRadius + (segment.toRadius - segment.fromRadius) * share; reversed.toRadius = segment.fromRadius;
    } else { reversed.fromAngle = segment.fromAngle + segment.delta * share; reversed.delta = -segment.delta * share; }
    reversed.length = done; reversed.progress = 0; back.push(reversed);
  }
  return back;
}

/**
 * Chestnut, the library cat, who is always there. It sleeps, wakes, stretches, walks to the next spot of a
 * fixed round, sits, grooms and curls up again, mostly in quiet places. Rain ends a stay outside and turns
 * a walk outdoors back the way it came. Someone standing in its way stops it, and a long wait turns it back.
 */
export function createCatLife({ plan = createCatPlan(), validateRoute = () => true, seed = 714219 } = {}) {
  let rng = seed >>> 0, time = 0, phase = 'sleep', phaseTime = 0, phaseDuration = 28;
  let currentId = plan.spots[0].id, destinationId = currentId, originId = currentId;
  let path = [], pathIndex = 0, tour = 0, returning = false, blockedFor = 0, disposed = false;
  let stats = { journeys: 0, completed: 0, rainReturns: 0, rejectedRoutes: 0, blockedReturns: 0, maxSpeed: 0, distance: 0 };
  const random = () => { rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0; return rng / 4294967296; };
  const start = plan.byId.get(currentId);
  const actor = { id: 'library-cat', name: words.cat.name, position: { ...start.position }, yaw: start.facing, speed: 0,
    support: clone(start.support), activity: start.label, active: true, viewHeight: .25, phase, phaseProgress: 0, stride: 0, level: 0 };
  function setPhase(next, duration) { phase = next; phaseTime = 0; phaseDuration = duration; actor.phase = next; }
  function turnBack(reason) {
    const back = walkedBack(path, pathIndex);
    if (!back.length) return false;
    path = back; pathIndex = 0; destinationId = originId; returning = true; blockedFor = 0;
    if (reason === 'rain') stats.rainReturns++;
    else stats.blockedReturns++;
    actor.activity = reason === 'rain' ? words.cat.rainHome : words.cat.backToWindow;
    return true;
  }
  /** After a stretch the cat walks to the next spot of its round. In rain or snow it stays indoors. */
  function setOff(environment) {
    const here = plan.byId.get(currentId), wet = (environment.rain || 0) > .16 || (environment.snow || 0) > .16;
    const round = ['cat-window-1', 'cat-water', 'cat-window-0', 'cat-porch', 'cat-window-0'];
    let next = here.outdoor ? plan.spots[0] : plan.byId.get(round[tour++ % round.length]);
    if (wet && next.outdoor || next.id === currentId) next = plan.spots[currentId === plan.spots[0].id ? 1 : 0];
    const route = plan.route(actor.position, here, next);
    if (!route.length || !validateRoute(route)) { stats.rejectedRoutes++; setPhase('sleep', 70); return false; }
    path = route; pathIndex = 0; originId = currentId; destinationId = next.id; returning = false; blockedFor = 0;
    actor.activity = next.outdoor ? words.cat.toCourt : words.cat.alongShelves;
    stats.journeys++; setPhase('walk', Infinity);
    return true;
  }
  function arrive() {
    const spot = plan.byId.get(destinationId);
    currentId = destinationId; path = []; pathIndex = 0; actor.speed = 0; actor.support = clone(spot.support); actor.activity = spot.label;
    stats.completed++; setPhase('sit', spot.outdoor ? 13 + random() * 9 : 5 + random() * 4);
  }
  function step(dt, environment, actors) {
    time += dt; phaseTime += dt;
    const wet = (environment.rain || 0) > .16 || (environment.snow || 0) > .16;
    if (phase === 'walk') {
      if (wet && !returning && plan.byId.get(destinationId).outdoor) turnBack('rain');
      let segment = path[pathIndex];
      if (!segment) { arrive(); return; }
      const ahead = pointAt(segment, Math.min(segment.length, segment.progress + .35));
      const dx = ahead.x - actor.position.x, dz = ahead.z - actor.position.z, heading = Math.atan2(dx, dz), turn = shortest(actor.yaw, heading);
      actor.yaw += clamp(turn, -dt * 2.1, dt * 2.1);
      const blocked = actors.some(other => other.active !== false && other.id !== actor.id && other.position &&
        Math.abs(other.position.y - ahead.y) < .8 && flatDistance(other.position, ahead) < .49);
      blockedFor = blocked ? blockedFor + dt : 0;
      if (blockedFor > 7 && !returning) { turnBack('blocked'); segment = path[pathIndex]; }
      // The cat slows down for sharp turns and stops for anyone standing just ahead.
      const facing = Math.abs(turn) > .95 ? 0 : Math.max(.22, 1 - Math.abs(turn) / 1.6);
      const wanted = blocked ? 0 : Math.min(.88, segment.speed || .72) * facing;
      actor.speed += clamp(wanted - actor.speed, -dt * 2.2, dt * 1);
      let travel = actor.speed * dt;
      while (travel > 1e-8 && segment) {
        const move = Math.min(travel, segment.length - segment.progress);
        segment.progress += move; travel -= move;
        const p = pointAt(segment);
        Object.assign(actor.position, p); actor.support = clone(segment.support); actor.stride += move; stats.distance += move;
        if (segment.progress >= segment.length - 1e-7) {
          pathIndex++; segment = path[pathIndex];
          if (!segment) { arrive(); break; }
        }
      }
      stats.maxSpeed = Math.max(stats.maxSpeed, actor.speed);
    } else {
      actor.speed = 0;
      const spot = plan.byId.get(currentId), turn = shortest(actor.yaw, spot.facing);
      actor.yaw += clamp(turn, -dt * .65, dt * .65);
      // Rain wakes a cat resting outside, and after waking it walks back in.
      if (spot.outdoor && wet && ['sleep', 'sit', 'groom', 'curl'].includes(phase)) { setPhase('wake', 1.6); returning = false; }
      if (phaseTime >= phaseDuration) {
        if (phase === 'sleep') setPhase('wake', 2);
        else if (phase === 'wake') setPhase('stretch', 3.3);
        else if (phase === 'stretch') setOff(environment);
        else if (phase === 'sit') setPhase(spot.outdoor ? 'wake' : 'groom', spot.outdoor ? 1.6 : 7.2);
        else if (phase === 'groom') setPhase('curl', 3.4);
        else if (phase === 'curl') setPhase('sleep', 100 + random() * 170);
      }
    }
    actor.phase = phase; actor.phaseProgress = Number.isFinite(phaseDuration) ? clamp(phaseTime / phaseDuration) : 0;
    actor.phaseTime = phaseTime; actor.time = time;
  }
  /** Advances in steps of at most 1/30 s and two seconds in all, so a long pause never turns into one jump. */
  function update(dt, { environment = {}, actors = [] } = {}) {
    if (disposed || !Number.isFinite(dt) || dt < 0) return;
    let left = Math.min(dt, 2);
    while (left > 1e-8) {
      const part = Math.min(left, 1 / 30);
      step(part, environment, actors); left -= part;
    }
  }
  function snapshot() {
    return { version: 1, signature: plan.signature, rng, time, phase, phaseTime, phaseDuration: Number.isFinite(phaseDuration) ? phaseDuration : null,
      currentId, destinationId, originId, path: clone(path), pathIndex, tour, rainReturn: returning, blockedFor, actor: clone(actor), stats: clone(stats) };
  }
  function restore(data) {
    if (!data || data.version !== 1 || data.signature !== plan.signature || !finitePoint(data.actor?.position) || !plan.byId.has(data.currentId) ||
      !plan.byId.has(data.originId) || !plan.byId.has(data.destinationId) ||
      !['sleep', 'wake', 'stretch', 'walk', 'sit', 'groom', 'curl'].includes(data.phase) ||
      ![data.time, data.phaseTime, data.rng, data.tour, data.actor.yaw, data.actor.stride].every(Number.isFinite) ||
      !Array.isArray(data.path) || data.path.length > 160) return false;
    const saved = clone(data.path);
    if (saved.some(s => !['line', 'arc', 'gardenArc', 'gardenStairs'].includes(s.kind) || ![s.length, s.progress].every(Number.isFinite) ||
      s.length <= 0 || s.progress < 0 || s.progress > s.length + 1e-6 || !finitePoint(pointAt(s, 0)) || !finitePoint(pointAt(s, s.length)) ||
      !['floor', 'garden'].includes(s.support?.type) || s.support?.level !== 0)) return false;
    if (data.phase === 'walk' && (!saved[data.pathIndex] || !validateRoute(saved) ||
      distance3(pointAt(saved[data.pathIndex]), data.actor.position) > .025)) return false;
    if (data.phase !== 'walk' && distance3(plan.byId.get(data.currentId).position, data.actor.position) > .025) return false;
    if (!Number.isInteger(data.pathIndex) || data.pathIndex < 0 || data.time < 0 || data.phaseTime < 0 || data.actor.stride < 0 ||
      !Number.isFinite(data.actor.speed) || data.actor.speed < 0 || data.actor.speed > 2) return false;
    rng = data.rng >>> 0; time = data.time; phase = data.phase; phaseTime = data.phaseTime;
    phaseDuration = data.phaseDuration === null ? Infinity : data.phaseDuration;
    if (!(phaseDuration > 0)) phaseDuration = 1;
    currentId = data.currentId; destinationId = data.destinationId; originId = data.originId; path = saved; pathIndex = data.pathIndex;
    tour = data.tour; returning = !!data.rainReturn; blockedFor = data.blockedFor || 0;
    Object.assign(actor, clone(data.actor), { id: 'library-cat', active: true });
    stats = { ...stats, ...data.stats };
    return true;
  }
  return { actor, plan, update, snapshot, restore,
    get stats() { return clone(stats); }, get route() { return clone(path); },
    dispose() { disposed = true; } };
}
