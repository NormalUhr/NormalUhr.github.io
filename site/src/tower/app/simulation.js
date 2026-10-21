import { BRIDGES, BridgeMechanism, FLOOR_HEIGHT, TAU, clamp, lerp, polar, shortest, smooth, stairProfile } from './mechanism.js';
import { FLOOR_OFFSET, CORE_WALK_RADIUS, STAIR_ANGLE, angleOf, appendLine, copyPoint, coreRoute, distance,
  fallbackDestinations, floorY, innerRadius, outerRoute, segmentPoint as floorSegmentPoint, walkRadius } from './navigation.js';
import { createBookTasks } from './book_tasks.js';
import { createOutdoorRouting, createOutdoorVisits, outdoorSegmentPoint, sampleGardenGround } from './outdoor_routes.js';
import { createPopulationSchedule, populationPortals, populationTarget } from './population.js';
import { createLifeTasks } from './life_tasks.js';
import { words } from './words.js';

const IDS = ['hour', 'minute'];
const NAMES = words.names;
const ROLES = [
  { role: words.roles.librarian, tag: 'librarian', carry: 'book' }, { role: words.roles.student, tag: 'student', carry: 'book' },
  { role: words.roles.clockmaker, tag: 'clockmaker', carry: 'tools' }, { role: words.roles.researcher, tag: 'scholar', carry: 'book' },
  { role: words.roles.visitor, tag: 'visitor', carry: null }, { role: words.roles.teacher, tag: 'teacher', carry: 'book' },
  { role: words.roles.courier, tag: 'visitor', carry: 'parcel' }, { role: words.roles.retired, tag: 'elder', carry: null },
];
const PURPOSES = words.purposes;
const EPS = .025;
const segmentPoint = (segment, at) => outdoorSegmentPoint(segment, at) || floorSegmentPoint(segment, at);

/**
 * Deterministic world-space residents. No DOM, WebGL, hidden teleport recovery,
 * nav-mesh jumps, or per-frame destination selection is used here.
 *
 * update owns BridgeMechanism.update. Positions are feet / tread surfaces.
 * Elevator y is the top of the cabin's walkable floor, including FLOOR_OFFSET.
 */
export function createSimulation({ mechanisms = null, count = 48, seed = 20261020,
  destinations = null, bookReturns = [], gardenPlan = null, population = null, living = null, initialAngles = { hour: 0, minute: 0 } } = {}) {
  mechanisms ||= Object.fromEntries(IDS.map(id => [id, new BridgeMechanism(BRIDGES[id])]));
  let randomState = seed >>> 0, time = 0, firstUpdate = true;
  const random = () => { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 4294967296; };
  let angles = { ...initialAngles }, allowBridgeEntry = true, gateFactor = 1;
  const profiles = Object.fromEntries(IDS.map(id => [id, stairProfile(BRIDGES[id])]));
  const sites = (destinations?.length ? destinations : fallbackDestinations()).map(d => ({ ...d,
    position: { ...d.position, y: floorY(d.level) },
    approach: (d.approach?.length ? d.approach : [polar(walkRadius(d.level), d.angle, floorY(d.level)), d.position])
      .map(p => ({ ...p, y: floorY(d.level) })) }));
  const siteById = new Map(sites.map(d => [d.id, d])), reserved = new Map();
  const outdoorRouting = gardenPlan ? createOutdoorRouting(gardenPlan) : null;
  for (const destination of outdoorRouting?.destinations || []) siteById.set(destination.id, destination);
  const nightCount = Math.max(1, Math.floor(count));
  const rosterCount = population ? nightCount * 3 : Math.min(count, sites.length);
  if (population && (!outdoorRouting || sites.length < rosterCount + 8))
    throw new Error('Dynamic attendance needs garden entry routes and enough separate indoor destinations');
  const portals = population ? populationPortals(population.portals) : [];
  for (const portal of portals) siteById.set(portal.id, portal);
  const initialCount = population ? populationTarget(population.minutes, nightCount) : rosterCount;
  const actors = [], actorById = new Map();
  let bookTasks = null, outdoorVisits = null, attendance = null, lifeTasks = null;
  const manual = { hour: null, minute: null }, seam = { hour: { inner: null, outer: null }, minute: { inner: null, outer: null } };
  let automaticAt = 52, automaticBridge = 0, trafficAt = 0;
  const stats = {
    elapsed: 0, completedJourneys: 0, fixedStairFlights: 0, bridgeEntries: 0, bridgeExits: 0,
    elevatorTrips: 0, bridgeLevelChanges: 0, centerJourneys: 0, entryFallbacks: 0,
    longestWait: 0, longestPauseDuringTravel: 0, longestLiftWait: 0, longestJourney: 0, maxElevatorQueue: 0, maxBridgeOccupants: 0,
    seamCrossings: 0, actors: initialCount, recent: [],
  };
  const elevators = [-1.02, 1.02].map((x, id) => ({ id, x, z: 0, y: floorY(id ? 4 : 1), level: id ? 4 : 1,
    targetLevel: id ? 4 : 1, door: 1, phase: 'idle', passengers: [], queue: [],
    _request: null, _time: 0, _fromY: 0, _duration: 0, _trip: null, _cooldown: 0 }));

  function reserve(site, id) { reserved.set(site.id, id); }
  function unreserve(site, id) { if (site && reserved.get(site.id) === id) reserved.delete(site.id); }
  function turn(a, target, dt) {
    const delta = shortest(a.yaw, target);
    a.yaw += clamp(delta, -2.8 * dt, 2.8 * dt);
    a._desiredYaw = target;
  }
  function walkSpeed(a, dt, factor = 1, desiredYaw = a.yaw) {
    turn(a, desiredYaw, dt);
    const facing = .20 + .80 * Math.max(0, Math.cos(shortest(a.yaw, desiredYaw)));
    const target = a._pace * factor * (a._headway ?? 1) * facing;
    a._velocity = lerp(a._velocity, target, Math.min(1, dt * 5));
    return a._velocity;
  }
  function still(a, dt, activity = 'wait') {
    a.speed = 0; a._velocity *= Math.max(0, 1 - dt * 7); a.activity = activity;
  }
  function place(a, p, walked, dt) {
    a.position.x = p.x; a.position.y = p.y; a.position.z = p.z;
    a.speed = dt > 0 ? walked / dt : 0;
    a.distanceTravelled += walked;
    a.gaitPhase += walked * 6.8;
    if (walked > .0001) a.activity = 'walk';
  }
  function setState(a, state) { a._state = state; a._stateAt = time; a._waitingSince = null; }
  function waiting(a, dt, activity = 'wait') {
    if (a._waitingSince === null) a._waitingSince = time;
    stats.longestWait = Math.max(stats.longestWait, time - a._waitingSince);
    still(a, dt, activity);
  }
  function setPath(a, path, after) {
    a._path = path.filter(s => s.length > 1e-6); a._pathIndex = 0; a._after = after;
    setState(a, 'path');
    if (!a._path.length) pathDone(a, after);
  }
  function pathTail(path, fallback) { return path.length ? segmentPoint(path[path.length - 1], path[path.length - 1].length) : fallback; }
  function approachGoal(a) {
    const path = [], d = a.destination;
    let p = copyPoint(a.position);
    for (const next of d.approach) p = appendLine(path, p, next, { type: 'floor', level: d.level }, .83);
    appendLine(path, p, d.position, { type: 'floor', level: d.level }, .72);
    setPath(a, path, 'arrive');
  }
  function journeyFinish(a) {
    if (a._journeyStarted != null) {
      stats.completedJourneys++; a.completedJourneys++;
      stats.longestJourney = Math.max(stats.longestJourney, time - a._journeyStarted);
      if (a._viaCenter) stats.centerJourneys++;
      stats.recent.push({ id: a.id, destination: a.destination.id, at: Math.round(time), viaCenter: a._viaCenter });
      if (stats.recent.length > 24) stats.recent.shift();
    }
    a.level = a.destination.level; a.support = { type: 'floor', level: a.level };
    a._atDestination = a.destination; a.purpose = PURPOSES.stay(a.destination.activity, a.destination.label);
    a._dwell = 20 + random() * 70; a.activity = a.destination.activity || 'observe';
    a._journeyStarted = null; a._viaCenter = false; a._exitLease = null;
    setState(a, 'idle'); a.speed = 0;
  }
  function pathDone(a, after) {
    a._path = []; a._pathIndex = 0;
    if (lifeTasks?.pathDone(a, after)) return;
    if (attendance?.pathDone(a, after)) return;
    if (bookTasks?.pathDone(a, after)) return;
    if (outdoorVisits?.pathDone(a, after)) return;
    if (after === 'arrive') journeyFinish(a);
    else if (after === 'goal_approach') approachGoal(a);
    else if (after === 'entry_seek') { setState(a, 'entry_seek'); a._entryWaitAt = time; a._entryStage = 'ring'; }
    else if (after === 'lift_wait') setState(a, 'lift_wait');
    else if (after === 'lift_front') setState(a, 'lift_front');
    else if (after === 'exit_seek') { setState(a, 'exit_seek'); a._entryStage = 'ring'; }
    else if (after === 'plan') planJourney(a);
  }
  function movePath(a, dt) {
    const s = a._path[a._pathIndex];
    if (!s) { pathDone(a, a._after); return; }
    const at = segmentPoint(s), ahead = segmentPoint(s, Math.min(s.length, s.progress + .12));
    const yaw = distance(at, ahead) > .00001 ? Math.atan2(ahead.x - at.x, ahead.z - at.z) : a.yaw;
    const factor = (s.kind === 'stairs' || s.kind === 'gardenStairs') && a.age !== 'adult' ? Math.min(s.speed, .62 / a._pace) : s.speed;
    const ds = Math.min(s.length - s.progress, walkSpeed(a, dt, factor, yaw) * dt);
    s.progress += ds; a.support = { ...s.support };
    if (s.kind === 'stairs') {
      a.level = clamp(Math.round(segmentPoint(s).y / 6), 0, 6);
    } else a.level = s.support.level ?? a.level;
    place(a, segmentPoint(s), ds, dt);
    if (s.progress >= s.length - 1e-8) {
      if (s.kind === 'stairs') stats.fixedStairFlights++;
      a._pathIndex++;
      if (a._pathIndex >= a._path.length) pathDone(a, a._after);
    }
  }
  function chooseDestination(a, forced = false) {
    const candidates = sites.filter(d => !reserved.has(d.id) && d.id !== a._atDestination?.id);
    if (!candidates.length) return null;
    let best = null, highest = -Infinity;
    for (const d of candidates) {
      let score = random();
      const delta = Math.abs(d.level - a.level);
      if (d.roleTags?.includes(a.roleTag)) score += .35;
      if (living && lifeTasks) {
        // Night favours quiet reading seats, and a dry bright day favours the windows.
        const weather = lifeTasks.weather;
        if (weather.night > .7 && ['read', 'sit'].includes(d.activity)) score += .1;
        if (weather.day > .65 && !weather.wet && d.activity === 'observe') score += .06;
      }
      if (forced) score += (a.level < 4 ? d.level >= 4 : d.level <= 3) ? 4 : 0;
      else { score += delta <= 2 ? .27 : 0; if (delta === 0) score += .08; }
      if (score > highest) { highest = score; best = d; }
    }
    return best;
  }
  function leave(a, forceCenter = false) {
    const next = chooseDestination(a, forceCenter);
    if (!next) { a._dwell = 5; return; }
    const old = a._atDestination;
    // The old place stays reserved until its previous occupant has actually left
    // the approach. This also prevents head-on movement in a narrow furniture bay.
    a._departingDestination = old; reserve(next, a.id); a.destination = next;
    a._journeyStarted = time; a._forcedCenter = forceCenter;
    a.activity = 'walk';
    a.purpose = PURPOSES.toward(next.label, next.activity);
    a._atDestination = null;
    const path = [];
    let p = copyPoint(a.position);
    if (old) for (const target of [...old.approach].reverse()) p = appendLine(path, p, target, { type: 'floor', level: a.level }, .85);
    setPath(a, path, 'plan');
  }
  function planJourney(a) {
    unreserve(a._departingDestination, a.id); a._departingDestination = null;
    const activeCenter = actors.filter(p => p._viaCenter && p._state !== 'idle').length;
    const center = (a._forcedCenter || (Math.abs(a.level - a.destination.level) >= 2 && random() < .26)) && activeCenter < 8;
    a._forcedCenter = false; a._viaCenter = center;
    if (center) {
      const id = a.level <= 3 ? 'hour' : 'minute';
      const m = mechanisms[id]; a._entryBridge = id;
      a._entryLevel = m.phase === 'open' ? m.level : m.targetLevel;
      a.purpose = PURPOSES.viaCore(BRIDGES[id].name, a.destination.label);
      setPath(a, outerRoute(a.position, a.level, a._entryLevel, angles[id]), 'entry_seek');
    } else setPath(a, outerRoute(a.position, a.level, a.destination.level, a.destination.angle), 'goal_approach');
  }
  function finishOuter(a) {
    a._entryBridge = null;
    setPath(a, outerRoute(a.position, a.level, a.destination.level, a.destination.angle), 'goal_approach');
  }
  function fallbackEntry(a) {
    a._entryBridge = null; a._viaCenter = false; stats.entryFallbacks++;
    a.purpose = PURPOSES.byStairs(a.destination.label);
    finishOuter(a);
  }
  function bridgePoint(id, r, lane) {
    const angle = angles[id];
    return { x: Math.sin(angle) * r + Math.cos(angle) * lane,
      y: mechanisms[id].y + profiles[id].height(r) + FLOOR_OFFSET,
      z: -Math.cos(angle) * r + Math.sin(angle) * lane };
  }
  function pointToBridge(id, p) {
    const angle = angles[id];
    return { r: Math.sin(angle) * p.x - Math.cos(angle) * p.z,
      lane: Math.cos(angle) * p.x + Math.sin(angle) * p.z };
  }
  function bridgeReady(id) {
    return allowBridgeEntry && gateFactor > .98 && mechanisms[id].phase === 'open' && mechanisms[id].open > .98;
  }
  function canClose(id) { return !seam[id].inner && !seam[id].outer; }
  function startSeam(a, id, end, direction) {
    if (!bridgeReady(id) || seam[id][end]) return false;
    const local = pointToBridge(id, a.position);
    a._bridge = id; a._r = local.r; a._lane = local.lane;
    a._direction = direction; a._seamEnd = end;
    a._seamTarget = end === 'outer' ? (direction < 0 ? BRIDGES[id].radius - 1.35 : BRIDGES[id].radius + .70)
      : (direction < 0 ? 4.0 : 5.75);
    seam[id][end] = a.id;
    setState(a, 'seam'); stats.seamCrossings++;
    return true;
  }
  function moveOnRing(a, dt, id, radius, lane, factor = 1) {
    const target = bridgePoint(id, radius, lane), targetAngle = angleOf(target);
    const currentAngle = angleOf(a.position), actualR = Math.hypot(a.position.x, a.position.z);
    const delta = shortest(currentAngle, targetAngle);
    if (Math.abs(delta) * radius > .03) {
      const sign = Math.sign(delta);
      const tangentYaw = Math.atan2(sign * Math.cos(currentAngle), sign * Math.sin(currentAngle));
      const ds = Math.min(Math.abs(delta) * actualR, walkSpeed(a, dt, factor, tangentYaw) * dt);
      const p = polar(actualR, currentAngle + sign * ds / actualR, floorY(a.level));
      place(a, p, ds, dt); return false;
    }
    const goal = { ...target, y: floorY(a.level) };
    return moveDirect(a, goal, dt, factor);
  }
  function moveDirect(a, target, dt, factor = 1) {
    const gap = distance(a.position, target);
    if (gap < 1e-7) { still(a, dt); return true; }
    const yaw = Math.atan2(target.x - a.position.x, target.z - a.position.z);
    const ds = Math.min(gap, walkSpeed(a, dt, factor, yaw) * dt), t = ds / gap;
    place(a, { x: lerp(a.position.x, target.x, t), y: lerp(a.position.y, target.y, t), z: lerp(a.position.z, target.z, t) }, ds, dt);
    return ds >= gap - 1e-7;
  }
  function entrySeek(a, dt) {
    const id = a._entryBridge, m = mechanisms[id], config = BRIDGES[id];
    a.support = { type: 'floor', level: a.level };
    if (time - a._entryWaitAt > 52) { fallbackEntry(a); return; }
    if (a._entryStage === 'ring') {
      if (moveOnRing(a, dt, id, walkRadius(a.level), -.34)) a._entryStage = 'port';
      return;
    }
    // A moving hand can be followed safely around the fixed ring while closed.
    // Never cut a chord through the central void to chase a fast time adjustment.
    const target = bridgePoint(id, config.radius + .70, -.34);
    const angleGap = Math.abs(shortest(angleOf(a.position), angleOf(target)));
    if (angleGap > .065) { a._entryStage = 'ring'; return; }
    if (moveDirect(a, { ...target, y: floorY(a.level) }, dt, .78)) {
      if (m.level === a.level && bridgeReady(id) && !seam[id].outer) startSeam(a, id, 'outer', -1);
      else waiting(a, dt);
    }
  }
  function updateSeam(a, dt) {
    const id = a._bridge, m = mechanisms[id], config = BRIDGES[id];
    const remaining = Math.abs(a._seamTarget - a._r);
    const desiredYaw = (a._direction > 0 ? Math.PI : 0) - angles[id];
    const ds = Math.min(remaining, walkSpeed(a, dt, .81, desiredYaw) * dt);
    a._r += a._direction * ds;
    const p = bridgePoint(id, a._r, a._lane);
    const isOuter = a._seamEnd === 'outer';
    const onFixed = isOuter ? a._r > config.radius + .28 : a._r < 4.27;
    a.support = onFixed ? { type: isOuter ? 'floor' : 'core', level: isOuter ? m.level : m.coreLevel }
      : { type: 'bridge', id, r: a._r, lane: a._lane };
    if (onFixed) p.y = floorY(isOuter ? m.level : m.coreLevel);
    a.level = onFixed ? a.support.level : clamp(Math.round(p.y / 6), 0, 6);
    place(a, p, ds, dt);
    if (remaining - ds <= 1e-7) {
      seam[id][a._seamEnd] = null;
      if (a._seamEnd === 'outer' && a._direction < 0) {
        stats.bridgeEntries++; setState(a, 'bridge_walk'); a._entryBridge = null;
      } else if (a._seamEnd === 'inner' && a._direction > 0) {
        stats.bridgeEntries++; a._exitLease = null; setState(a, 'bridge_walk');
      } else if (a._seamEnd === 'inner') {
        stats.bridgeExits++; a.level = m.coreLevel; a.support = { type: 'core', level: a.level };
        a._bridge = null; enterCore(a);
      } else {
        stats.bridgeExits++; a.level = m.level; a.support = { type: 'floor', level: a.level };
        a._bridge = null; finishOuter(a);
      }
    }
  }
  function updateBridgeWalk(a, dt) {
    if (lifeTasks?.tryStartBridge(a)) { lifeTasks.tickActor(a, dt); return; }
    const id = a._bridge, m = mechanisms[id], config = BRIDGES[id];
    const target = a._direction < 0 ? 5.75 : config.radius - 1.35;
    const remaining = Math.abs(target - a._r);
    if (remaining > 1e-7) {
      const onStairs = a._r > profiles[id].start && a._r < profiles[id].end;
      const speedFactor = onStairs ? (a.age === 'adult' ? .77 : .62 / a._pace) : .92;
      const ds = Math.min(remaining, walkSpeed(a, dt, speedFactor, (a._direction > 0 ? Math.PI : 0) - angles[id]) * dt);
      a._r += a._direction * ds;
      place(a, bridgePoint(id, a._r, a._lane), ds, dt);
    } else {
      place(a, bridgePoint(id, a._r, a._lane), 0, dt);
      waiting(a, dt, m.phase === 'open' ? 'observe' : 'ride');
      startSeam(a, id, a._direction < 0 ? 'inner' : 'outer', a._direction);
    }
    a.support = { type: 'bridge', id, r: a._r, lane: a._lane };
    a.level = clamp(Math.round(a.position.y / 6), 0, 6);
  }
  function enterCore(a) {
    const id = a.destination.level < 4 ? 'hour' : 'minute', m = mechanisms[id];
    const outerLevel = m.phase === 'open' ? m.level : m.targetLevel;
    a._exitBridge = id; a._exitLease = { id, level: outerLevel, at: time };
    const targetLevel = outerLevel - 1;
    if (a.level === targetLevel) headForExit(a);
    else callLift(a, targetLevel);
  }
  function headForExit(a) {
    const id = a._exitBridge;
    const target = bridgePoint(id, 4.0, .34);
    setPath(a, coreRoute(a.position, a.level, { ...target, y: floorY(a.level) }), 'exit_seek');
  }
  function exitSeek(a, dt) {
    const id = a._exitBridge, m = mechanisms[id];
    a.support = { type: 'core', level: a.level };
    if (m.phase === 'open' && m.coreLevel !== a.level) {
      // This is only needed after restoring an older save or an external mechanism request.
      a._exitLease = { id, level: m.level, at: time };
      callLift(a, m.coreLevel); return;
    }
    if (moveOnRing(a, dt, id, 4.0, .34, .80)) {
      if (bridgeReady(id) && m.coreLevel === a.level && !seam[id].inner) startSeam(a, id, 'inner', 1);
      else waiting(a, dt);
    }
  }
  function chooseElevator(a, targetLevel) {
    let best = elevators[0], cost = Infinity;
    for (const car of elevators) {
      const c = car.queue.length * 30 + (car._request ? 26 : 0) + Math.abs(car.level - a.level) * 2.3;
      if (c < cost) { cost = c; best = car; }
    }
    return best;
  }
  function callLift(a, targetLevel) {
    if (targetLevel === a.level) { headForExit(a); return; }
    const car = chooseElevator(a, targetLevel);
    const occupied = new Set([...car.queue, ...(car._request ? [car._request] : [])]
      .filter(q => q.pickup === a.level).map(q => q.slot));
    let slot = 0; while (occupied.has(slot)) slot++;
    const request = { actorId: a.id, pickup: a.level, dropoff: targetLevel, created: time, slot };
    car.queue.push(request); a._liftId = car.id; a._liftTarget = targetLevel;
    const queueAngle = (car.id ? 1 : -1) * (.45 + slot * .28);
    a._queuePoint = polar(CORE_WALK_RADIUS, queueAngle, floorY(a.level));
    a.purpose = PURPOSES.byLift(targetLevel, a.destination.label);
    stats.maxElevatorQueue = Math.max(stats.maxElevatorQueue, car.queue.length + (car._request ? 1 : 0));
    setPath(a, coreRoute(a.position, a.level, a._queuePoint), 'lift_wait');
  }
  function elevatorFront(a) {
    const car = elevators[a._liftId];
    setPath(a, coreRoute(a.position, a.level, { x: car.x, y: floorY(a.level), z: -2.65 }), 'lift_front');
  }
  function liftClosing(car, target, trip) {
    car.targetLevel = target; car._trip = trip; car.phase = 'closing'; car._time = 0;
  }
  function updateCars(dt) {
    for (const car of elevators) {
      car._time += dt;
      if (car.phase === 'idle') {
        car._cooldown = Math.max(0, car._cooldown - dt);
        if (car._cooldown > 0 || !car.queue.length) continue;
        car._request = car.queue.shift();
        if (car.level === car._request.pickup) { car.phase = 'boarding'; car.door = 1; }
        else liftClosing(car, car._request.pickup, 'pickup');
      } else if (car.phase === 'closing') {
        car.door = 1 - smooth(car._time / 1.15);
        if (car._time >= 1.15) {
          car.door = 0; car.phase = 'moving'; car._time = 0; car._fromY = car.y;
          car._duration = Math.max(2.5, Math.abs(floorY(car.targetLevel) - car.y) / 2.7);
        }
      } else if (car.phase === 'moving') {
        car.y = lerp(car._fromY, floorY(car.targetLevel), smooth(car._time / car._duration));
        if (car._time >= car._duration) {
          car.y = floorY(car.targetLevel); car.level = car.targetLevel; car.phase = 'opening'; car._time = 0;
        }
      } else if (car.phase === 'opening') {
        car.door = smooth(car._time / 1.15);
        if (car._time >= 1.15) { car.door = 1; car.phase = car._trip === 'pickup' ? 'boarding' : 'alighting'; car._time = 0; }
      }
    }
  }
  function updateLiftActor(a, dt) {
    const car = elevators[a._liftId];
    if (a._state === 'lift_wait') {
      waiting(a, dt);
      if (car._request?.actorId === a.id) elevatorFront(a);
    } else if (a._state === 'lift_front') {
      waiting(a, dt);
      if (car._request?.actorId === a.id && car.phase === 'boarding' && car.door > .98 && car.level === a.level) setState(a, 'lift_board');
    } else if (a._state === 'lift_board') {
      const done = moveDirect(a, { x: car.x, y: car.y, z: -.10 }, dt, .64);
      if (a.position.z >= -.80) {
        a.support = { type: 'lift', id: car.id };
        if (!car.passengers.length) car.passengers.push(a.id);
      } else a.support = { type: 'core', level: a.level, landing: car.id };
      if (done) {
        stats.longestLiftWait = Math.max(stats.longestLiftWait, time - car._request.created);
        setState(a, 'lift_ride'); car.passengers = [a.id];
        liftClosing(car, a._liftTarget, 'delivery');
      }
    } else if (a._state === 'lift_ride') {
      a.position.x = car.x; a.position.z = -.10; a.position.y = car.y;
      a.support = { type: 'lift', id: car.id }; a.level = clamp(Math.round(car.y / 6), 0, 6);
      still(a, dt, 'ride');
      if (car.phase === 'alighting' && car.door > .98) { a.level = car.level; setState(a, 'lift_exit'); }
    } else if (a._state === 'lift_exit') {
      const done = moveDirect(a, { x: car.x, y: car.y, z: -2.65 }, dt, .70);
      a.support = a.position.z >= -.80 ? { type: 'lift', id: car.id } : { type: 'core', level: car.level, landing: car.id };
      if (done) {
        stats.elevatorTrips++; a.level = car.level; a.support = { type: 'core', level: a.level };
        car.passengers = []; car._request = null; car.phase = 'idle'; car._time = 0; car._cooldown = 1.5;
        a._liftId = null; headForExit(a);
      }
    }
  }
  function leasesFor(id) {
    return actors.filter(a => a._exitLease?.id === id);
  }
  function bridgeRequestPossible(id, level) {
    const m = mechanisms[id];
    if (m.phase !== 'open' || !canClose(id) || !allowBridgeEntry || gateFactor <= .98) return false;
    if (IDS.some(other => other !== id && mechanisms[other].phase !== 'open')) return false;
    return !leasesFor(id).some(a => a._exitLease.level !== level);
  }
  function scheduleBridges() {
    // Manual floor choices drain live seams first; callers need not micromanage locks.
    for (const id of IDS) if (manual[id] != null) {
      const level = manual[id];
      if (level === mechanisms[id].level && mechanisms[id].phase === 'open') manual[id] = null;
      else if (bridgeRequestPossible(id, level)) {
        mechanisms[id].request(level); manual[id] = null; stats.bridgeLevelChanges++;
        automaticAt = time + 45 + random() * 30; return;
      }
    }
    // A visitor who has reached an actual gallery may call the bridge to that floor.
    // Waiting expires, so a moving bridge never makes an outer-gallery journey infinite.
    for (const id of IDS) {
      const waiters = actors.filter(a => a._state === 'entry_seek' && a._entryBridge === id)
        .sort((a, b) => a._entryWaitAt - b._entryWaitAt || a.index - b.index);
      if (waiters.length) {
        const current = waiters.find(a => a.level === mechanisms[id].level);
        if (current && mechanisms[id].phase === 'open') continue;
        const level = waiters[0].level;
        if (mechanisms[id].level !== level && bridgeRequestPossible(id, level)) {
          mechanisms[id].request(level); stats.bridgeLevelChanges++;
          automaticAt = Math.max(automaticAt, time + 45); return;
        }
      }
    }
    if (time < automaticAt) return;
    const id = IDS[automaticBridge], m = mechanisms[id];
    const hasBoarders = actors.some(a => a._state === 'entry_seek' && a._entryBridge === id && a.level === m.level);
    if (leasesFor(id).length || hasBoarders) { automaticAt = time + 3; return; }
    const levels = BRIDGES[id].levels, index = levels.indexOf(m.level);
    const target = levels[(index + 1) % levels.length];
    if (bridgeRequestPossible(id, target)) {
      m.request(target); stats.bridgeLevelChanges++; automaticBridge = 1 - automaticBridge;
      automaticAt = time + 45 + random() * 30;
    }
  }
  function requestBridgeLevel(id, level) {
    if (!BRIDGES[id] || !BRIDGES[id].levels.includes(level)) return false;
    manual[id] = level; return true;
  }
  function traffic() {
    const present = actors.filter(a => a.active !== false);
    for (const a of present) a._headway = 1;
    for (const a of present) {
      if (a._state === 'idle' || a._state === 'outdoor_dwell' || a.support.type === 'lift') continue;
      const sin = Math.sin(a._desiredYaw), cos = Math.cos(a._desiredYaw);
      for (const b of present) {
        if (a === b || Math.abs(a.position.y - b.position.y) > .7) continue;
        if (a.support.type !== b.support.type) continue;
        if (a.support.type === 'bridge' && a.support.id !== b.support.id) continue;
        const dx = b.position.x - a.position.x, dz = b.position.z - a.position.z;
        // Headway is following, never a symmetric obstacle-repulsion rule. The
        // opposite direction already has another physical lane; crossing flows
        // must not both brake forever while facing one another.
        if (Math.cos(shortest(a._desiredYaw, b._desiredYaw)) < .65) continue;
        const ahead = dx * sin + dz * cos;
        if (ahead <= .08 || ahead > 1.5 || Math.abs(dx * cos - dz * sin) > .40) continue;
        // A boarding cabin and its front approach are managed by a FIFO lock.
        if (a._state.startsWith('lift_') || b._state.startsWith('lift_')) continue;
        a._headway = Math.min(a._headway, clamp((ahead - .52) / .9));
      }
    }
  }
  function sampleGround(x, z, a) {
    const support = a.support;
    if (support.type === 'garden') return sampleGardenGround(x, z, support);
    if (support.type === 'bridge') {
      const id = support.id, local = pointToBridge(id, { x, z });
      return mechanisms[id].y + profiles[id].height(local.r) + FLOOR_OFFSET;
    }
    if (support.type === 'stairs') {
      const fraction = clamp(shortest(support.baseAngle, Math.atan2(x, -z)) / STAIR_ANGLE);
      return floorY(support.lowerLevel) + Math.min(30, Math.ceil(fraction * 30 - 1e-8)) * .2;
    }
    if (support.type === 'lift') return elevators[support.id].y;
    return floorY(support.level ?? a.level);
  }
  function feet(a) {
    const sideX = Math.cos(a.yaw) * .12, sideZ = -Math.sin(a.yaw) * .12;
    const step = a.speed > .01 ? Math.cos(a.gaitPhase) * .18 : 0;
    const forwardX = Math.sin(a.yaw) * step, forwardZ = Math.cos(a.yaw) * step;
    a.footHeights[0] = sampleGround(a.position.x - sideX + forwardX, a.position.z - sideZ + forwardZ, a);
    a.footHeights[1] = sampleGround(a.position.x + sideX - forwardX, a.position.z + sideZ - forwardZ, a);
  }
  function actorTick(a, dt) {
    if (a.active === false) return;
    const beforeDistance = a.distanceTravelled;
    if (lifeTasks?.tickActor(a, dt)) {
      // Everyday stories pose the resident only while standing. Their walks use movePath.
    } else if (attendance?.tickActor(a, dt)) {
      // A viewed gatehouse occupant remains present until it is occluded.
    } else if (bookTasks?.tickActor(a, dt)) {
      // The book task controls only the stationary notice/pick/shelve poses.
      // Its walking legs still use movePath and the ordinary traffic system.
    } else if (outdoorVisits?.tickActor(a, dt)) {
      // Outdoor dwell and route selection share the same walker and feet.
      // A garden visitor is never available for an indoor book interruption.
    } else if (a._state === 'idle') {
      still(a, dt, a.destination.activity || 'observe');
      turn(a, a.destination.facing ?? a.yaw, dt);
      a._dwell -= dt;
      if (a._dwell <= 0) leave(a, a._initialCenter);
      a._initialCenter = false;
    } else if (a._state === 'path') movePath(a, dt);
    else if (a._state === 'entry_seek') entrySeek(a, dt);
    else if (a._state === 'seam') updateSeam(a, dt);
    else if (a._state === 'bridge_walk') updateBridgeWalk(a, dt);
    else if (a._state === 'exit_seek') exitSeek(a, dt);
    else if (a._state.startsWith('lift_')) updateLiftActor(a, dt);
    if (a._state === 'idle' || a._state === 'outdoor_dwell' || a._state === 'outdoor_rain' || a._state?.startsWith('life_') || a._weatherWaiting ||
      a._state === 'population_wait' || a.active === false || a.distanceTravelled - beforeDistance > .0001) a._stalledFor = 0;
    else a._stalledFor = (a._stalledFor || 0) + dt;
    stats.longestPauseDuringTravel = Math.max(stats.longestPauseDuringTravel, a._stalledFor);
    feet(a);
  }
  function chooseInitialSite(index) {
    const level = index % 7;
    const candidates = sites.filter(d => d.level === level && !reserved.has(d.id));
    return candidates[Math.floor(random() * candidates.length)] || sites.find(d => !reserved.has(d.id));
  }
  for (let i = 0; i < rosterCount; i++) {
    const d = i < initialCount ? chooseInitialSite(i) : null, r = ROLES[i % ROLES.length];
    const portal = population ? portals[(i + portals.length * rosterCount - nightCount) % portals.length] : null;
    const position = copyPoint(d ? d.position : portal.position), level = d?.level ?? 0;
    const age = i % 11 === 5 ? 'child' : r.tag === 'elder' || i % 13 === 10 ? 'elder' : 'adult';
    const a = { id: `resident-${String(i + 1).padStart(2, '0')}`, index: i, name: NAMES[i % NAMES.length],
      role: age === 'child' ? words.roles.child : r.role, roleTag: age === 'child' ? 'student' : r.tag, age,
      appearanceSeed: (Math.imul(seed ^ (i + 1), 2654435761) >>> 0),
      position, yaw: d?.facing ?? 0, speed: 0, gaitPhase: random() * TAU,
      activity: d?.activity || 'away', carrying: r.carry && age !== 'child' ? { type: r.carry } : age === 'child' ? { type: 'book' } : null,
      level, support: d ? { type: 'floor', level } : { type: 'garden', level: 0, zone: 'outer', surfaceY: position.y - FLOOR_OFFSET },
      footHeights: [position.y, position.y],
      destination: d, purpose: d ? PURPOSES.stay(d.activity, d.label) : PURPOSES.away,
      distanceTravelled: 0, completedJourneys: 0,
      _state: d ? 'idle' : 'absent', _stateAt: 0, _waitingSince: null, _velocity: 0, _desiredYaw: d?.facing ?? 0, _headway: 1,
      _pace: (age === 'elder' ? .93 : age === 'child' ? 1.04 : 1.12) + random() * .20,
      _dwell: 3 + random() * 58, _atDestination: d, _journeyStarted: null, _viaCenter: false, _exitLease: null,
      _path: [], _pathIndex: 0, _bridge: null,
      ...(population ? { active: !!d, populationPhase: d ? 'present' : 'absent', _populationPortalId: d ? null : portal.id } : {}),
    };
    if (d) reserve(d, a.id); actors.push(a); actorById.set(a.id, a);
  }
  // A few residents start in the middle of legal journeys, like a building already
  // in use. After this initialization no position is respawned or reassigned.
  for (let i = 0; i < Math.min(6, initialCount); i++) {
    const a = actors[i], id = i % 2 ? 'minute' : 'hour', m = mechanisms[id];
    unreserve(a._atDestination, a.id);
    a.level = id === 'hour' ? 2 : 5;
    const dest = chooseDestination(a, true) || a.destination;
    reserve(dest, a.id); a.destination = dest; a._atDestination = null; a._journeyStarted = 0; a._viaCenter = true;
    a.purpose = PURPOSES.crossing(BRIDGES[id].name, dest.label);
    if (i < 2) {
      a._bridge = id; a._r = profiles[id].middle + 3; a._lane = -.34; a._direction = -1;
      a.position = bridgePoint(id, a._r, a._lane); a.yaw = -angles[id]; a._desiredYaw = a.yaw;
      a.support = { type: 'bridge', id, r: a._r, lane: a._lane }; setState(a, 'bridge_walk');
    } else if (i < 4) {
      a.level = m.coreLevel; a.position = polar(CORE_WALK_RADIUS, i === 2 ? 1.6 : 4.7, floorY(a.level));
      a.support = { type: 'core', level: a.level }; enterCore(a);
    } else {
      a.level = m.level; a._entryBridge = id; a._entryLevel = m.level;
      a.position = bridgePoint(id, BRIDGES[id].radius + .70, -.34); a.position.y = floorY(a.level);
      a.yaw = -angles[id]; a._desiredYaw = a.yaw; a.support = { type: 'floor', level: a.level };
      a._entryWaitAt = 0; a._entryStage = 'port'; setState(a, 'entry_seek');
    }
    feet(a);
  }

  attendance = population ? createPopulationSchedule({ actors, nightCount, minutes: population.minutes, portals,
    routing: outdoorRouting, getTime: () => time, reserved, chooseDestination,
    setPath, setState, finish: journeyFinish }) : null;
  bookTasks = createBookTasks({ actors, returns: bookReturns, getTime: () => time, setPath, setState, still, turn,
    isLifeFloorBusy: level => lifeTasks?.isPaperFloorBusy(level) || false });
  outdoorVisits = outdoorRouting ? createOutdoorVisits({ actors, routing: outdoorRouting, getTime: () => time,
    getSite: id => siteById.get(id), setPath, setState, still, turn, finish: journeyFinish, seed,
    ...(living ? { getWeather: () => lifeTasks?.weather || null } : {}),
    ...(population ? { getPopulationCount: () => actors.filter(a => a.active !== false).length } : {}) }) : null;
  lifeTasks = living ? createLifeTasks({ actors, plan: living, getTime: () => time, getAngles: () => angles, mechanisms,
    bridgePoint, walkSpeed, place, setState, setPath, still, turn, finish: journeyFinish, reserved,
    getSite: id => siteById.get(id), getBookJobs: () => bookTasks.jobs, permanentCount: population ? nightCount : actors.length }) : null;

  function update(dt, input = {}) {
    if (!Number.isFinite(dt) || dt <= 0) return;
    allowBridgeEntry = input.allowBridgeEntry ?? true; gateFactor = input.gateFactor ?? 1;
    const nextAngles = input.angles || angles;
    const priorAngles = { ...angles };
    const angleDelta = Object.fromEntries(IDS.map(id => [id, shortest(angles[id], nextAngles[id])]));
    const steps = Math.max(1, Math.ceil(dt / .10)), step = dt / steps;
    for (let s = 1; s <= steps; s++) {
      for (const id of IDS) {
        const next = priorAngles[id] + angleDelta[id] * s / steps;
        const turnDelta = shortest(angles[id], next);
        for (const a of actors) if (a._bridge === id && (a._state === 'bridge_walk' || a._state === 'seam' || a._state === 'life_bridge')) {
          a.yaw -= turnDelta; a._desiredYaw -= turnDelta;
        }
        angles[id] = next;
      }
      time += step; lifeTasks?.update(input); attendance?.update(input); scheduleBridges();
      bookTasks.update();
      outdoorVisits?.update();
      for (const id of IDS) mechanisms[id].update(step);
      updateCars(step);
      if (time >= trafficAt) { traffic(); trafficAt = time + .30; }
      for (const a of actors) actorTick(a, step);
      const onBridges = actors.filter(a => a.support.type === 'bridge').length;
      stats.maxBridgeOccupants = Math.max(stats.maxBridgeOccupants, onBridges);
    }
    firstUpdate = false; stats.elapsed = time;
    if (attendance) stats.actors = actors.filter(a => a.active !== false).length;
  }
  function snapshot() {
    return JSON.parse(JSON.stringify({ version: 1, time, randomState, angles, allowBridgeEntry, gateFactor,
      automaticAt, automaticBridge, trafficAt, manual, seam, stats, actors, elevators,
      reserved: [...reserved], mechanisms: Object.fromEntries(IDS.map(id => [id, { ...mechanisms[id], config: undefined }])),
      bookState: bookTasks.snapshot(), ...(outdoorVisits ? { outdoorState: outdoorVisits.snapshot() } : {}),
      ...(attendance ? { populationState: attendance.snapshot() } : {}),
      ...(lifeTasks ? { lifeState: lifeTasks.snapshot() } : {}) }));
  }
  function restore(data) {
    if (!data || data.version !== 1 || !Array.isArray(data.actors)) return false;
    if (attendance ? !attendance.validate(data.populationState, data.actors, data.reserved)
      : data.populationState || data.actors.length !== actors.length) return false;
    if (data.actors.some((a, i) => !a || a.id !== actors[i]?.id || a.index !== i)) return false;
    if (lifeTasks ? !lifeTasks.validate(data.lifeState, data.actors, data.reserved)
      : data.lifeState || data.actors.some(a => a._lifeJobId || a._weatherJobId)) return false;
    if (!bookTasks.validate(data.bookState, data.actors, data.reserved)) return false;
    if (outdoorVisits ? !outdoorVisits.validate(data.outdoorState, data.actors, data.reserved)
      : data.actors.some(a => a._outdoorVisitId || a._state === 'outdoor_dwell')) return false;
    const saved = JSON.parse(JSON.stringify(data));
    time = saved.time; randomState = saved.randomState; angles = saved.angles;
    allowBridgeEntry = saved.allowBridgeEntry; gateFactor = saved.gateFactor;
    automaticAt = saved.automaticAt; automaticBridge = saved.automaticBridge; trafficAt = saved.trafficAt;
    for (const id of IDS) { manual[id] = saved.manual[id]; Object.assign(seam[id], saved.seam[id]); Object.assign(mechanisms[id], saved.mechanisms[id]); }
    for (let i = 0; i < actors.length; i++) {
      const a = actors[i], item = saved.actors[i];
      if (!item) { attendance.resetAbsent(a); continue; }
      for (const key of Object.keys(a)) delete a[key]; Object.assign(a, item);
      a.destination = item.destination ? siteById.get(item.destination.id) || item.destination : null;
      a._atDestination = item._atDestination ? siteById.get(item._atDestination.id) || item._atDestination : null;
      if (attendance && !saved.populationState) { a.active = true; a.populationPhase = 'present'; a._populationPortalId = null; }
    }
    for (let i = 0; i < elevators.length; i++) Object.assign(elevators[i], saved.elevators[i]);
    reserved.clear(); for (const entry of saved.reserved) reserved.set(...entry);
    bookTasks.restore(saved.bookState);
    outdoorVisits?.restore(saved.outdoorState);
    attendance?.restore(saved.populationState);
    lifeTasks?.restore(saved.lifeState);
    Object.assign(stats, saved.stats); firstUpdate = false; return true;
  }
  return { actors, elevators, mechanisms, stats, update, canClose, requestBridgeLevel, snapshot, restore, sampleGround,
    get activeActors() { return actors.filter(a => a.active !== false); },
    get populationState() { return attendance?.state || { nightCount: actors.length, dayCount: actors.length,
      target: actors.length, active: actors.length, present: actors.length, arriving: 0, departing: 0, absent: 0 }; },
    requestBookSpill: options => bookTasks.request(options),
    get bookJobs() { return bookTasks.jobs; },
    get bookStats() { return bookTasks.stats; },
    get outdoorVisits() { return outdoorVisits?.visits || []; },
    get outdoorStats() { return outdoorVisits?.stats || null; },
    get lifeState() { return lifeTasks?.state || null; },
    get lifeStats() { return lifeTasks?.stats || null; },
    requestPaperInteraction: request => lifeTasks?.requestPaperInteraction(request) || { ok: false, reason: 'disabled' },
    markPaperReady: id => lifeTasks?.markPaperReady(id) || false,
    cancelPaperInteraction: id => lifeTasks?.cancelPaperInteraction(id) || false,
    get seamsClear() { return IDS.every(canClose); },
    get time() { return time; },
    get pendingBridgeLevels() { return { ...manual }; },
    get destinations() { return sites; },
  };
}
