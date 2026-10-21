import { clamp, shortest, smooth } from './mechanism.js';
import { appendLine, copyPoint, distance, floorY, outerRoute, segmentPoint } from './navigation.js';
import { outdoorSegmentPoint } from './outdoor_routes.js';
import { words } from './words.js';

const clone = value => JSON.parse(JSON.stringify(value));
const finitePoint = p => p && [p.x, p.y, p.z].every(Number.isFinite);
const pointAt = (segment, at = segment.progress) => outdoorSegmentPoint(segment, at) || segmentPoint(segment, at);
const radiusOf = p => Math.hypot(p.x, p.z);
const pathLength = path => path.reduce((sum, segment) => sum + segment.length, 0);
const pathEnd = (path, fallback) => path.length ? pointAt(path[path.length - 1], path[path.length - 1].length) : fallback;
const active = job => job.phase !== 'complete';
const cycleMinutes = value => ((value % 1440) + 1440) % 1440;
const deepFreeze = value => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
};

/** Seconds for each timed step of a story, and how long a finished job stays visible. */
const TIMES = Object.freeze({ paperNotice: .95, paperTimeout: 45, pickup: 2.1, open: 2.2, read: 9, fold: 2.1, store: 2.2,
  take: 2.7, inspect: 8, hang: 2.8, umbrellaTransfer: 2.9, coat: 2, complete: 12 });
const LIMITS = Object.freeze({ paperJobs: 2, paperCooldown: 3, helperRest: 70, bridgeCooldown: 95, readerRest: 500, patrolCooldown: 260 });

const initialStats = () => ({ bridgeStarted: 0, bridgeCompleted: 0, patrolStarted: 0, patrolCompleted: 0, paperRequested: 0,
  paperAccepted: 0, paperCompleted: 0, paperCancelled: 0, paperRejected: {}, umbrellasOpened: 0, umbrellasStored: 0,
  umbrellasRetrieved: 0, maxReaders: 0, maxPatrols: 0, maxPaperJobs: 0, recent: [] });

/** A walk on one floor, back out along the old place's approach, round the gallery and in along the site's approach. */
function routeTo(from, level, origin, site) {
  if (level === site.level && distance(from, site.position) < 1e-6 && Math.abs(from.y - site.position.y) < 1e-6) return [];
  const path = [], support = { type: 'floor', level };
  let p = copyPoint(from);
  if (origin) for (const target of [...origin.approach].reverse()) p = appendLine(path, p, target, support, .85);
  const ring = outerRoute(p, level, site.level, site.angle);
  path.push(...ring); p = pathEnd(ring, p);
  for (const target of site.approach) p = appendLine(path, p, target, { type: 'floor', level: site.level }, .84);
  appendLine(path, p, site.position, { type: 'floor', level: site.level }, .72);
  return path;
}

/**
 * Small everyday stories that always finish. They are a reader on the hour bridge, the night lantern
 * round, letters and paper planes taken to a return desk, and umbrellas stowed and taken back at the doors.
 * The simulation keeps owning position, walking, support and traffic. Each story borrows one resident,
 * keeps that resident's reservation, and gives the resident back where the story started.
 */
export function createLifeTasks({ actors, plan, getTime, getAngles, mechanisms, bridgePoint, walkSpeed, place, setState, setPath,
  still, turn, finish, reserved, getSite, getBookJobs, permanentCount = actors.length }) {
  const actorById = new Map(actors.map(a => [a.id, a]));
  const seat = plan.bridgeSeat || null, patrol = plan.patrol || null;
  const stands = plan.umbrellaStands || plan.umbrellas || [], standById = new Map(stands.map(s => [s.id, s]));
  const stations = (plan.paperStations || []).filter(s => finitePoint(s.position) && finitePoint(s.shelfPosition) && Array.isArray(s.approach));
  const stationById = new Map(stations.map(s => [s.id, s]));
  let jobs = [], umbrellaJobs = [], umbrellaItems = [], sequence = 0, umbrellaSequence = 0;
  let nextBridgeAt = 8, nextPatrolAt = 22, patrolCheckAt = 0, lastPaperAt = -1e6, minutes = 0, protectedActorId = null;
  let weather = { wet: false, rain: 0, snow: 0, night: 1, day: 0 };
  let stats = initialStats();
  const lifeJobOf = actor => actor._lifeJobId ? jobs.find(j => j.id === actor._lifeJobId && active(j)) : null;
  const umbrellaJobOf = actor => actor._weatherJobId ? umbrellaJobs.find(j => j.id === actor._weatherJobId) : null;
  const umbrellaOf = actor => umbrellaItems.find(item => item.actorId === actor.id);
  const present = actor => actor.active !== false && (!actor.populationPhase || actor.populationPhase === 'present');
  const free = actor => present(actor) && actor.id !== protectedActorId && actor._state === 'idle' && actor.support?.type === 'floor' &&
    actor._atDestination && !actor._bookJobId && !actor._outdoorVisitId && !actor._lifeJobId && !actor._weatherJobId &&
    !actor._bridge && !actor._exitLease && actor._liftId == null && !(actor._lifeCooldownUntil > getTime());
  const isNight = () => minutes < 300 || minutes >= 1320;

  function record(type, job) {
    stats.recent.push({ type, id: job.id, actorId: job.actorId, at: getTime(), seconds: getTime() - job.createdAt });
    if (stats.recent.length > 24) stats.recent.shift();
  }
  function setPhase(job, phase) { job.phase = phase; job.phaseStartedAt = getTime(); job.progress = 0; job.aligning = false; }
  function stationOf(job) {
    return job.kind === 'paper' ? stationById.get(job.stationId) : job.stationIndex < 0 ? patrol.rack : patrol.stops[job.stationIndex];
  }
  /** The chair turns with the hour bridge, so its world position follows the bridge angle. */
  function seatPoint() {
    const angle = getAngles()[seat.bridgeId], local = seat.seatLocal;
    return { x: Math.cos(angle) * local.x - Math.sin(angle) * local.z, y: mechanisms[seat.bridgeId].y + local.y,
      z: Math.sin(angle) * local.x + Math.cos(angle) * local.z };
  }
  /** What the views and the follow camera read about a resident's story. */
  function show(job, actor) {
    const task = { jobId: job.id, kind: job.kind, phase: job.phase, progress: job.progress };
    if (job.kind === 'bridge_reading') {
      task.seated = job.phase === 'sit' ? smooth(job.progress) : job.phase === 'read' ? 1 : job.phase === 'stand' ? 1 - smooth(job.progress) : 0;
      task.seatPosition = seatPoint(); task.seatHeight = seat.seatHeight; task.seatFacing = seat.yawOffset - getAngles()[seat.bridgeId];
      task.target = task.seatPosition;
    } else if (job.kind === 'patrol') {
      const station = stationOf(job);
      task.target = station.targetPosition || station.position; task.handTarget = patrol.rack.handTarget; task.facing = station.facing;
      task.holding = job.holding ? 1 : 0; task.stationId = station.id;
    } else {
      task.target = job.target; task.facing = stationById.get(job.stationId).facing; task.sourceKind = job.sourceKind;
      task.cancelled = !!job.cancelled;
    }
    actor.lifeTask = task;
  }
  function complete(job, actor) {
    const now = getTime();
    setPhase(job, 'complete'); job.progress = 1; job.finishedAt = now;
    delete actor._lifeJobId; delete actor.lifeTask; actor._lifeCooldownUntil = now + LIMITS.helperRest;
    if (job.kind === 'bridge_reading') {
      stats.bridgeCompleted++; nextBridgeAt = now + LIMITS.bridgeCooldown; actor._bridgeReadingUntil = now + LIMITS.readerRest;
      setState(actor, 'bridge_walk'); actor.activity = 'walk';
    } else {
      if (job.kind === 'patrol') {
        for (const id of job.reservationIds) if (id !== job.originId && reserved.get(id) === actor.id) reserved.delete(id);
        stats.patrolCompleted++; nextPatrolAt = now + LIMITS.patrolCooldown;
      } else if (job.cancelled) stats.paperCancelled++;
      else stats.paperCompleted++;
      actor.destination = getSite(job.originId); finish(actor);
    }
    record(job.cancelled ? `${job.kind}-cancelled` : `${job.kind}-complete`, job);
  }
  function goHome(job, actor, from) {
    setPhase(job, 'return'); actor.purpose = words.purposes.packUp;
    const origin = getSite(job.originId);
    setPath(actor, routeTo(actor.position, actor.level, from, origin), `life_home:${job.id}`); show(job, actor);
  }

  /** Only an adult already walking outward on the hour bridge's outbound lane, just short of the chair, sits down. */
  function tryStartBridge(actor) {
    if (!seat || getTime() < nextBridgeAt || !present(actor) || actor.id === protectedActorId || actor.age === 'child' ||
      actor._state !== 'bridge_walk' || actor._bridge !== seat.bridgeId || actor._direction !== 1 || Math.abs(actor._lane - .34) > .015 ||
      actor._r < seat.r - .46 || actor._r > seat.r - .04 || actor._lifeJobId || actor._bookJobId || actor._outdoorVisitId ||
      actor._weatherJobId || actor._exitLease || actor._bridgeReadingUntil > getTime() ||
      jobs.some(j => active(j) && j.kind === 'bridge_reading')) return false;
    const now = getTime();
    const job = { id: `life-reader-${++sequence}`, kind: 'bridge_reading', actorId: actor.id, bridgeId: seat.bridgeId, phase: 'approach',
      progress: 0, createdAt: now, phaseStartedAt: now, originId: actor.destination.id, originalLane: actor._lane,
      originalDirection: actor._direction, finishedAt: null };
    jobs.push(job); actor._lifeJobId = job.id; actor.purpose = words.purposes.bridgeReading; setState(actor, 'life_bridge'); show(job, actor);
    stats.bridgeStarted++; stats.maxReaders = Math.max(stats.maxReaders, 1); record('bridge_reading-start', job);
    return true;
  }
  /** Moves the reader in bridge coordinates, so the walk stays on the bridge while it turns. */
  function stepOnBridge(actor, r, lane, dt, speed = .47) {
    const dr = r - actor._r, dl = lane - actor._lane, gap = Math.hypot(dr, dl), angle = getAngles()[seat.bridgeId];
    if (gap < 1e-8) { place(actor, bridgePoint(seat.bridgeId, actor._r, actor._lane), 0, dt); return true; }
    const dx = Math.sin(angle) * dr + Math.cos(angle) * dl, dz = -Math.cos(angle) * dr + Math.sin(angle) * dl;
    const step = Math.min(gap, walkSpeed(actor, dt, speed, Math.atan2(dx, dz)) * dt);
    actor._r += dr * step / gap; actor._lane += dl * step / gap;
    place(actor, bridgePoint(seat.bridgeId, actor._r, actor._lane), step, dt);
    return step >= gap - 1e-8;
  }
  function tickBridge(job, actor, dt) {
    const facing = seat.yawOffset - getAngles()[seat.bridgeId];
    if (job.phase === 'approach') { if (stepOnBridge(actor, seat.r, job.originalLane, dt)) setPhase(job, 'align'); }
    else if (job.phase === 'align') {
      if (stepOnBridge(actor, seat.r, seat.lane, dt, .37)) {
        still(actor, dt, 'observe'); turn(actor, facing, dt);
        if (Math.abs(shortest(actor.yaw, facing)) < .02) setPhase(job, 'sit');
      }
    } else if (['sit', 'read', 'stand'].includes(job.phase)) {
      place(actor, bridgePoint(seat.bridgeId, actor._r, actor._lane), 0, dt);
      still(actor, dt, job.phase === 'stand' ? 'observe' : 'read'); turn(actor, facing, dt);
      const duration = job.phase === 'read' ? seat.dwellSeconds : seat.transitionSeconds;
      job.progress = clamp((getTime() - job.phaseStartedAt) / duration);
      if (job.progress >= 1) setPhase(job, job.phase === 'sit' ? 'read' : job.phase === 'read' ? 'stand' : 'rejoin');
    } else if (job.phase === 'rejoin') {
      // The reader waits beside the chair until the walking lane next to it is clear.
      const laneBusy = actors.some(other => other !== actor && other.active !== false && other._bridge === seat.bridgeId &&
        other.support?.type === 'bridge' && Math.abs(other._r - seat.r) < 1.35 && Math.abs(other._lane - job.originalLane) < .22);
      if (laneBusy && Math.abs(actor._lane - seat.lane) < .02) {
        place(actor, bridgePoint(seat.bridgeId, actor._r, actor._lane), 0, dt); still(actor, dt, 'observe');
      } else if (stepOnBridge(actor, seat.r, job.originalLane, dt, .4)) complete(job, actor);
    }
    actor.support = { type: 'bridge', id: seat.bridgeId, r: actor._r, lane: actor._lane };
    actor.level = clamp(Math.round(actor.position.y / 6), 0, 6);
    if (active(job)) show(job, actor);
    return true;
  }

  /** At night one free adult on the lantern's level, a librarian or clockmaker first, walks the lantern round. */
  function startPatrol() {
    if (!patrol?.stops?.length || !isNight() || getTime() < nextPatrolAt || jobs.some(j => active(j) && j.kind === 'patrol')) return;
    const reservationIds = [...new Set([patrol.rack, ...patrol.stops].map(stop => stop.reservationId).filter(Boolean))];
    const candidates = actors.filter(a => free(a) && a.age !== 'child' && a.level === patrol.rack.level && a.index < permanentCount);
    const actor = candidates.filter(a => reservationIds.every(id => !reserved.has(id) || reserved.get(id) === a.id))
      .sort((a, b) => (['librarian', 'clockmaker'].includes(b.roleTag) ? 1 : 0) - (['librarian', 'clockmaker'].includes(a.roleTag) ? 1 : 0) ||
        a.index - b.index)[0];
    if (!actor) return;
    const now = getTime(), origin = actor._atDestination;
    const job = { id: `life-patrol-${++sequence}`, kind: 'patrol', actorId: actor.id, originId: origin.id, createdAt: now,
      phaseStartedAt: now, phase: 'approach', progress: 0, stationIndex: -1, reservationIds, holding: false, returnRequested: false,
      finishedAt: null };
    jobs.push(job);
    for (const id of reservationIds) reserved.set(id, actor.id);
    actor._lifeJobId = job.id; actor._atDestination = null; actor._initialCenter = false; actor.purpose = words.purposes.fetchLantern;
    show(job, actor); setPath(actor, routeTo(actor.position, actor.level, origin, patrol.rack), `life_patrol_arrive:${job.id}`);
    stats.patrolStarted++; stats.maxPatrols = Math.max(stats.maxPatrols, 1); record('patrol-start', job);
  }
  function nextPatrolStop(job, actor) {
    const from = stationOf(job);
    if (job.returnRequested || job.stationIndex >= patrol.stops.length - 1) {
      job.stationIndex = -1; setPhase(job, 'walk'); job.returningRack = true; actor.purpose = words.purposes.hangLantern;
    } else {
      job.stationIndex++; setPhase(job, 'walk'); actor.purpose = words.purposes.patrol;
    }
    const to = stationOf(job);
    show(job, actor); setPath(actor, routeTo(actor.position, actor.level, from, to), `life_patrol_arrive:${job.id}`);
  }

  /** A letter or a paper plane landed on a return desk. A free resident on that floor will notice it and come. */
  function requestPaperInteraction({ kind = 'letter', stationId, position } = {}) {
    stats.paperRequested++;
    const reject = reason => { stats.paperRejected[reason] = (stats.paperRejected[reason] || 0) + 1; return { ok: false, reason }; };
    const station = stationById.get(stationId), now = getTime();
    if (!['letter', 'plane'].includes(kind) || !station || !finitePoint(position) || distance(position, station.shelfPosition) > .4 ||
      Math.abs(position.y - station.shelfPosition.y) > .12) return reject('invalid-place');
    const paperJobs = jobs.filter(j => active(j) && j.kind === 'paper');
    if (paperJobs.length >= LIMITS.paperJobs) return reject('busy');
    if (paperJobs.some(j => j.level === station.level) || (getBookJobs?.() || []).some(j => active(j) && j.level === station.level)) return reject('floor-busy');
    if (now - lastPaperAt < LIMITS.paperCooldown) return reject('cooldown');
    const helpers = actors.filter(a => free(a) && a.level === station.level)
      .map(a => ({ actor: a, path: routeTo(a.position, a.level, a._atDestination, station) }));
    helpers.sort((a, b) => pathLength(a.path) - pathLength(b.path) || a.actor.index - b.actor.index);
    if (!helpers.length) return reject('no-helper');
    const { actor, path } = helpers[0], origin = actor._atDestination;
    const job = { id: `life-paper-${++sequence}`, kind: 'paper', actorId: actor.id, originId: origin.id, sourceKind: kind, stationId,
      level: station.level, target: copyPoint(position), createdAt: now, phase: 'await', phaseStartedAt: now, progress: 0, ready: false,
      cancelled: false, approachPath: path, finishedAt: null };
    jobs.push(job); lastPaperAt = now;
    actor._lifeJobId = job.id; actor._atDestination = null; actor._initialCenter = false;
    setState(actor, 'life_paper'); still(actor, 0, 'notice');
    actor.purpose = kind === 'plane' ? words.purposes.planeNotice : words.purposes.letterNotice;
    show(job, actor); stats.paperAccepted++; stats.maxPaperJobs = Math.max(stats.maxPaperJobs, paperJobs.length + 1);
    return { ok: true, jobId: job.id, actorId: actor.id };
  }
  /** The paper has landed, so the resident who noticed it may walk over. */
  function markPaperReady(id) {
    const job = jobs.find(j => j.id === id && j.kind === 'paper' && active(j));
    if (!job || job.cancelled) return false;
    job.ready = true; return true;
  }
  function cancelPaperInteraction(id) {
    const job = jobs.find(j => j.id === id && j.kind === 'paper' && active(j));
    if (!job) return false;
    job.cancelled = true;
    if (job.phase === 'await') complete(job, actorById.get(job.actorId));
    else if (['pickup', 'open', 'read', 'fold', 'store'].includes(job.phase)) goHome(job, actorById.get(job.actorId), stationById.get(job.stationId));
    return true;
  }
  /** Turns to the station first. A timed step starts its clock only once the turn is done. */
  function settle(job, actor, facing, dt, activity) {
    still(actor, dt, activity); turn(actor, facing, dt);
    if (!job.aligning) return true;
    job.progress = 0;
    if (Math.abs(shortest(actor.yaw, facing)) < .015) { job.aligning = false; job.phaseStartedAt = getTime(); }
    return false;
  }
  function tickPaper(job, actor, dt) {
    if (job.phase === 'await') {
      still(actor, dt, 'notice');
      const origin = getSite(job.originId);
      turn(actor, origin.facing ?? actor.yaw, dt);
      job.progress = clamp((getTime() - job.phaseStartedAt) / TIMES.paperNotice);
      if (getTime() - job.createdAt > TIMES.paperTimeout) cancelPaperInteraction(job.id);
      else if (job.ready && job.progress >= 1) {
        setPhase(job, 'approach'); setPath(actor, job.approachPath, `life_paper_arrive:${job.id}`);
        actor.purpose = words.purposes.toPaper;
      }
      if (active(job)) show(job, actor);
      return true;
    }
    if (['approach', 'return'].includes(job.phase)) { show(job, actor); return false; }
    const station = stationById.get(job.stationId);
    const activity = job.phase === 'read' ? 'read' : job.phase === 'pickup' ? 'pickup' : 'work';
    if (settle(job, actor, station.facing, dt, activity)) {
      job.progress = clamp((getTime() - job.phaseStartedAt) / TIMES[job.phase]);
      if (job.progress >= 1) {
        const next = { pickup: 'open', open: 'read', read: 'fold', fold: 'store' }[job.phase];
        if (next) setPhase(job, next);
        else goHome(job, actor, station);
      }
    }
    show(job, actor);
    return true;
  }
  function tickPatrol(job, actor, dt) {
    if (['approach', 'walk', 'return'].includes(job.phase)) { show(job, actor); return false; }
    const station = stationOf(job);
    if (settle(job, actor, station.facing, dt, job.phase === 'inspect' ? 'observe' : 'work')) {
      job.progress = clamp((getTime() - job.phaseStartedAt) / TIMES[job.phase]);
      // The lantern changes hands part of the way through taking it down and hanging it up.
      if (job.phase === 'take' && job.progress >= .4) job.holding = true;
      if (job.phase === 'hang' && job.progress >= .7) job.holding = false;
      if (job.progress >= 1) {
        if (job.phase === 'hang') goHome(job, actor, patrol.rack);
        else nextPatrolStop(job, actor);
      }
    }
    show(job, actor);
    return true;
  }
  function pathDone(actor, after) {
    if (typeof after !== 'string' || !after.startsWith('life_')) return false;
    if (after.startsWith('life_weather_')) return umbrellaPathDone(actor, after);
    const job = lifeJobOf(actor);
    if (!job || !after.endsWith(`:${job.id}`)) return false;
    if (after.startsWith('life_home:')) complete(job, actor);
    else if (after.startsWith('life_paper_arrive:')) {
      if (job.cancelled) goHome(job, actor, stationById.get(job.stationId));
      else { setPhase(job, 'pickup'); job.aligning = true; setState(actor, 'life_paper'); actor.purpose = words.purposes.readLetter; }
    } else if (after.startsWith('life_patrol_arrive:')) {
      setPhase(job, job.stationIndex < 0 ? job.returningRack ? 'hang' : 'take' : 'inspect');
      job.aligning = true; setState(actor, 'life_patrol');
    } else return false;
    if (active(job)) show(job, actor);
    return true;
  }

  /** The umbrella stand at a doorway that the rest of this ground-floor walk passes through, if any. */
  function standOnPath(actor) {
    if (actor._state !== 'path') return null;
    for (let i = actor._pathIndex; i < actor._path.length; i++) {
      const segment = actor._path[i];
      if (segment.kind !== 'line' || Math.abs(segment.from.y - floorY(0)) > .02 || Math.abs(segment.to.y - floorY(0)) > .02) continue;
      const dx = segment.to.x - segment.from.x, dz = segment.to.z - segment.from.z, lengthSquared = dx * dx + dz * dz;
      for (const stand of stands) {
        const door = stand.approach[0], t = ((door.x - segment.from.x) * dx + (door.z - segment.from.z) * dz) / lengthSquared;
        if (t < 0 || t > 1) continue;
        if (Math.hypot(segment.from.x + t * dx - door.x, segment.from.z + t * dz - door.z) < .55) return stand;
      }
    }
    return null;
  }
  /** Gives the resident an umbrella of their own, with a fixed slot in the stand they will reach first. */
  function issueUmbrella(actor) {
    const ahead = standOnPath(actor);
    const order = ahead ? [ahead, ...stands.filter(s => s !== ahead)]
      : [...stands].sort((a, b) => distance(actor.position, a.position) - distance(actor.position, b.position));
    for (const stand of order) {
      const taken = new Set(umbrellaItems.filter(item => item.standId === stand.id).map(item => item.slot));
      let slot = 0;
      while (taken.has(slot)) slot++;
      if (slot >= stand.capacity) continue;
      const item = { id: `umbrella-${++umbrellaSequence}`, standId: stand.id, slot, actorId: actor.id, colorIndex: actor.index % 4, stored: false };
      umbrellaItems.push(item); stats.umbrellasOpened++;
      return item;
    }
    return null;
  }
  /** Rain wets residents in the open and makes them hug their books. An umbrella opens past the eaves and closes under them. */
  function updateWeatherPose(actor, dt) {
    const outside = actor.support?.type === 'garden' && actor.position.y < -.1;
    const pastEaves = outside && radiusOf(actor.position) > 39.5;
    let umbrella = umbrellaOf(actor);
    const errand = umbrellaJobOf(actor);
    if (!umbrella && actor.age !== 'child' && actor.active !== false && weather.wet && pastEaves && !actor._lifeJobId && !actor._bookJobId &&
      (actor._outdoorVisitId || actor.populationPhase === 'arriving' && actor.index % 4 === 0)) umbrella = issueUmbrella(actor);
    const pose = actor.weatherPose || (actor.weatherPose = { umbrella: 0, held: 0, wet: 0, hug: 0, phase: 'carry', progress: 0 });
    const ease = (key, target, rate) => {
      const step = Math.min(Math.abs(target - (pose[key] || 0)), dt * rate);
      pose[key] = (pose[key] || 0) + Math.sign(target - (pose[key] || 0)) * step;
    };
    ease('wet', outside && weather.wet ? 1 : 0, outside ? .18 : .075);
    ease('hug', outside && weather.wet && actor.carrying?.type === 'book' ? 1 : 0, .48);
    ease('umbrella', umbrella && !umbrella.stored && pastEaves && weather.wet && !errand ? 1 : 0, .62);
    if (!errand) {
      ease('held', umbrella && !umbrella.stored ? 1 : 0, .62);
      pose.phase = umbrella?.stored ? 'stored' : 'carry'; pose.progress = 0; pose.transfer = 0; delete pose.handTarget;
    }
    if (umbrella) { pose.standId = umbrella.standId; pose.slot = umbrella.slot; pose.itemId = umbrella.id; }
  }
  /**
   * At a doorway, an arriving resident stows an open umbrella and a leaving resident takes theirs back.
   * Only one handover runs at a stand, and later arrivals wait beside the door. The walk's own path is
   * set aside with its progress and resumed afterwards. Returns true while the resident waits.
   */
  function startUmbrellaErrand(actor) {
    delete actor._weatherWaiting;
    const item = umbrellaOf(actor);
    if (!item || actor._weatherJobId || actor._lifeJobId || actor._bookJobId || actor._state !== 'path' || actor.level !== 0 ||
      Math.abs(actor.position.y - floorY(0)) > .015) return;
    const comingIn = actor.populationPhase === 'arriving' || actor.outdoorTask?.phase === 'return';
    const goingOut = actor.populationPhase === 'departing' || ['outbound', 'stroll'].includes(actor.outdoorTask?.phase);
    const mode = comingIn && !item.stored ? 'stow' : goingOut && item.stored ? 'retrieve' : null;
    if (!mode) return;
    const door = standOnPath(actor);
    if (!door) return;
    let stand = standById.get(item.standId);
    if (mode === 'stow' && stand !== door) {
      // An umbrella coming in moves to the stand at the door in use when that stand has room.
      const taken = new Set(umbrellaItems.filter(other => other !== item && other.standId === door.id).map(other => other.slot));
      let slot = 0;
      while (taken.has(slot)) slot++;
      if (slot < door.capacity) { item.standId = door.id; item.slot = slot; stand = door; }
    }
    if (umbrellaJobs.some(job => job.standId === stand.id)) {
      if (distance(actor.position, door.approach[0]) < 2.4) { actor._weatherWaiting = stand.id; return true; }
      return;
    }
    if (distance(actor.position, door.approach[0]) > .5) return;
    const job = { id: `life-weather-${++sequence}`, actorId: actor.id, itemId: item.id, standId: stand.id, mode, phase: 'approach',
      progress: 0, phaseStartedAt: getTime(), createdAt: getTime(),
      resume: { point: copyPoint(actor.position), support: clone(actor.support), path: actor._path, pathIndex: actor._pathIndex,
        after: actor._after, activity: actor.activity, purpose: actor.purpose, stateAt: actor._stateAt },
      entryStandId: door.id };
    const path = [], support = { type: 'floor', level: 0 };
    let p = appendLine(path, actor.position, door.approach[0], support, .72);
    if (stand !== door) {
      const ring = outerRoute(p, 0, 0, stand.angle);
      path.push(...ring); p = pathEnd(ring, p);
    }
    for (const target of stand.approach) p = appendLine(path, p, target, support, .7);
    umbrellaJobs.push(job); actor._weatherJobId = job.id;
    actor.purpose = mode === 'stow' ? words.purposes.stowUmbrella : words.purposes.takeUmbrella;
    setPath(actor, path, `life_weather_arrive:${job.id}`);
  }
  function umbrellaPathDone(actor, after) {
    const job = umbrellaJobOf(actor);
    if (!job || !after.endsWith(`:${job.id}`)) return false;
    if (after.startsWith('life_weather_arrive:')) { setPhase(job, job.mode); job.aligning = true; setState(actor, 'life_weather'); }
    else if (after.startsWith('life_weather_resume:')) {
      const saved = job.resume;
      actor._path = saved.path; actor._pathIndex = saved.pathIndex; actor._after = saved.after; actor.support = saved.support;
      actor.activity = saved.activity; actor.purpose = saved.purpose; setState(actor, 'path'); actor._stateAt = saved.stateAt;
      delete actor._weatherJobId; umbrellaJobs = umbrellaJobs.filter(j => j !== job);
    } else return false;
    return true;
  }
  /** Back from the stand to the exact point where the walk was set aside. */
  function returnToWalk(job, actor) {
    const stand = standById.get(job.standId), door = standById.get(job.entryStandId), support = { type: 'floor', level: 0 }, path = [];
    let p = copyPoint(actor.position);
    for (const target of [...stand.approach].reverse()) p = appendLine(path, p, target, support, .72);
    if (stand !== door) {
      const ring = outerRoute(p, 0, 0, door.angle);
      path.push(...ring); p = pathEnd(ring, p); p = appendLine(path, p, door.approach[0], support, .8);
    }
    appendLine(path, p, job.resume.point, support, .7);
    setPhase(job, 'return'); setPath(actor, path, `life_weather_resume:${job.id}`);
  }
  function tickUmbrella(job, actor, dt) {
    const pose = actor.weatherPose, stand = standById.get(job.standId), item = umbrellaOf(actor);
    pose.phase = job.phase; pose.progress = job.progress; pose.standId = stand.id; pose.slot = item.slot; pose.transfer = job.progress;
    pose.handTarget = copyPoint(stand.handTargets?.[item.slot] || stand.handTarget);
    if (['approach', 'return'].includes(job.phase)) return false;
    if (settle(job, actor, stand.facing, dt, 'work')) {
      job.progress = clamp((getTime() - job.phaseStartedAt) / (job.phase === 'coat' ? TIMES.coat : TIMES.umbrellaTransfer));
      if (job.phase === 'stow') {
        pose.held = 1 - smooth((job.progress - .6) / .2);
        if (job.progress >= .7 && !item.stored) { item.stored = true; stats.umbrellasStored++; }
      } else if (job.phase === 'retrieve') {
        pose.held = smooth((job.progress - .32) / .16);
        if (job.progress >= .4 && item.stored) { item.stored = false; stats.umbrellasRetrieved++; }
      }
      pose.progress = job.progress; pose.transfer = job.progress;
      if (job.progress >= 1) {
        if (job.phase === 'coat' || job.phase === 'retrieve') returnToWalk(job, actor);
        else setPhase(job, 'coat');
      }
    }
    return true;
  }

  function tickActor(actor, dt) {
    updateWeatherPose(actor, dt);
    if (!actor._weatherJobId && startUmbrellaErrand(actor)) { still(actor, dt, 'wait'); return true; }
    const errand = umbrellaJobOf(actor);
    if (errand) return tickUmbrella(errand, actor, dt);
    const job = lifeJobOf(actor);
    if (!job) return false;
    if (job.kind === 'bridge_reading') return tickBridge(job, actor, dt);
    if (job.kind === 'paper') return tickPaper(job, actor, dt);
    return tickPatrol(job, actor, dt);
  }
  /** Reads the displayed clock and the scene's weather. Nothing here asks an outside weather service. */
  function update(options = {}) {
    if (Number.isFinite(options.minutes)) minutes = cycleMinutes(options.minutes);
    protectedActorId = options.protectedActorId || null;
    const environment = options.environment || {};
    const rain = clamp(Number.isFinite(environment.rain) ? environment.rain : 0), snow = clamp(Number.isFinite(environment.snow) ? environment.snow : 0);
    weather = { rain, snow, wet: rain > .08 || snow > .05 || ['rain', 'snow', 'storm'].includes(environment.weather),
      night: Number.isFinite(environment.night) ? environment.night : isNight() ? 1 : 0,
      day: Number.isFinite(environment.day) ? environment.day : isNight() ? 0 : 1,
      season: environment.season || null, weather: environment.weather || 'clear' };
    // At daybreak the lantern round finishes the walk already under way, then the lantern goes back.
    for (const job of jobs) if (active(job) && job.kind === 'patrol' && !isNight()) job.returnRequested = true;
    jobs = jobs.filter(job => active(job) || getTime() - job.finishedAt < TIMES.complete);
    for (const actor of actors) if (actor.active === false) {
      umbrellaItems = umbrellaItems.filter(item => item.actorId !== actor.id);
      delete actor.weatherPose; delete actor._weatherWaiting;
    }
    if (getTime() >= patrolCheckAt) { patrolCheckAt = getTime() + 4; startPatrol(); }
  }
  function snapshot() {
    return clone({ version: 1, sequence, umbrellaSequence, jobs, umbrellaJobs, umbrellaItems, nextBridgeAt, nextPatrolAt, patrolCheckAt,
      lastPaperAt, minutes, weather, stats });
  }
  function validPath(path, level = null) {
    return Array.isArray(path) && path.every(s => s && ['line', 'arc', 'stairs', 'gardenArc', 'gardenStairs'].includes(s.kind) &&
      Number.isFinite(s.length) && s.length > 0 && Number.isFinite(s.progress) && s.progress >= 0 && s.progress <= s.length + 1e-7 &&
      Number.isFinite(s.speed) && s.speed > 0 && s.support && ['floor', 'stairs', 'garden'].includes(s.support.type) &&
      (level == null || s.support.level === level) &&
      (s.kind === 'line' ? finitePoint(s.from) && finitePoint(s.to) : Number.isFinite(s.radius ?? s.fromRadius)));
  }
  function onPath(actor, level = null) {
    if (!validPath(actor._path, level) || !Number.isInteger(actor._pathIndex) || !actor._path[actor._pathIndex]) return false;
    const p = pointAt(actor._path[actor._pathIndex]);
    return finitePoint(p) && finitePoint(actor.position) && Math.hypot(p.x - actor.position.x, p.y - actor.position.y, p.z - actor.position.z) < 1e-3;
  }
  function validate(data, savedActors, reservedList = []) {
    if (!data) return !savedActors.some(a => a._lifeJobId || a._weatherJobId || a.lifeTask || a.weatherPose || a._state?.startsWith('life_'));
    if (data.version !== 1 || !Number.isInteger(data.sequence) || !Number.isInteger(data.umbrellaSequence) || !Array.isArray(data.jobs) ||
      !Array.isArray(data.umbrellaJobs) || !Array.isArray(data.umbrellaItems) || data.jobs.length > 12 ||
      data.umbrellaJobs.length > stands.length || data.umbrellaItems.length > stands.length * 6 ||
      ![data.nextBridgeAt, data.nextPatrolAt, data.patrolCheckAt, data.lastPaperAt, data.minutes].every(Number.isFinite) ||
      !data.stats || !data.weather) return false;
    const busy = new Set(), seen = new Set(), reservations = new Map(reservedList), counts = {}, paperFloors = new Set();
    for (const job of data.jobs) {
      const actor = savedActors.find(a => a.id === job.actorId);
      if (!job || !actor || typeof job.id !== 'string' || seen.has(job.id) || !['bridge_reading', 'patrol', 'paper'].includes(job.kind) ||
        ![job.createdAt, job.phaseStartedAt, job.progress].every(Number.isFinite) || job.progress < 0 || job.progress > 1 ||
        !getSite(job.originId)) return false;
      seen.add(job.id);
      if (job.kind === 'paper' && (!stationById.has(job.stationId) || !finitePoint(job.target) || !['letter', 'plane'].includes(job.sourceKind) ||
        typeof job.ready !== 'boolean' || typeof job.cancelled !== 'boolean')) return false;
      if (!active(job)) {
        if (!Number.isFinite(job.finishedAt)) return false;
        continue;
      }
      if (busy.has(actor.id) || actor._lifeJobId !== job.id || actor.lifeTask?.jobId !== job.id || actor.lifeTask?.kind !== job.kind ||
        actor.active === false || actor.populationPhase && actor.populationPhase !== 'present' || actor._bookJobId || actor._outdoorVisitId ||
        actor._weatherJobId || actor.destination?.id !== job.originId || reservations.get(job.originId) !== actor.id || actor._atDestination) return false;
      busy.add(actor.id); counts[job.kind] = (counts[job.kind] || 0) + 1;
      if (job.kind === 'bridge_reading') {
        if (!seat || job.bridgeId !== seat.bridgeId || actor._state !== 'life_bridge' || actor._bridge !== seat.bridgeId ||
          actor.support?.type !== 'bridge' || actor._exitLease || actor._direction !== job.originalDirection || job.originalDirection !== 1 ||
          Math.abs(job.originalLane - .34) > .01 || !['approach', 'align', 'sit', 'read', 'stand', 'rejoin'].includes(job.phase) ||
          !Number.isFinite(actor._r) || !Number.isFinite(actor._lane) || actor._r < seat.r - .47 || actor._r > seat.r + .01 ||
          actor._lane < .33 || actor._lane > seat.lane + .01) return false;
      } else {
        const walking = ['approach', 'walk', 'return'].includes(job.phase);
        if (actor.support?.type !== 'floor' || actor._bridge || actor._exitLease || actor._liftId != null ||
          walking && (actor._state !== 'path' || !onPath(actor, actor.level) || !actor._after?.endsWith(`:${job.id}`)) ||
          !walking && actor._state !== (job.kind === 'paper' ? 'life_paper' : 'life_patrol')) return false;
        if (job.kind === 'paper') {
          const station = stationById.get(job.stationId);
          if (!station || station.level !== job.level || job.level !== actor.level || paperFloors.has(job.level) ||
            !['await', 'approach', 'pickup', 'open', 'read', 'fold', 'store', 'return'].includes(job.phase) || !finitePoint(job.target) ||
            distance(job.target, station.shelfPosition) > .4 || Math.abs(job.target.y - station.shelfPosition.y) > .12 ||
            !['letter', 'plane'].includes(job.sourceKind) || typeof job.ready !== 'boolean' || typeof job.cancelled !== 'boolean' ||
            !validPath(job.approachPath, job.level)) return false;
          paperFloors.add(job.level);
        } else if (!patrol || actor.level !== patrol.rack.level || !['approach', 'take', 'walk', 'inspect', 'hang', 'return'].includes(job.phase) ||
          !Number.isInteger(job.stationIndex) || job.stationIndex < -1 || job.stationIndex >= patrol.stops.length ||
          typeof job.holding !== 'boolean' || !Array.isArray(job.reservationIds) ||
          job.reservationIds.some(id => reservations.get(id) !== actor.id)) return false;
      }
    }
    if (counts.bridge_reading > 1 || counts.patrol > 1 || counts.paper > LIMITS.paperJobs) return false;
    const slots = new Set(), owners = new Set(), itemIds = new Set(), busyStands = new Set(), onErrand = new Set();
    for (const item of data.umbrellaItems) {
      const actor = savedActors.find(a => a.id === item.actorId), stand = standById.get(item.standId), slot = `${item.standId}:${item.slot}`;
      if (!stand || !actor || actor.active === false || actor.age === 'child' || typeof item.id !== 'string' || itemIds.has(item.id) ||
        owners.has(item.actorId) || slots.has(slot) || !Number.isInteger(item.slot) || item.slot < 0 || item.slot >= stand.capacity ||
        typeof item.stored !== 'boolean') return false;
      itemIds.add(item.id); slots.add(slot); owners.add(item.actorId);
    }
    for (const job of data.umbrellaJobs) {
      const actor = savedActors.find(a => a.id === job.actorId), item = data.umbrellaItems.find(i => i.id === job.itemId), saved = job.resume;
      if (!actor || actor.active === false || actor._weatherJobId !== job.id || actor._lifeJobId || actor._bookJobId || onErrand.has(actor.id) ||
        busyStands.has(job.standId) || !standById.has(job.standId) || !standById.has(job.entryStandId) || !item || item.actorId !== actor.id ||
        item.standId !== job.standId || !['stow', 'retrieve'].includes(job.mode) ||
        !['approach', 'stow', 'retrieve', 'coat', 'return'].includes(job.phase) ||
        ![job.phaseStartedAt, job.progress, job.createdAt].every(Number.isFinite) || !saved || !finitePoint(saved.point) ||
        !validPath(saved.path) || !Number.isInteger(saved.pathIndex) || !saved.path[saved.pathIndex] ||
        !['population_arrive', 'population_depart', 'outdoor_arrive', 'outdoor_finish'].includes(saved.after) ||
        Math.hypot(pointAt(saved.path[saved.pathIndex]).x - saved.point.x, pointAt(saved.path[saved.pathIndex]).y - saved.point.y,
          pointAt(saved.path[saved.pathIndex]).z - saved.point.z) > 1e-3 ||
        actor.level !== 0 || actor.support?.type !== 'floor' ||
        (['approach', 'return'].includes(job.phase) ? actor._state !== 'path' || !onPath(actor, 0) || !actor._after?.endsWith(`:${job.id}`)
          : actor._state !== 'life_weather')) return false;
      onErrand.add(actor.id); busyStands.add(job.standId);
    }
    return savedActors.every(a => (!a._lifeJobId || busy.has(a.id)) && (!a._weatherJobId || onErrand.has(a.id)));
  }
  function restore(data) {
    if (!data) {
      jobs = []; umbrellaJobs = []; umbrellaItems = []; sequence = umbrellaSequence = 0;
      nextBridgeAt = getTime() + 8; nextPatrolAt = getTime() + 22; patrolCheckAt = getTime(); lastPaperAt = -1e6; stats = initialStats();
      return;
    }
    const saved = clone(data);
    ({ jobs, umbrellaJobs, umbrellaItems, sequence, umbrellaSequence, nextBridgeAt, nextPatrolAt, patrolCheckAt, lastPaperAt, minutes, weather } = saved);
    stats = { ...initialStats(), ...saved.stats };
    for (const job of jobs) if (active(job)) {
      const actor = actorById.get(job.actorId);
      // A helper walking to a desk follows the job's own path, so after loading they are one object again.
      if (job.kind === 'paper' && job.phase === 'approach') job.approachPath = actor._path;
      show(job, actor);
    }
  }
  function describe(job) {
    const view = { id: job.id, jobId: job.id, actorId: job.actorId, kind: job.kind, phase: job.phase, progress: job.progress,
      createdAt: job.createdAt, finishedAt: job.finishedAt };
    if (job.kind === 'paper') Object.assign(view, { stationId: job.stationId, target: copyPoint(job.target), sourceKind: job.sourceKind,
      ready: job.ready, cancelled: job.cancelled, level: job.level });
    if (job.kind === 'patrol') Object.assign(view, { holding: job.holding, stationIndex: job.stationIndex, stationId: stationOf(job).id,
      returnRequested: job.returnRequested });
    return view;
  }
  return { update, tickActor, pathDone, tryStartBridge, snapshot, validate, restore, requestPaperInteraction, markPaperReady,
    cancelPaperInteraction, isPaperFloorBusy: level => jobs.some(job => active(job) && job.kind === 'paper' && job.level === level),
    get weather() { return { ...weather }; }, get stats() { return clone(stats); },
    get state() {
      return deepFreeze({
        bridgeReading: jobs.filter(job => active(job) && job.kind === 'bridge_reading').map(describe)[0] || null,
        patrol: jobs.filter(job => active(job) && job.kind === 'patrol').map(describe)[0] || null,
        paperJobs: jobs.filter(job => job.kind === 'paper').map(describe), weather: { ...weather }, minutes,
        umbrellaCounts: Object.fromEntries(stands.map(stand => [stand.id, umbrellaItems.filter(item => item.standId === stand.id && item.stored).length])),
        umbrellaSlots: clone(umbrellaItems),
        umbrellaJobs: clone(umbrellaJobs.map(job => ({ id: job.id, actorId: job.actorId, standId: job.standId, mode: job.mode, phase: job.phase,
          progress: job.progress }))),
        stats: clone(stats) });
    } };
}
