import { clamp, smooth } from './mechanism.js';
import { FLOOR_OFFSET, appendLine, copyPoint, distance, segmentPoint as floorSegmentPoint } from './navigation.js';
import { outdoorSegmentPoint } from './outdoor_routes.js';
import { words } from './words.js';

const clone = value => JSON.parse(JSON.stringify(value));
const finitePoint = p => p && [p.x, p.y, p.z].every(Number.isFinite);
const at = segment => outdoorSegmentPoint(segment) || floorSegmentPoint(segment);
const phases = ['absent', 'arriving', 'present', 'departing'];
const cycleMinutes = value => ((value % 1440) + 1440) % 1440;

/** The displayed clock owns attendance. Weather and frame rate do not. */
export function populationTarget(minutes, nightCount) {
  const hour = cycleMinutes(Number.isFinite(minutes) ? minutes : 0) / 60;
  const day = hour < 6 || hour >= 22 ? 0 : hour < 9 ? smooth((hour - 6) / 3)
    : hour < 18 ? 1 : 1 - smooth((hour - 18) / 4);
  return Math.round(nightCount * (1 + 2 * day));
}

/** Garden plans contain physical surfaces; walkers contain the shoe offset. */
export function populationPortals(portals) {
  const feet = p => ({ ...p, y: p.y + FLOOR_OFFSET });
  if (!Array.isArray(portals) || !portals.length) throw new Error('Population needs physical arrival portals');
  return portals.map(portal => {
    if (!finitePoint(portal.position) || !Array.isArray(portal.approach) || !portal.approach.every(finitePoint))
      throw new Error('Invalid population portal');
    return { ...portal, position: feet(portal.position), approach: portal.approach.map(feet),
      enterApproach: (portal.enterApproach || portal.approach).map(feet),
      exitApproach: (portal.exitApproach || portal.approach).map(feet) };
  });
}

const initialStats = () => ({ arrivalsStarted: 0, entered: 0, departuresStarted: 0, left: 0,
  maxActive: 0, legacyMigrations: 0, recent: [] });

/** A fixed identity roster, with actual walks to and from covered gatehouses.
 * A committed journey is never reversed just because the clock changes again.
 * Existing jobs, seams, elevators and garden visits keep their ownership. */
export function createPopulationSchedule({ actors, nightCount, minutes = 0, portals, routing,
  getTime, reserved, chooseDestination, setPath, setState, finish }) {
  const dayCount = nightCount * 3;
  const portalById = new Map(portals.map(portal => [portal.id, portal]));
  const maximumArriving = Math.max(6, Math.min(24, Math.ceil(nightCount / 2)));
  let clockMinutes = cycleMinutes(Number.isFinite(minutes) ? minutes : 0);
  let nextArrivalAt = getTime() + 1.5, nextDepartureAt = getTime() + 1.25;
  let nextPortalAt = portals.map(() => getTime()), nextVisitor = nightCount;
  let presenceGate = null, protectedActorId = null;
  let stats = initialStats();
  const visible = () => actors.filter(actor => actor.active !== false);
  const counts = () => {
    const count = { active: 0, present: 0, arriving: 0, departing: 0, absent: 0 };
    for (const actor of actors) {
      if (actor.active !== false) count.active++;
      count[actor.populationPhase]++;
    }
    return count;
  };
  function record(type, actor) {
    stats.recent.push({ type, actorId: actor.id, portalId: actor._populationPortalId, at: getTime() });
    if (stats.recent.length > 48) stats.recent.shift();
  }
  const routePortal = (portal, entering) => ({ ...portal,
    approach: entering ? portal.enterApproach : portal.exitApproach });
  function clearJourney(actor) {
    actor._bridge = null; actor._entryBridge = null; actor._exitBridge = null;
    actor._exitLease = null; actor._liftId = null; actor._liftTarget = null;
    actor._journeyStarted = null; actor._viaCenter = false; actor._forcedCenter = false;
    actor._initialCenter = false; actor._departingDestination = null; actor._atDestination = null;
    actor._waitingSince = null; actor._stalledFor = 0; actor._velocity = 0; actor._headway = 1;
    actor._path = []; actor._pathIndex = 0; actor._after = null;
  }
  function resetAbsent(actor) {
    const portal = portalById.get(actor._populationPortalId) || portals[(actor.index - nightCount + portals.length * dayCount) % portals.length];
    clearJourney(actor);
    actor.active = false; actor.populationPhase = 'absent'; actor._populationPortalId = portal.id;
    actor.position = copyPoint(portal.position); actor.level = 0;
    actor.support = { type: 'garden', level: 0, zone: 'outer', surfaceY: portal.position.y - FLOOR_OFFSET };
    actor.footHeights = [portal.position.y, portal.position.y];
    actor.destination = null; actor.activity = 'away'; actor.purpose = words.purposes.away; actor.speed = 0;
    setState(actor, 'absent');
  }
  function portalAvailable(portal, index) {
    if (getTime() < nextPortalAt[index]) return false;
    // A returning visitor must finish entering the concealed chamber before
    // the same point is reused. Inactive roster members are not obstacles.
    return !actors.some(actor => actor.active !== false && distance(actor.position, portal.position) < 1.25);
  }
  function concealed(actor, portal) {
    return actor.id !== protectedActorId && (!presenceGate || presenceGate(actor, portal) === true);
  }
  function arrive(actor, portal, portalIndex) {
    if (!concealed(actor, portal)) return false;
    const destination = chooseDestination(actor);
    if (!destination) return false;
    const route = routing.toGallery(portal.position, routePortal(portal, true), destination);
    if (!route?.path?.length) return false;
    clearJourney(actor);
    actor.position = copyPoint(portal.position); actor.level = 0;
    actor.support = { type: 'garden', level: 0, zone: 'outer', surfaceY: portal.position.y - FLOOR_OFFSET };
    actor.footHeights = [portal.position.y, portal.position.y];
    const segment = route.path[0], end = segment.kind === 'line' ? segment.to : at(segment);
    actor.yaw = actor._desiredYaw = Math.atan2(end.x - actor.position.x, end.z - actor.position.z);
    actor.destination = destination; reserved.set(destination.id, actor.id);
    actor.active = true; actor.populationPhase = 'arriving'; actor._populationPortalId = portal.id;
    actor._journeyStarted = getTime(); actor.activity = 'walk'; actor.purpose = words.purposes.arriving; actor.speed = 0;
    setPath(actor, route.path, 'population_arrive');
    nextPortalAt[portalIndex] = getTime() + 4.5;
    stats.arrivalsStarted++; record('arrival-start', actor);
    return true;
  }
  function safeToLeave(actor, protectedActorId) {
    return actor.index >= nightCount && actor.active !== false && actor.populationPhase === 'present' &&
      actor.id !== protectedActorId && actor._state === 'idle' && actor.support?.type === 'floor' &&
      actor._atDestination && !actor._bookJobId && !actor._outdoorVisitId && !actor._lifeJobId && !actor._weatherJobId &&
      !actor._bridge && !actor._exitLease && actor._liftId == null;
  }
  function leave(actor) {
    const old = actor._atDestination;
    const portal = portals[(actor.index - nightCount) % portals.length];
    const path = []; let point = copyPoint(actor.position);
    for (const target of [...old.approach].reverse())
      point = appendLine(path, point, target, { type: 'floor', level: actor.level }, .85);
    actor.populationPhase = 'departing'; actor._populationPortalId = portal.id;
    actor._departingDestination = old; actor._atDestination = null; actor.destination = portal;
    actor._journeyStarted = getTime(); actor.activity = 'walk'; actor.purpose = words.purposes.leaving;
    setPath(actor, path, 'population_exit_route');
    stats.departuresStarted++; record('departure-start', actor);
  }
  function pathDone(actor, after) {
    if (after === 'population_arrive') {
      finish(actor); actor.populationPhase = 'present';
      stats.entered++; record('entered', actor); actor._populationPortalId = null;
      return true;
    }
    if (after === 'population_exit_route') {
      const old = actor._departingDestination;
      if (old && reserved.get(old.id) === actor.id) reserved.delete(old.id);
      actor._departingDestination = null;
      const portal = portalById.get(actor._populationPortalId);
      const route = routing.fromGallery(actor.position, actor.level, routePortal(portal, false));
      setPath(actor, route.path, 'population_depart');
      return true;
    }
    if (after === 'population_depart') {
      const portal = portalById.get(actor._populationPortalId);
      if (!concealed(actor, portal)) {
        setState(actor, 'population_wait'); actor.speed = 0; actor._velocity = 0;
        actor.activity = 'wait'; actor.purpose = words.purposes.pausing;
        return true;
      }
      // The walker is already at the physical hidden endpoint. No visible
      // position changes accompany the transition to an inactive identity.
      for (const [siteId, owner] of reserved) if (owner === actor.id) reserved.delete(siteId);
      stats.left++; record('left', actor);
      resetAbsent(actor);
      return true;
    }
    return false;
  }
  function update(input = {}) {
    presenceGate = typeof input.canChangePresence === 'function' ? input.canChangePresence : null;
    protectedActorId = input.protectedActorId || null;
    if (Number.isFinite(input.minutes)) clockMinutes = cycleMinutes(input.minutes);
    const target = populationTarget(clockMinutes, nightCount), count = counts(), now = getTime();
    const committed = count.active - count.departing;
    if (committed > target && now >= nextDepartureAt) {
      const candidate = actors.filter(actor => safeToLeave(actor, input.protectedActorId))
        .sort((a, b) => a.level - b.level || a.index - b.index)[0];
      if (candidate) leave(candidate);
      nextDepartureAt = now + (candidate ? 1.25 : .5);
    } else if (committed < target && count.active < dayCount && count.arriving < maximumArriving && now >= nextArrivalAt) {
      let admitted = false;
      for (let n = 0; n < dayCount - nightCount && !admitted; n++) {
        const index = nightCount + (nextVisitor - nightCount + n) % (dayCount - nightCount), actor = actors[index];
        if (actor.active !== false) continue;
        const first = (index - nightCount) % portals.length;
        for (let k = 0; k < portals.length && !admitted; k++) {
          const portalIndex = (first + k) % portals.length, portal = portals[portalIndex];
          if (portalAvailable(portal, portalIndex)) admitted = arrive(actor, portal, portalIndex);
        }
        if (admitted) nextVisitor = nightCount + (index - nightCount + 1) % (dayCount - nightCount);
      }
      nextArrivalAt = now + (admitted ? 1.5 : .5);
    }
    stats.maxActive = Math.max(stats.maxActive, visible().length);
  }
  function tickActor(actor, dt) {
    if (actor._state !== 'population_wait') return false;
    actor.speed = 0; actor._velocity = 0; actor._stalledFor = 0;
    const portal = portalById.get(actor._populationPortalId);
    if (concealed(actor, portal)) pathDone(actor, 'population_depart');
    return true;
  }
  function validPath(actor) {
    if (!Array.isArray(actor._path) || !actor._path.length || !Number.isInteger(actor._pathIndex) ||
      actor._pathIndex < 0 || actor._pathIndex >= actor._path.length) return false;
    for (const segment of actor._path) {
      if (!['line', 'arc', 'stairs', 'gardenArc', 'gardenStairs'].includes(segment.kind) ||
        !Number.isFinite(segment.length) || segment.length <= 0 || !Number.isFinite(segment.progress) ||
        segment.progress < 0 || segment.progress > segment.length + 1e-7 || !Number.isFinite(segment.speed) || segment.speed <= 0 ||
        !segment.support || !['floor', 'stairs', 'garden'].includes(segment.support.type)) return false;
      if (segment.kind === 'line' && (!finitePoint(segment.from) || !finitePoint(segment.to))) return false;
      if (['arc', 'stairs'].includes(segment.kind) && ![segment.radius, segment.fromAngle, segment.delta, segment.level].every(Number.isFinite)) return false;
      if (segment.kind === 'gardenArc' && ![segment.radius, segment.fromAngle, segment.delta, segment.surfaceY].every(Number.isFinite)) return false;
      if (segment.kind === 'gardenStairs' && (!segment.support.steps ||
        ![segment.fromRadius, segment.toRadius, segment.tangent, segment.support.steps.stepCount].every(Number.isFinite))) return false;
      if (segment.support.type === 'garden' && !Number.isFinite(segment.support.surfaceY)) return false;
    }
    const p = at(actor._path[actor._pathIndex]);
    return finitePoint(p) && distance(p, actor.position) < .001;
  }
  function validate(data, savedActors, savedReserved = []) {
    if (!data) return savedActors.length === nightCount && savedActors.every(actor => actor.active !== false && !actor.populationPhase);
    if (data.version !== 1 || data.nightCount !== nightCount || data.dayCount !== dayCount || savedActors.length !== dayCount ||
      !Number.isFinite(data.minutes) || !Number.isFinite(data.nextArrivalAt) || !Number.isFinite(data.nextDepartureAt) ||
      !Array.isArray(data.nextPortalAt) || data.nextPortalAt.length !== portals.length || !data.nextPortalAt.every(Number.isFinite) ||
      !Number.isInteger(data.nextVisitor) || data.nextVisitor < nightCount || data.nextVisitor >= dayCount || !data.stats) return false;
    const reservations = new Map(savedReserved);
    for (let index = 0; index < savedActors.length; index++) {
      const actor = savedActors[index];
      if (!actor || actor.id !== actors[index].id || actor.index !== index || !finitePoint(actor.position) ||
        !phases.includes(actor.populationPhase) || typeof actor.active !== 'boolean' ||
        actor.active !== (actor.populationPhase !== 'absent') || (index < nightCount && actor.populationPhase !== 'present')) return false;
      if (actor.populationPhase !== 'present') {
        const portal = portalById.get(actor._populationPortalId);
        if (!portal || actor._bookJobId || actor._outdoorVisitId || actor._lifeJobId || actor._bridge || actor._exitLease || actor._liftId != null) return false;
        if (actor.populationPhase === 'absent') {
          if (actor._state !== 'absent' || actor._weatherJobId || actor.destination || actor._atDestination || actor._path?.length ||
            distance(actor.position, portal.position) > 1e-6 || savedReserved.some(([, id]) => id === actor.id)) return false;
        } else {
          const waitingAtPortal = actor.populationPhase === 'departing' && actor._state === 'population_wait' &&
            distance(actor.position, portal.position) < .001 && actor._path?.length === 0;
          // An umbrella errand holds the walker's own path aside until the umbrella is stored or taken.
          const onUmbrellaErrand = !!actor._weatherJobId;
          if (!waitingAtPortal && !onUmbrellaErrand && (actor._state !== 'path' || !validPath(actor))) return false;
          if (actor.populationPhase === 'arriving' && (!onUmbrellaErrand && actor._after !== 'population_arrive' ||
            !actor.destination || reservations.get(actor.destination.id) !== actor.id)) return false;
          if (actor.populationPhase === 'departing' && (!onUmbrellaErrand && !['population_exit_route', 'population_depart'].includes(actor._after) ||
            actor.destination?.id !== portal.id)) return false;
        }
      }
    }
    return true;
  }
  function snapshot() { return clone({ version: 1, nightCount, dayCount, minutes: clockMinutes,
    nextArrivalAt, nextDepartureAt, nextPortalAt, nextVisitor, stats }); }
  function restore(data) {
    if (!data) {
      nextArrivalAt = getTime() + 1.5; nextDepartureAt = getTime() + 1.25;
      nextPortalAt = portals.map(() => getTime()); nextVisitor = nightCount;
      stats = initialStats(); stats.legacyMigrations = 1;
    } else {
      const saved = clone(data); clockMinutes = saved.minutes;
      nextArrivalAt = saved.nextArrivalAt; nextDepartureAt = saved.nextDepartureAt;
      nextPortalAt = saved.nextPortalAt; nextVisitor = saved.nextVisitor; stats = saved.stats;
    }
  }
  return { update, tickActor, pathDone, validate, snapshot, restore, resetAbsent,
    get state() { return { nightCount, dayCount, target: populationTarget(clockMinutes, nightCount),
      minutes: clockMinutes, maximumArriving, ...counts(), ...clone(stats) }; },
  };
}
