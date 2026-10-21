import { clamp, polar, shortest } from './mechanism.js';
import { FLOOR_OFFSET, appendLine, copyPoint, distance, floorY, outerRoute, segmentPoint as floorSegmentPoint } from './navigation.js';
import { words } from './words.js';

const clone = value => JSON.parse(JSON.stringify(value));
const finitePoint = p => p && [p.x, p.y, p.z].every(Number.isFinite);
const pathLength = path => path.reduce((sum, s) => sum + s.length, 0);
const feetPoint = p => ({ ...p, y: p.y + FLOOR_OFFSET });
const radiusOf = p => Math.hypot(p.x, p.z);
const groundSupport = (zone, surfaceY, extra = {}) => ({ type: 'garden', level: 0, zone, surfaceY, ...extra });

/** Exact top of the real, radial rectangular treads, in either direction. */
export function gardenStepHeight(steps, x, z) {
  const r = x * Math.sin(steps.angle) - z * Math.cos(steps.angle);
  const fraction = steps.ascendingOutward
    ? (steps.outerRadius - r) / (steps.outerRadius - steps.innerRadius)
    : (r - steps.innerRadius) / (steps.outerRadius - steps.innerRadius);
  const tread = Math.min(steps.stepCount, Math.max(0, Math.ceil(clamp(fraction) * steps.stepCount - 1e-8)));
  return steps.topY + (steps.bottomY - steps.topY) * tread / steps.stepCount;
}

export function sampleGardenGround(x, z, support) {
  if (support.steps) return gardenStepHeight(support.steps, x, z) + FLOOR_OFFSET;
  if (support.ramp) {
    const { from, to } = support.ramp, dx = to.x - from.x, dz = to.z - from.z;
    const t = clamp(((x - from.x) * dx + (z - from.z) * dz) / Math.max(1e-8, dx * dx + dz * dz));
    return from.y + (to.y - from.y) * t + FLOOR_OFFSET;
  }
  return support.surfaceY + FLOOR_OFFSET;
}

/** Return null for the unchanged indoor path types. */
export function outdoorSegmentPoint(s, at = s.progress) {
  const t = s.length > 1e-8 ? clamp(at / s.length) : 1;
  if (s.kind === 'gardenArc') return polar(s.radius, s.fromAngle + s.delta * t, s.surfaceY + FLOOR_OFFSET);
  if (s.kind === 'gardenStairs') {
    const r = s.fromRadius + (s.toRadius - s.fromRadius) * t, a = s.support.steps.angle;
    const p = { x: Math.sin(a) * r + Math.cos(a) * s.tangent, y: 0,
      z: -Math.cos(a) * r + Math.sin(a) * s.tangent };
    p.y = sampleGardenGround(p.x, p.z, s.support); return p;
  }
  return null;
}
const pointOn = (s, at) => outdoorSegmentPoint(s, at) || floorSegmentPoint(s, at);
const tail = (path, fallback) => path.length ? pointOn(path[path.length - 1], path[path.length - 1].length) : fallback;

/** The rendered landscape and these routes use the same immutable metre plan. */
export function createOutdoorRouting(plan) {
  if (!plan || !Array.isArray(plan.nodes) || !Array.isArray(plan.edges) || !Array.isArray(plan.destinations) || !Array.isArray(plan.entrances))
    throw new Error('A complete garden plan is required');
  const nodes = new Map(plan.nodes.map(n => [n.id, n]));
  const destinations = plan.destinations.map(d => ({ ...d, position: feetPoint(d.position),
    approach: d.approach.map(feetPoint), ...(d.posePosition ? { posePosition: feetPoint(d.posePosition) } : {}) }));
  const byId = new Map(destinations.map(d => [d.id, d]));
  const links = new Map(plan.nodes.map(n => [n.id, []]));
  const lengthOf = edge => edge.kind === 'arc' ? Math.abs(edge.delta) * edge.radius
    : distance(nodes.get(edge.from).position, nodes.get(edge.to).position);
  for (const edge of plan.edges) {
    if (!nodes.has(edge.from) || !nodes.has(edge.to)) throw new Error('Unknown garden path endpoint');
    const length = lengthOf(edge);
    links.get(edge.from).push({ edge, reverse: false, to: edge.to, length });
    links.get(edge.to).push({ edge, reverse: true, to: edge.from, length });
  }
  const routeCache = new Map();
  function graphRoute(fromId, toId, zone) {
    const key = `${zone}:${fromId}:${toId}`;
    if (routeCache.has(key)) return routeCache.get(key);
    const allowed = new Set(zone === 'court' ? ['court', 'gallery'] : ['outer', 'entrance', 'gallery']);
    const costs = new Map([[fromId, 0]]), prior = new Map(), visited = new Set();
    while (true) {
      let current = null, best = Infinity;
      for (const [id, cost] of costs) if (!visited.has(id) && cost < best) { best = cost; current = id; }
      if (current === null || current === toId) break;
      visited.add(current);
      for (const link of links.get(current)) {
        if (!allowed.has(nodes.get(link.to).zone)) continue;
        const candidate = best + link.length;
        if (candidate < (costs.get(link.to) ?? Infinity) - 1e-8) {
          costs.set(link.to, candidate); prior.set(link.to, { from: current, ...link });
        }
      }
    }
    if (!costs.has(toId)) throw new Error(`No supported garden route: ${fromId} → ${toId}`);
    const result = []; let id = toId;
    while (id !== fromId) { const leg = prior.get(id); result.unshift(leg); id = leg.from; }
    routeCache.set(key, result); return result;
  }
  function legInfo(leg) {
    const edge = leg.edge;
    const from = nodes.get(leg.from), to = nodes.get(leg.to);
    return { edge, from, to, delta: edge.kind === 'arc' ? edge.delta * (leg.reverse ? -1 : 1) : 0,
      fromAngle: edge.kind === 'arc' ? edge.fromAngle + (leg.reverse ? edge.delta : 0) : from.angle };
  }
  function addGroundLine(path, from, to, zone, surfaceFrom = from.y - FLOOR_OFFSET, surfaceTo = to.y - FLOOR_OFFSET, speed = .90) {
    const support = groundSupport(zone, surfaceFrom, Math.abs(surfaceFrom - surfaceTo) > 1e-7 ? {
      ramp: { from: { ...from, y: surfaceFrom }, to: { ...to, y: surfaceTo } },
    } : {});
    return appendLine(path, from, to, support, speed);
  }
  function compile(legs, current) {
    const path = []; let p = copyPoint(current);
    for (let i = 0; i < legs.length;) {
      const first = legInfo(legs[i]);
      if (first.edge.kind === 'arc') {
        let delta = first.delta, last = first, j = i + 1;
        while (j < legs.length) {
          const next = legInfo(legs[j]);
          if (next.edge.kind !== 'arc' || next.edge.radius !== first.edge.radius || next.edge.surfaceY !== first.edge.surfaceY ||
            Math.sign(next.delta) !== Math.sign(first.delta)) break;
          delta += next.delta; last = next; j++;
        }
        const radius = first.edge.radius + (delta >= 0 ? .27 : -.27), surfaceY = first.edge.surfaceY;
        const begin = polar(radius, first.fromAngle, surfaceY + FLOOR_OFFSET);
        p = addGroundLine(path, p, begin, first.edge.zone);
        const segment = { kind: 'gardenArc', radius, fromAngle: first.fromAngle, delta, surfaceY,
          length: radius * Math.abs(delta), progress: 0, speed: .94, support: groundSupport(first.edge.zone, surfaceY) };
        if (segment.length > 1e-8) path.push(segment);
        p = pointOn(segment, segment.length);
        p = addGroundLine(path, p, feetPoint(last.to.position), first.edge.zone);
        i = j; continue;
      }
      // Merge collinear radial legs before assigning the two walking lanes.
      // A staircase does not make the resident return to the centre of the lane.
      const dx = first.to.position.x - first.from.position.x, dz = first.to.position.z - first.from.position.z;
      const length = Math.hypot(dx, dz), ux = dx / length, uz = dz / length;
      const offset = { x: uz * .27, z: -ux * .27 };
      let j = i + 1;
      while (j < legs.length) {
        const next = legInfo(legs[j]), nx = next.to.position.x - next.from.position.x, nz = next.to.position.z - next.from.position.z;
        if (next.edge.kind === 'arc' || nx * ux + nz * uz <= 0 || Math.abs(nx * uz - nz * ux) > 1e-6) break;
        j++;
      }
      const offsetPoint = raw => ({ x: raw.x + offset.x, y: raw.y + FLOOR_OFFSET, z: raw.z + offset.z });
      p = addGroundLine(path, p, offsetPoint(first.from.position), first.edge.zone);
      for (let k = i; k < j; k++) {
        const leg = legInfo(legs[k]), edge = leg.edge, end = offsetPoint(leg.to.position);
        if (edge.kind === 'stairs') {
          const steps = { ...edge.steps }, angle = steps.angle;
          const tangent = offset.x * Math.cos(angle) + offset.z * Math.sin(angle);
          const segment = { kind: 'gardenStairs', fromRadius: leg.from.radius, toRadius: leg.to.radius, tangent,
            length: Math.abs(leg.to.radius - leg.from.radius), progress: 0, speed: .68,
            support: groundSupport(edge.zone, leg.from.position.y, { steps }) };
          path.push(segment); p = pointOn(segment, segment.length);
        } else p = addGroundLine(path, p, end, edge.zone, leg.from.position.y, leg.to.position.y);
      }
      const last = legInfo(legs[j - 1]);
      p = addGroundLine(path, p, feetPoint(last.to.position), last.edge.zone);
      i = j;
    }
    return path;
  }
  function spur(from, destination, reverse = false) {
    const path = []; let p = copyPoint(from);
    const points = reverse ? [...destination.approach].reverse() : destination.approach;
    for (const next of points) p = addGroundLine(path, p, next, destination.zone, destination.position.y - FLOOR_OFFSET,
      destination.position.y - FLOOR_OFFSET, .82);
    return path;
  }
  function fromGallery(from, level, destination) {
    let best = null;
    for (const entry of plan.entrances) {
      const indoor = outerRoute(from, level, 0, entry.angle);
      const start = tail(indoor, from), legs = graphRoute(entry.galleryNode, destination.nodeId, destination.zone);
      const garden = compile(legs, start), approach = spur(tail(garden, start), destination);
      const path = [...indoor, ...garden, ...approach], length = pathLength(path);
      if (!best || length < best.length) best = { path, length, entryId: entry.id };
    }
    return best;
  }
  function between(from, origin, destination) {
    const departure = spur(from, origin, true), start = tail(departure, from);
    const route = compile(graphRoute(origin.nodeId, destination.nodeId, destination.zone), start);
    const path = [...departure, ...route, ...spur(tail(route, start), destination)];
    return { path, length: pathLength(path) };
  }
  function toGallery(from, origin, destination) {
    let best = null;
    const departure = spur(from, origin, true), start = tail(departure, from);
    for (const entry of plan.entrances) {
      const garden = compile(graphRoute(origin.nodeId, entry.galleryNode, origin.zone), start);
      const indoorStart = tail(garden, start), indoor = outerRoute(indoorStart, 0, destination.level, destination.angle);
      const approach = []; let p = tail(indoor, indoorStart);
      for (const next of destination.approach) p = appendLine(approach, p, next, { type: 'floor', level: destination.level }, .83);
      appendLine(approach, p, destination.position, { type: 'floor', level: destination.level }, .72);
      const path = [...departure, ...garden, ...indoor, ...approach], length = pathLength(path);
      if (!best || length < best.length) best = { path, length, entryId: entry.id };
    }
    return best;
  }
  let fingerprint = 2166136261;
  for (const char of JSON.stringify({ nodes: plan.nodes, edges: plan.edges, destinations: plan.destinations }))
    fingerprint = Math.imul(fingerprint ^ char.charCodeAt(0), 16777619) >>> 0;
  return { plan, destinations, byId, fromGallery, between, toGallery, graphRoute, compile,
    signature: `${plan.version}:${fingerprint.toString(16)}` };
}

const initialStats = () => ({ scheduled: 0, completed: 0, exits: 0, entries: 0, courtEntries: 0, courtReturns: 0,
  outerStops: 0, courtStops: 0, maxActive: 0, longestVisit: 0, firstExitAt: null, firstEntryAt: null, rainReturns: 0, recent: [] });

/** A bounded promenade, owned by one existing idle resident from start to end.
 * A visit keeps its old indoor place and two garden stops reserved. It never
 * competes with a bridge seam, lift reservation or active book-collecting job. */
export function createOutdoorVisits({ actors, routing, getTime, getSite, setPath, setState, still, turn, finish, seed = 1, getPopulationCount = null, getWeather = null }) {
  const maximum = getPopulationCount ? 8 : Math.max(2, Math.min(8, Math.ceil(actors.length / 6)));
  const currentMaximum = () => getPopulationCount ? Math.max(2, Math.min(8, Math.ceil(getPopulationCount() / 6))) : maximum;
  const actorById = new Map(actors.map(a => [a.id, a]));
  let sequence = 0, nextAt = getTime() + 2.5, randomState = (seed ^ 0xA1B42D77) >>> 0;
  let visits = [], cooldowns = {}, stats = initialStats();
  const random = () => { randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0; return randomState / 4294967296; };
  const busyStops = () => new Set(visits.flatMap(v => v.stops));
  function show(actor, visit) {
    actor.outdoorTask = { id: visit.id, phase: visit.phase, zone: visit.zone, stop: visit.stopIndex + 1, stops: visit.stops.length };
  }
  function choose(actor) {
    const busy = busyStops(), preferred = sequence % 3 === 1 ? 'court' : 'outer';
    for (const zone of [preferred, preferred === 'outer' ? 'court' : 'outer']) {
      const candidates = routing.destinations.filter(d => d.zone === zone && !busy.has(d.id));
      if (candidates.length < 2) continue;
      let best = null;
      for (const d of candidates) {
        const route = routing.fromGallery(actor._atDestination.approach[0], actor.level, d);
        const score = route.length + random() * 17 - (d.roleTags?.includes(actor.roleTag) ? 7 : 0);
        if (!best || score < best.score) best = { destination: d, route, score };
      }
      let second = null;
      for (const d of candidates) {
        if (d.id === best.destination.id) continue;
        const route = routing.between(best.destination.position, best.destination, d);
        const score = route.length + random() * 12 - (d.activity !== best.destination.activity ? 6 : 0);
        if (!second || score < second.score) second = { destination: d, score };
      }
      return { ...best, second: second.destination };
    }
    return null;
  }
  function launch(actor, choice) {
    const origin = actor._atDestination, now = getTime();
    const visit = { id: `garden-visit-${++sequence}`, actorId: actor.id, originId: origin.id,
      zone: choice.destination.zone, phase: 'outbound', startedAt: now, phaseStartedAt: now,
      stops: [choice.destination.id, choice.second.id], stopIndex: 0, dwellUntil: null,
      entryId: choice.route.entryId, exitId: null, outside: false, inCourt: false };
    const departure = []; let p = copyPoint(actor.position);
    for (const target of [...origin.approach].reverse()) p = appendLine(departure, p, target, { type: 'floor', level: actor.level }, .85);
    const route = routing.fromGallery(p, actor.level, choice.destination);
    visits.push(visit); actor._outdoorVisitId = visit.id; actor._atDestination = null;
    actor.destination = choice.destination; actor._initialCenter = false; actor._journeyStarted = now; actor._viaCenter = false;
    actor.purpose = choice.destination.zone === 'outer' ? words.purposes.toGardens : words.purposes.toCourt;
    show(actor, visit); setPath(actor, [...departure, ...route.path], 'outdoor_arrive');
    stats.scheduled++; stats.maxActive = Math.max(stats.maxActive, visits.length);
  }
  function update() {
    const now = getTime();
    // Rain and snow start no new walks, and walks already outside are asked to come back under cover.
    if (getWeather?.()?.wet) {
      for (const visit of visits) if (visit.phase !== 'return') visit.rainRequested = true;
      return;
    }
    if (now < nextAt) return;
    nextAt = now + 4;
    if (visits.length >= currentMaximum()) return;
    const candidates = actors.filter(a => a.active !== false && !['arriving', 'departing'].includes(a.populationPhase) &&
      a._state === 'idle' && a.support?.type === 'floor' && a._atDestination &&
      !a._bookJobId && !a._outdoorVisitId && !a._lifeJobId && !a._weatherJobId && !a._bridge && !a._exitLease && !a._liftId && !(cooldowns[a.id] > now));
    if (!candidates.length) return;
    const ranked = candidates.map(a => ({ actor: a, score: random() * 7 + (a.roleTag === 'visitor' ? 2 : 0) - a.level * .7 +
      (sequence < 3 && a.level === 0 ? 30 : 0) })).sort((a, b) => b.score - a.score || a.actor.index - b.actor.index);
    const actor = ranked[0].actor, choice = choose(actor);
    if (!choice) return;
    launch(actor, choice); nextAt = now + 10 + random() * 7;
  }
  function recordCrossing(actor, visit) {
    const now = getTime(), radius = radiusOf(actor.position);
    if (visit.zone === 'outer') {
      if (!visit.outside && visit.phase === 'outbound' && radius > 34.60 && actor.position.y < 0) {
        visit.outside = true; stats.exits++; stats.firstExitAt ??= now;
      }
      if (visit.outside && visit.phase === 'return' && radius < 32.40 && actor.position.y >= 0) {
        visit.outside = false; stats.entries++; stats.firstEntryAt ??= now;
      }
    } else {
      if (!visit.inCourt && visit.phase === 'outbound' && radius < 19.50 && actor.position.y < -.5) {
        visit.inCourt = true; stats.courtEntries++;
      }
      if (visit.inCourt && visit.phase === 'return' && radius > 22.30 && actor.position.y >= 0) {
        visit.inCourt = false; stats.courtReturns++;
      }
    }
  }
  function beginReturn(actor, visit) {
    const origin = getSite(visit.originId), at = routing.byId.get(visit.stops[visit.stopIndex]);
    const route = routing.toGallery(actor.position, at, origin);
    visit.phase = 'return'; visit.phaseStartedAt = getTime(); visit.exitId = route.entryId;
    actor.destination = origin; actor.purpose = words.purposes.backFromWalk(origin.label);
    show(actor, visit); setPath(actor, route.path, 'outdoor_finish');
  }
  /** A resident caught by rain at a garden stop closes the book first, then walks back. */
  function closeForRain(actor, visit) {
    visit.phase = 'rain_close'; visit.phaseStartedAt = getTime(); visit.rainRequested = true;
    actor.purpose = words.purposes.rainClose;
    setState(actor, 'outdoor_rain'); show(actor, visit); stats.rainReturns++;
  }
  function pathDone(actor, after) {
    if (!['outdoor_arrive', 'outdoor_finish'].includes(after)) return false;
    const visit = visits.find(v => v.id === actor._outdoorVisitId); if (!visit) return false;
    recordCrossing(actor, visit);
    if (after === 'outdoor_finish') {
      const now = getTime(); stats.completed++; stats.longestVisit = Math.max(stats.longestVisit, now - visit.startedAt);
      stats.recent.push({ id: visit.id, actorId: actor.id, zone: visit.zone, seconds: now - visit.startedAt,
        stops: [...visit.stops], entryId: visit.entryId, exitId: visit.exitId });
      if (stats.recent.length > 16) stats.recent.shift();
      cooldowns[actor.id] = now + 90 + random() * 100;
      visits = visits.filter(v => v !== visit); delete actor._outdoorVisitId; delete actor.outdoorTask;
      finish(actor); return true;
    }
    const site = routing.byId.get(visit.stops[visit.stopIndex]);
    visit.phase = 'dwell'; visit.phaseStartedAt = getTime(); visit.dwellUntil = getTime() + 19 + random() * 27;
    actor.destination = site; actor.support = groundSupport(site.zone, site.position.y - FLOOR_OFFSET);
    actor.purpose = words.purposes.stay(site.activity === 'sit' ? 'sit' : 'observe', site.label);
    actor.level = 0; still(actor, 0, site.activity); setState(actor, 'outdoor_dwell'); show(actor, visit);
    stats[site.zone === 'outer' ? 'outerStops' : 'courtStops']++;
    if (visit.rainRequested || getWeather?.()?.wet) closeForRain(actor, visit);
    return true;
  }
  function tickActor(actor, dt) {
    if (!actor._outdoorVisitId) return false;
    const visit = visits.find(v => v.id === actor._outdoorVisitId); if (!visit) return false;
    recordCrossing(actor, visit);
    if (visit.phase === 'rain_close') {
      const site = routing.byId.get(visit.stops[visit.stopIndex]);
      still(actor, dt, 'observe'); turn(actor, site.facing ?? actor.yaw, dt);
      if (getTime() - visit.phaseStartedAt >= 2.3) beginReturn(actor, visit);
      return true;
    }
    if (visit.phase !== 'dwell') return false;
    if (visit.rainRequested || getWeather?.()?.wet) { closeForRain(actor, visit); return true; }
    const site = routing.byId.get(visit.stops[visit.stopIndex]);
    still(actor, dt, site.activity); turn(actor, site.facing ?? actor.yaw, dt);
    if (getTime() < visit.dwellUntil) return true;
    if (visit.stopIndex + 1 >= visit.stops.length) beginReturn(actor, visit);
    else {
      const next = routing.byId.get(visit.stops[++visit.stopIndex]), route = routing.between(actor.position, site, next);
      visit.phase = 'stroll'; visit.phaseStartedAt = getTime(); actor.destination = next;
      actor.purpose = words.purposes.alongPath(next.label); show(actor, visit); setPath(actor, route.path, 'outdoor_arrive');
    }
    return true;
  }
  function snapshot() { return clone({ version: 1, signature: routing.signature, sequence, nextAt, randomState, visits, cooldowns, stats }); }
  function validate(data, savedActors, reserved = []) {
    if (!data) return !savedActors.some(a => a._outdoorVisitId || a._state === 'outdoor_dwell' || a._state === 'outdoor_rain');
    if (data.version !== 1 || data.signature !== routing.signature || !Number.isInteger(data.sequence) || data.sequence < 0 ||
      !Number.isFinite(data.nextAt) || !Number.isInteger(data.randomState) || !Array.isArray(data.visits) || data.visits.length > maximum ||
      !data.cooldowns || !data.stats || typeof data.stats !== 'object') return false;
    const reservations = new Map(reserved), seen = new Set(), assigned = new Set(), stops = new Set();
    for (const v of data.visits) {
      if (!v || typeof v.id !== 'string' || seen.has(v.id) || assigned.has(v.actorId) || !['outer', 'court'].includes(v.zone) ||
        !['outbound', 'dwell', 'stroll', 'return', 'rain_close'].includes(v.phase) || !Number.isFinite(v.startedAt) || !Number.isFinite(v.phaseStartedAt) ||
        !Array.isArray(v.stops) || v.stops.length !== 2 || !Number.isInteger(v.stopIndex) || v.stopIndex < 0 || v.stopIndex > 1 ||
        v.stops.some(id => !routing.byId.has(id) || stops.has(id) || routing.byId.get(id).zone !== v.zone) || new Set(v.stops).size !== 2 ||
        !getSite(v.originId) || reservations.get(v.originId) !== v.actorId) return false;
      const actor = savedActors.find(a => a.id === v.actorId);
      const expectedDestination = v.phase === 'return' ? v.originId : v.stops[v.stopIndex];
      if (!actor || actor._outdoorVisitId !== v.id || actor.outdoorTask?.id !== v.id || actor._bookJobId || actor._lifeJobId || actor._atDestination ||
        actor.destination?.id !== expectedDestination || !finitePoint(actor.position)) return false;
      if (v.phase === 'dwell' || v.phase === 'rain_close') {
        if (actor._state !== (v.phase === 'dwell' ? 'outdoor_dwell' : 'outdoor_rain') || actor.support?.type !== 'garden' || !Number.isFinite(v.dwellUntil)) return false;
      } else if (actor._weatherJobId) {
        // An umbrella errand pauses the walk's path and restores it afterwards, so only the phase is checked here.
        if (!['outbound', 'stroll', 'return'].includes(v.phase)) return false;
      } else {
        if (actor._state !== 'path' || actor._after !== (v.phase === 'return' ? 'outdoor_finish' : 'outdoor_arrive') ||
          !Array.isArray(actor._path) || !actor._path.length || !Number.isInteger(actor._pathIndex) || actor._pathIndex < 0 || actor._pathIndex >= actor._path.length) return false;
        for (const s of actor._path) {
          if (!['line', 'arc', 'stairs', 'gardenArc', 'gardenStairs'].includes(s.kind) || !Number.isFinite(s.length) || s.length < 0 ||
            !Number.isFinite(s.progress) || s.progress < 0 || s.progress > s.length + 1e-7 || !Number.isFinite(s.speed) || s.speed <= 0) return false;
          if (!s.support || !['floor', 'stairs', 'garden'].includes(s.support.type)) return false;
          if (s.support.type === 'garden' && (!['court', 'outer', 'entrance', 'gallery'].includes(s.support.zone) || !Number.isFinite(s.support.surfaceY))) return false;
          if (s.kind === 'line' && (!finitePoint(s.from) || !finitePoint(s.to))) return false;
          if (s.kind === 'gardenStairs') {
            const steps = s.support.steps;
            if (!steps || !['innerRadius', 'outerRadius', 'stepCount', 'topY', 'bottomY', 'angle', 'run', 'rise'].every(k => Number.isFinite(steps[k])) ||
              !Number.isInteger(steps.stepCount) || steps.stepCount < 1 || steps.stepCount > 30 || steps.outerRadius <= steps.innerRadius ||
              typeof steps.ascendingOutward !== 'boolean' || !Number.isFinite(s.fromRadius) || !Number.isFinite(s.toRadius) || !Number.isFinite(s.tangent)) return false;
          }
          if (s.kind === 'gardenArc' && (![s.radius, s.fromAngle, s.delta, s.surfaceY].every(Number.isFinite) || s.radius < 10)) return false;
          if (['arc', 'stairs'].includes(s.kind) && ![s.radius, s.fromAngle, s.delta, s.level].every(Number.isFinite)) return false;
          if (s.support.ramp && (!finitePoint(s.support.ramp.from) || !finitePoint(s.support.ramp.to))) return false;
        }
        const p = pointOn(actor._path[actor._pathIndex]);
        if (!finitePoint(p) || Math.hypot(p.x - actor.position.x, p.y - actor.position.y, p.z - actor.position.z) > .001) return false;
      }
      seen.add(v.id); assigned.add(v.actorId); for (const id of v.stops) stops.add(id);
    }
    return savedActors.every(a => !a._outdoorVisitId || assigned.has(a.id));
  }
  function restore(data) {
    if (!data) { sequence = 0; nextAt = getTime() + 2.5; randomState = (seed ^ 0xA1B42D77) >>> 0; visits = []; cooldowns = {}; stats = initialStats(); return; }
    const saved = clone(data); sequence = saved.sequence; nextAt = saved.nextAt; randomState = saved.randomState;
    visits = saved.visits; cooldowns = saved.cooldowns; stats = { ...initialStats(), ...saved.stats };
    for (const visit of visits) show(actorById.get(visit.actorId), visit);
  }
  return { update, tickActor, pathDone, snapshot, validate, restore,
    get visits() { return visits; }, get stats() { return { ...stats, active: visits.length, maximum: currentMaximum() }; } };
}
