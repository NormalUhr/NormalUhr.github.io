import { clamp, damp, TAU, wrapAngle } from './math.js';
import { createWildlifeModels, WILDLIFE_SCALE } from './wildlife_models.js';

/*
 * Finite, persisted animal visits. Each visible touch asks the garden to record
 * a small repairable incident; this module never owns or redraws the sand.
 * Wall-clock lighting previews do not advance this independent simulation.
 */
const STATES = new Set(['sleep', 'approach', 'inspect', 'scratch', 'rest', 'roam', 'alert', 'retreat', 'depart']);
const VISIT_ORDER = [0, 1, 2, 0, 1, 3];
const QUIET_SECONDS = { firstMin: 45, firstRange: 45, betweenMin: 180, betweenRange: 120 };
const PROFILES = [
  { id: 0, species: 'cat', origin: { x: 12.0, z: 2.3 }, speed: .58, radius: .24 },
  { id: 1, species: 'rabbit', origin: { x: -11.8, z: 2.6 }, speed: .66, radius: .22 },
  { id: 2, species: 'sparrow', origin: { x: 10.9, z: -2.7 }, speed: .40, radius: .105 },
  { id: 3, species: 'sparrow', origin: { x: 11.6, z: -.9 }, speed: .43, radius: .105 },
].map(p => ({ ...p, speed: p.speed * WILDLIFE_SCALE[p.species], radius: p.radius * WILDLIFE_SCALE[p.species] }));
const SPOTS = [
  { x: .86, z: .63 }, { x: -1.12, z: .46 }, { x: .25, z: -1.12 },
  { x: 1.34, z: -.46 }, { x: -.56, z: 1.28 }, { x: -.77, z: -.74 },
];
const finitePoint = p => p && Number.isFinite(p.x) && Number.isFinite(p.z) && Math.abs(p.x) < 15 && Math.abs(p.z) < 12;
const distance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);

/** Physical-obstacle routing, independent of the humans' protected grooves. */
class AnimalNavigation {
  constructor(obstacles, radius) {
    this.obstacles = obstacles; this.radius = radius;
    this.minX = -12.9; this.minZ = -10.2; this.cell = .30; this.nx = 87; this.nz = 69;
  }
  free(x, z) {
    return Number.isFinite(x) && Number.isFinite(z) && x > -12.8 && x < 12.8 && z > -10.1 && z < 10.1 &&
      !(x > -7.28 - this.radius && x < -1.0 + this.radius && z > -9.35 - this.radius && z < -4.05 + this.radius) &&
      !this.obstacles.some(o => Math.hypot(x - o.x, z - o.z) < o.r + this.radius);
  }
  visible(a, b) {
    if (!this.free(a.x, a.z) || !this.free(b.x, b.z)) return false;
    const dx = b.x - a.x, dz = b.z - a.z, d2 = dx * dx + dz * dz;
    for (const o of this.obstacles) {
      const t = d2 ? clamp(((o.x - a.x) * dx + (o.z - a.z) * dz) / d2) : 0;
      if (Math.hypot(a.x + t * dx - o.x, a.z + t * dz - o.z) < o.r + this.radius) return false;
    }
    // The cottage is a rectangle, including its projecting porch, not a circle.
    let lo = 0, hi = 1;
    for (const [p, d, low, high] of [[a.x, dx, -7.28 - this.radius, -1.0 + this.radius], [a.z, dz, -9.35 - this.radius, -4.05 + this.radius]]) {
      if (Math.abs(d) < 1e-9) { if (p < low || p > high) return true; }
      else { const t0 = (low - p) / d, t1 = (high - p) / d; lo = Math.max(lo, Math.min(t0, t1)); hi = Math.min(hi, Math.max(t0, t1)); }
    }
    return lo > hi || hi < 0 || lo > 1;
  }
  route(a, b) {
    if (!this.free(a.x, a.z) || !this.free(b.x, b.z)) return [];
    if (this.visible(a, b)) return [{ x: b.x, z: b.z }];
    const point = i => ({ x: this.minX + i % this.nx * this.cell, z: this.minZ + Math.floor(i / this.nx) * this.cell });
    const attach = p => {
      const x = clamp(Math.round((p.x - this.minX) / this.cell), 0, this.nx - 1), z = clamp(Math.round((p.z - this.minZ) / this.cell), 0, this.nz - 1);
      let best = -1, d2 = Infinity;
      for (let ring = 0; ring < 5; ring++) {
        for (let iz = Math.max(0, z - ring); iz <= Math.min(this.nz - 1, z + ring); iz++) for (let ix = Math.max(0, x - ring); ix <= Math.min(this.nx - 1, x + ring); ix++) {
          const i = iz * this.nx + ix, q = point(i), d = (p.x - q.x) ** 2 + (p.z - q.z) ** 2;
          if (d < d2 && this.visible(p, q)) { best = i; d2 = d; }
        }
        if (best >= 0 && ring * this.cell > Math.sqrt(d2) + this.cell) break;
      }
      return best;
    };
    const start = attach(a), goal = attach(b);
    if (start < 0 || goal < 0) return [];
    const count = this.nx * this.nz, cost = new Float64Array(count).fill(Infinity), previous = new Int32Array(count).fill(-1), closed = new Uint8Array(count), queued = new Uint8Array(count);
    const open = [start], gx = goal % this.nx, gz = Math.floor(goal / this.nx);
    const heuristic = i => Math.hypot(i % this.nx - gx, Math.floor(i / this.nx) - gz);
    cost[start] = 0; queued[start] = 1;
    let reached = false;
    while (open.length) {
      let choice = 0;
      for (let i = 1; i < open.length; i++) if (cost[open[i]] + heuristic(open[i]) < cost[open[choice]] + heuristic(open[choice])) choice = i;
      const current = open.splice(choice, 1)[0]; queued[current] = 0;
      if (current === goal) { reached = true; break; }
      closed[current] = 1;
      const cx = current % this.nx, cz = Math.floor(current / this.nx), here = point(current);
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const x = cx + dx, z = cz + dz;
        if (x < 0 || x >= this.nx || z < 0 || z >= this.nz) continue;
        const next = z * this.nx + x;
        if (closed[next] || !this.visible(here, point(next))) continue;
        const g = cost[current] + Math.hypot(dx, dz);
        if (g >= cost[next]) continue;
        cost[next] = g; previous[next] = current;
        if (!queued[next]) { queued[next] = 1; open.push(next); }
      }
    }
    if (!reached) return [];
    const chain = [{ x: b.x, z: b.z }];
    for (let i = goal; i >= 0; i = previous[i]) { chain.push(point(i)); if (i === start) break; }
    chain.push({ x: a.x, z: a.z }); chain.reverse();
    const route = [];
    for (let at = 0; at < chain.length - 1;) {
      let next = chain.length - 1;
      while (next > at + 1 && !this.visible(chain[at], chain[next])) next--;
      if (!this.visible(chain[at], chain[next])) return [];
      route.push(chain[next]); at = next;
    }
    return route;
  }
}

function newActor(profile) {
  return {
    id: profile.id, species: profile.species, active: false, state: 'sleep', x: profile.origin.x, z: profile.origin.z,
    y: -.145, lift: 0, heading: 0, speed: 0, phase: 0, time: 0, stateTime: 0, wakeAt: 1e12,
    route: [], routeIndex: 0, origin: { ...profile.origin }, target: null, visit: 0, visitStart: 0,
    incidents: 0, footprintEvents: 0, scratchEvents: 0, nextContact: 0, lastContact: null,
    lastFootprints: [], footprintContacts: 0,
    nextScratch: .18, scratched: false, roams: 0, wait: 2, flying: false, scared: 0, avoidUntil: 0,
    alertPerson: null, lastState: null,
    actionBlend: 0, inspectBlend: 0, alertBlend: 0, flightBlend: 0, actionTime: 0,
  };
}

export function createWildlife(scene, { habitat, sand, onDisturb = () => {}, canBeginVisit = () => true, quality = 'high', seed = 381721, saved = null } = {}) {
  if (!habitat || !sand) throw new Error('Wildlife requires the existing habitat and sand.');
  let elapsed = 0, randomState = seed >>> 0 || 381721, disposed = false;
  let nextVisitAt = 0, visitCursor = 0, invitedId = null;
  const models = createWildlifeModels(scene, { quality });
  const navigators = PROFILES.map(p => new AnimalNavigation(habitat.obstacles || [], p.radius));
  const rigs = PROFILES.map(p => models.create(p.species, p.id));
  let actors = PROFILES.map(newActor);
  let stats = { visits: 0, footprintBatches: 0, footprintContacts: 0, scratchBatches: 0, disturbanceBatches: 0, points: 0, retreats: 0, failedRoutes: 0, firstSandContactAt: null };
  function random() { let a = randomState; a ^= a << 13; a ^= a >>> 17; a ^= a << 5; randomState = a >>> 0; return randomState / 4294967296; }
  nextVisitAt = QUIET_SECONDS.firstMin + random() * QUIET_SECONDS.firstRange;
  function groundAt(x, z) {
    if (sand.contains(x, z)) return sand.heightAt(x, z);
    // The raised timber is physical. Mammals hop over it instead of sliding
    // through; individual paws can plant on its upper face during the hop.
    if ((Math.abs(Math.abs(x) - 5.69) < .091 && Math.abs(z) < 4.28) ||
      (Math.abs(Math.abs(z) - 4.19) < .091 && Math.abs(x) < 5.78 && !(z < 0 && x > -3.7 && x < -2.7))) return .09;
    return typeof habitat.heightAt === 'function' ? habitat.heightAt(x, z) : -.145;
  }
  function contactAllowed(p, width) {
    return finitePoint(p) && sand.contains(p.x, p.z) && Math.abs(p.x) < 5.38 && Math.abs(p.z) < 3.89 &&
      !(habitat.obstacles || []).some(o => Math.hypot(p.x - o.x, p.z - o.z) < o.r + width * .8 + .025);
  }
  function event(a, points, { width, depth, action, contacts = 0 }) {
    // Marks belong to actual contacts, not to a visit quota. Spending two
    // prints at the border used to make the entire later walk invisible.
    // Finite routes/visits and per-foot placement deduplication bound the work.
    if (!a.active || a.flying || !points.length) return false;
    const safe = [];
    let penBreak = true;
    for (const p of points.slice(0, 30)) {
      if (!contactAllowed(p, width)) { penBreak = true; continue; }
      safe.push({ x: p.x, z: p.z, ...(penBreak || p.break ? { break: true } : {}) }); penBreak = false;
    }
    if (safe.length < 2) return false;
    if (onDisturb(safe, { species: a.species, width, depth, kind: 'visitor', action, contacts, animalId: a.id, visit: a.visit, incidentKey: `animal:${a.id}:${a.visit}`, at: elapsed }) === false) return false;
    a.incidents++; stats.disturbanceBatches++; stats.points += safe.length;
    if (action === 'footprints') { stats.footprintBatches++; stats.footprintContacts += contacts; } else stats.scratchBatches++;
    if (stats.firstSandContactAt === null) stats.firstSandContactAt = elapsed;
    return true;
  }
  function footprints(a) {
    const feet = rigs[a.id].live.feet || [], points = [], printed = [];
    const sn = Math.sin(a.heading), cs = Math.cos(a.heading), scale = WILDLIFE_SCALE[a.species];
    const bird = a.species === 'sparrow';
    const width = (a.species === 'rabbit' ? .062 : a.species === 'cat' ? .052 : .026) * scale;
    for (let i = 0; i < feet.length; i++) {
      const p = feet[i], previous = a.lastFootprints[i];
      // The published anchors locate the rendered pads/toes near the ground.
      // A planted paw stays put for many frames; only a new placement writes.
      if (p.y - groundAt(p.x, p.z) > (bird ? .010 : .024) * scale || !contactAllowed(p, width) ||
        previous && distance(p, previous) < (a.species === 'rabbit' ? .10 : .065) * scale) continue;
      if (bird) {
        points.push({ x: p.x + (sn * .015 - cs * .012) * scale, z: p.z + (cs * .015 + sn * .012) * scale, break: true },
          { x: p.x, z: p.z }, { x: p.x + (sn * .015 + cs * .012) * scale, z: p.z + (cs * .015 - sn * .012) * scale });
      } else {
        const hindRabbit = a.species === 'rabbit' && i >= 2;
        const back = hindRabbit ? .015 : .008, front = hindRabbit ? .055 : .032;
        points.push({ x: p.x - sn * back * scale, z: p.z - cs * back * scale, break: true },
          { x: p.x + sn * front * scale, z: p.z + cs * front * scale });
      }
      printed.push({ index: i, x: p.x, z: p.z });
    }
    if (event(a, points, { width, depth: bird ? .0052 : a.species === 'rabbit' ? .0088 : .0082, action: 'footprints', contacts: printed.length })) {
      a.footprintEvents++; a.lastContact = { x: a.x, z: a.z };
      a.footprintContacts += printed.length;
      for (const p of printed) a.lastFootprints[p.index] = { x: p.x, z: p.z };
    }
    a.nextContact = elapsed + .025;
  }
  function scratch(a) {
    const live = rigs[a.id].live, sn = Math.sin(a.heading), cs = Math.cos(a.heading), points = [], scale = WILDLIFE_SCALE[a.species];
    if (a.species === 'sparrow') {
      // A brief visible peck has a small V-shaped indentation under the beak.
      const p = live.paw;
      if (!p || p.y - groundAt(p.x, p.z) > .010 * scale) return;
      points.push({ x: p.x - (cs * .014 + sn * .014) * scale, z: p.z + (sn * .014 - cs * .014) * scale, break: true },
        { x: p.x, z: p.z }, { x: p.x + (cs * .012 - sn * .01) * scale, z: p.z - (sn * .012 + cs * .01) * scale });
      if (event(a, points, { width: .020 * scale, depth: .0044, action: 'peck' })) { a.scratchEvents++; return true; }
    } else {
      const front = (a.species === 'rabbit' ? live.feet.slice(0, 2) : live.feet.slice(0, 1)).filter(p => p.y - groundAt(p.x, p.z) < .024 * scale);
      if (!front.length) return false;
      for (const p of front) for (const claw of a.species === 'cat' ? [-1, 0, 1] : [0]) {
        const x = p.x + cs * claw * .018 * scale, z = p.z - sn * claw * .018 * scale;
        points.push({ x: x - sn * .083 * scale, z: z - cs * .083 * scale, break: true }, { x: x + sn * .016 * scale, z: z + cs * .016 * scale });
      }
      if (event(a, points, { width: (a.species === 'rabbit' ? .043 : .019) * scale, depth: .0068, action: 'scratch' })) { a.scratchEvents++; return true; }
    }
  }
  function setState(a, state, wait = 0) { if (state === 'scratch' && a.state !== 'scratch') a.actionTime = 0; a.state = state; a.stateTime = 0; a.wait = wait; }
  function assignRoute(a, destination, state) {
    const route = navigators[a.id].route(a, destination);
    if (!route.length) { stats.failedRoutes++; return false; }
    a.route = route; a.routeIndex = 0; a.target = { x: destination.x, z: destination.z }; setState(a, state); return true;
  }
  function chooseSpot(a, extra = false) {
    const offset = (a.id === 3 ? 2 : a.id) + (a.visit - 1) * 2 + (extra ? 1 : 0);
    const preferred = SPOTS[((offset % SPOTS.length) + SPOTS.length) % SPOTS.length];
    const span = a.species === 'sparrow' ? .18 : .22;
    for (let i = 0; i < 15; i++) {
      const p = { x: preferred.x + (random() - .5) * span, z: preferred.z + (random() - .5) * span };
      if (navigators[a.id].free(p.x, p.z)) return p;
    }
    return { ...preferred };
  }
  function beginVisit(a) {
    const p = PROFILES[a.id];
    a.active = true; a.x = a.origin.x; a.z = a.origin.z; a.y = groundAt(a.x, a.z);
    a.visit++; a.visitStart = elapsed; a.incidents = a.footprintEvents = a.scratchEvents = 0;
    a.scratched = false; a.roams = 0; a.scared = 0; a.phase = 0; a.speed = 0; a.lastContact = null;
    a.lastFootprints = []; a.footprintContacts = 0;
    a.nextContact = elapsed; a.nextScratch = .18; a.flying = a.species === 'sparrow'; a.lift = a.flying ? .90 : 0;
    a.actionBlend = a.inspectBlend = a.alertBlend = a.actionTime = 0; a.flightBlend = a.flying ? 1 : 0;
    const spot = chooseSpot(a); a.heading = Math.atan2(spot.x - a.x, spot.z - a.z);
    if (!assignRoute(a, spot, 'approach')) { a.active = false; a.state = 'sleep'; a.wakeAt = elapsed + 30; nextVisitAt = a.wakeAt; return false; }
    stats.visits++;
    // Night still has occasional visitors; heavy rain simply makes them brisk.
    a.wait = p.species === 'cat' ? 2.2 : p.species === 'rabbit' ? 1.65 : .72;
    return true;
  }
  function depart(a) {
    a.flying = a.species === 'sparrow';
    if (!assignRoute(a, a.origin, 'depart')) {
      // No teleport fallback: remain visible and retry from this exact place.
      setState(a, 'rest', 2.5); a.roams = 2;
    }
  }
  function completeVisit(a) {
    // Visibility changes only after the animal is back outside the courtyard.
    if (Math.hypot(a.x, a.z) < 10.2) { depart(a); return; }
    a.active = false; a.state = 'sleep'; a.speed = 0; a.route = []; a.routeIndex = 0;
    // One shared woodland clock: each visit is followed by a real quiet gap.
    // Legacy saves may contain several visitors; let each walk home normally,
    // then start this gap only after the last one has gone.
    a.wakeAt = elapsed + QUIET_SECONDS.betweenMin + random() * QUIET_SECONDS.betweenRange;
    if (!actors.some(other => other.active)) nextVisitAt = invitedId === null ? a.wakeAt : elapsed + 1;
  }
  function nearestPerson(a, people) {
    let best = null, bestDistance = Infinity;
    for (const p of people) {
      if (!p || p.visible === false || !finitePoint(p)) continue;
      const d = distance(a, p);
      if (d < bestDistance) { best = p; bestDistance = d; }
    }
    return { person: best, distance: bestDistance };
  }
  function evade(a, person) {
    a.scared++; stats.retreats++; a.avoidUntil = elapsed + 6.5;
    if (a.species === 'sparrow' || a.scared >= 2 || a.scratched) { depart(a); return; }
    const angle = Math.atan2(a.x - person.x, a.z - person.z);
    for (const turn of [0, .45, -.45, .90, -.90, 1.6, -1.6]) {
      const p = { x: a.x + Math.sin(angle + turn) * 1.8, z: a.z + Math.cos(angle + turn) * 1.8 };
      if (navigators[a.id].free(p.x, p.z) && assignRoute(a, p, 'retreat')) return;
    }
    depart(a);
  }
  function borderHop(a) {
    if (a.species === 'sparrow') return 0;
    const x = Math.abs(a.x), z = Math.abs(a.z);
    let d = Infinity;
    if (z < 4.05) d = Math.min(d, Math.abs(x - 5.69));
    if (x < 5.48 && !(a.z < 0 && a.x > -3.7 && a.x < -2.7)) d = Math.min(d, Math.abs(z - 4.19));
    return d < .52 ? Math.pow(Math.cos(d / .52 * Math.PI / 2), 1.2) * .265 : 0;
  }
  function move(a, dt, env) {
    const bird = a.species === 'sparrow', nav = navigators[a.id], p = PROFILES[a.id];
    if (a.routeIndex >= a.route.length) return true;
    const goal = a.route[a.routeIndex], dx = goal.x - a.x, dz = goal.z - a.z, remaining = Math.hypot(dx, dz);
    const targetHeading = Math.atan2(dx, dz), headingError = wrapAngle(targetHeading - a.heading);
    a.heading += clamp(headingError, -(bird ? 6.4 : 3.2) * dt, (bird ? 6.4 : 3.2) * dt);
    const hurry = a.state === 'retreat' || a.state === 'depart' && (env.rain || 0) > .4;
    const flightSpeed = a.flying ? 2.05 : p.speed;
    const finalApproach = a.routeIndex === a.route.length - 1 ? clamp(remaining / (a.flying ? .32 : .21), .12, 1) : 1;
    const desiredSpeed = flightSpeed * (hurry ? 1.32 : 1) * clamp(1 - Math.abs(headingError) / 1.65, .035, 1) * finalApproach;
    a.speed = damp(a.speed, desiredSpeed, 6, dt);
    let pace = 1;
    if (a.species !== 'cat' && !a.flying) {
      // Grounded anticipation, a short propulsive hop and a landing pause.
      // Translation and lift use the SAME persisted phase, so rabbits and
      // sparrows cannot slide forward at constant speed while bouncing.
      a.phase = (a.phase + dt * (bird ? 3.05 : 1.75) * clamp(a.speed / p.speed, .28, 1.32)) % 1;
      const airborne = clamp((a.phase - .24) / .65);
      pace = (.10 + 1.32 * Math.sin(airborne * Math.PI)) / .6463;
    }
    const amount = Math.min(remaining, a.speed * pace * dt);
    const next = remaining > 1e-8 ? { x: a.x + dx / remaining * amount, z: a.z + dz / remaining * amount } : goal;
    if (!nav.visible(a, next)) {
      a.speed = 0;
      if (!assignRoute(a, a.target || a.origin, a.state)) depart(a);
      return false;
    }
    a.x = next.x; a.z = next.z;
    if (a.species === 'cat') a.phase = (a.phase + amount / (.394 * WILDLIFE_SCALE.cat * clamp(a.speed / (.55 * WILDLIFE_SCALE.cat), .28, 1))) % 1;
    if (remaining <= amount + .001) { a.x = goal.x; a.z = goal.z; a.routeIndex++; }
    return a.routeIndex >= a.route.length;
  }
  function updateActor(a, dt, env, people) {
    a.time = elapsed;
    if (!a.active) return;
    a.stateTime += dt;
    const oldState = a.state;
    const nearby = nearestPerson(a, people);
    if (nearby.person && nearby.distance < (a.species === 'sparrow' ? .77 : .92) && elapsed > a.avoidUntil && !['depart', 'retreat', 'alert'].includes(a.state)) {
      a.alertPerson = { x: nearby.person.x, z: nearby.person.z }; setState(a, 'alert', a.species === 'sparrow' ? .18 : .48); a.speed *= .4;
    }
    if (['approach', 'roam', 'retreat', 'depart'].includes(a.state)) {
      const finished = move(a, dt, env);
      if (finished) {
        if (a.state === 'depart') completeVisit(a);
        else {
          a.flying = false;
          setState(a, 'inspect', a.species === 'cat' ? 2.0 + random() : a.species === 'rabbit' ? 1.45 + random() * .6 : .55 + random() * .4);
        }
      }
    } else {
      a.speed = damp(a.speed, 0, 13, dt);
      if (a.state === 'alert' && a.stateTime >= a.wait) evade(a, a.alertPerson || a.origin);
      else if (a.state === 'inspect' && a.stateTime >= a.wait) {
        if (a.scratched) setState(a, 'rest', a.species === 'cat' ? 2.8 : 1.0);
        else { setState(a, 'scratch'); a.nextScratch = a.species === 'sparrow' ? .19 : .18; }
      } else if (a.state === 'scratch' && a.stateTime >= (a.species === 'sparrow' ? 1.55 : 1.95)) {
        a.scratched = true; setState(a, 'rest', a.species === 'cat' ? 2.6 : a.species === 'rabbit' ? 1.7 : .45);
      } else if (a.state === 'rest' && a.stateTime >= a.wait * ((env.rain || 0) > .5 ? .45 : 1)) {
        if (a.roams < 1 && elapsed - a.visitStart < 62) {
          a.roams++;
          if (!assignRoute(a, chooseSpot(a, true), 'roam')) depart(a);
        } else depart(a);
      }
    }
    // Visits finish even if people keep the animal from its preferred spot.
    if (a.active && elapsed - a.visitStart > 86 && a.state !== 'depart') depart(a);
    if (!a.active) { rigs[a.id].group.visible = false; return; }
    let lift = 0;
    if (a.flying) {
      const toGoal = a.target ? distance(a, a.target) : 5;
      lift = a.state === 'depart' ? Math.min(1, a.stateTime / 1.1) * 1.22 : clamp(toGoal / 1.9) * 1.16;
    } else if (['approach', 'roam', 'retreat', 'depart'].includes(a.state) && a.speed > .08) {
      if (a.species !== 'cat') { const scale = WILDLIFE_SCALE[a.species]; lift = Math.sin(Math.PI * clamp((a.phase - .24) / .65)) * (a.species === 'rabbit' ? .125 : .065) * scale * clamp(a.speed / ((a.species === 'rabbit' ? .5 : .32) * scale), .35, 1); }
      lift = Math.max(lift, borderHop(a));
    }
    a.lift = damp(a.lift, lift, a.flying ? 11 : 28, dt);
    a.y = damp(a.y, groundAt(a.x, a.z), 18, dt);
    if (a.state === 'scratch' || a.actionBlend > .005) a.actionTime += dt;
    a.actionBlend = damp(a.actionBlend, a.state === 'scratch' ? 1 : 0, 8, dt);
    a.inspectBlend = damp(a.inspectBlend, ['inspect', 'rest'].includes(a.state) ? 1 : 0, 5, dt);
    a.alertBlend = damp(a.alertBlend, ['alert', 'retreat'].includes(a.state) ? 1 : 0, 7, dt);
    a.flightBlend = damp(a.flightBlend, a.flying ? 1 : 0, 9, dt);
    rigs[a.id].update(a, groundAt);
    const travellingOnSand = ['approach', 'roam', 'retreat', 'depart', 'inspect'].includes(a.state) && !a.flying;
    if (travellingOnSand && elapsed >= a.nextContact) footprints(a);
    if (a.state === 'scratch' && a.stateTime >= a.nextScratch && a.scratchEvents < 2) {
      const touched = scratch(a);
      // Wait for actual contact; a lifted paw or an unfinished landing cannot
      // spend the next incident and write a mark underneath empty air.
      if (touched) a.nextScratch = a.stateTime + (a.species === 'sparrow' ? TAU / 8.2 : TAU / 8.6) * .88;
    }
    if (oldState !== a.state) a.lastState = oldState;
  }
  function restore(value) {
    if (!value || value.format !== 1 || !Array.isArray(value.actors) || value.actors.length !== PROFILES.length || !Number.isFinite(value.elapsed)) return false;
    const restored = [];
    for (let i = 0; i < PROFILES.length; i++) {
      const item = value.actors[i], base = newActor(PROFILES[i]);
      if (!item || item.id !== i || item.species !== base.species || !STATES.has(item.state) || !finitePoint(item) ||
        !['y', 'lift', 'heading', 'speed', 'phase', 'stateTime', 'wakeAt', 'visit', 'visitStart', 'incidents'].every(k => Number.isFinite(item[k])) ||
        !Array.isArray(item.route) || item.route.length > 256 || !item.route.every(finitePoint)) return false;
      const a = { ...base };
      // Copy only recognised state fields; object references never enter the
      // scene graph, and a malformed save cannot change prototype properties.
      for (const key of Object.keys(base)) if (Object.hasOwn(item, key)) a[key] = structuredClone(item[key]);
      if (finitePoint(item.alertPerson)) a.alertPerson = { ...item.alertPerson };
      a.active = Boolean(item.active); a.flying = Boolean(item.flying);
      for (const key of ['actionBlend', 'inspectBlend', 'alertBlend', 'flightBlend']) a[key] = Number.isFinite(a[key]) ? clamp(a[key]) : 0;
      if (!Object.hasOwn(item, 'actionTime')) a.actionTime = a.state === 'scratch' ? a.stateTime : 0;
      if (!Number.isFinite(a.actionTime) || a.actionTime < 0) a.actionTime = 0;
      a.routeIndex = clamp(Math.floor(a.routeIndex), 0, a.route.length);
      if (a.phase < 0 || a.phase >= 1) a.phase = ((a.phase % 1) + 1) % 1;
      if (!finitePoint(a.origin)) a.origin = { ...PROFILES[i].origin };
      if (a.target && !finitePoint(a.target)) a.target = null;
      a.lastFootprints = Array.isArray(a.lastFootprints) ? a.lastFootprints.slice(0, a.species === 'sparrow' ? 2 : 4).map(p => finitePoint(p) ? { x: p.x, z: p.z } : null) : [];
      if (!Number.isFinite(a.footprintContacts) || a.footprintContacts < 0) a.footprintContacts = 0;
      a.time = value.elapsed;
      restored.push(a);
    }
    elapsed = Math.max(0, value.elapsed); randomState = value.randomState >>> 0 || seed >>> 0 || 381721;
    actors = restored;
    if (value.scheduleVersion === 2 && Number.isFinite(value.nextVisitAt) && Number.isInteger(value.visitCursor) && value.visitCursor >= 0 &&
      (value.invitedId === null || Number.isInteger(value.invitedId) && value.invitedId >= 0 && value.invitedId < actors.length)) {
      nextVisitAt = value.nextVisitAt; visitCursor = value.visitCursor; invitedId = value.invitedId;
    } else {
      // Ignore the old four independent wake-up timers while retaining every
      // active position, route and footprint. Nothing disappears on upgrade.
      nextVisitAt = elapsed + QUIET_SECONDS.firstMin + random() * QUIET_SECONDS.firstRange;
      visitCursor = Number.isFinite(value.stats?.visits) ? Math.max(0, Math.floor(value.stats.visits)) : 0; invitedId = null;
    }
    if (value.stats && typeof value.stats === 'object') for (const key of Object.keys(stats)) {
      if (Number.isFinite(value.stats[key]) && value.stats[key] >= 0 || key === 'firstSandContactAt' && value.stats[key] === null) stats[key] = value.stats[key];
    }
    return true;
  }
  if (saved) restore(saved);
  for (const a of actors) rigs[a.id].update(a, groundAt);
  function update(dt, _elapsed, env = {}, people = []) {
    if (disposed) return;
    dt = Number.isFinite(dt) ? clamp(dt, 0, 1) : 0;
    if (!dt) { for (const a of actors) rigs[a.id].update(a, groundAt); return; }
    let remaining = dt;
    while (remaining > 1e-8) {
      const step = Math.min(remaining, .05); remaining -= step; elapsed += step;
      // A present animal always finishes its visible visit. Future natural
      // arrivals wait for the finite care backlog; explicit invitations remain
      // available for deliberate interaction and development checks.
      if (!actors.some(a => a.active) && elapsed >= nextVisitAt && (invitedId !== null || canBeginVisit())) {
        const a = actors[invitedId ?? VISIT_ORDER[visitCursor % VISIT_ORDER.length]];
        if (beginVisit(a)) { if (invitedId === null) visitCursor++; invitedId = null; }
      }
      for (const a of actors) {
        updateActor(a, step, env, people);
      }
    }
  }
  function invite(species = 'cat') {
    if (species === 'bird' || species === 'birds') species = 'sparrow';
    const candidates = actors.filter(a => a.species === species);
    if (!candidates.length) return { accepted: false, reason: 'unknown-species' };
    const present = candidates.find(a => a.active);
    if (present) return { accepted: true, id: present.id, alreadyPresent: true };
    const sleeping = candidates[0], queued = actors.some(a => a.active);
    invitedId = sleeping.id; if (!queued) nextVisitAt = elapsed + .1;
    return { accepted: true, id: sleeping.id, arrivesFromWoodland: true, queued };
  }
  function serialize() { return { format: 1, scheduleVersion: 2, elapsed, randomState, nextVisitAt, visitCursor, invitedId, actors: structuredClone(actors), stats: { ...stats } }; }
  function snapshot() {
    return {
      elapsed, nextVisitAt, stats: { ...stats },
      animals: actors.map(a => ({ id: a.id, species: a.species, visible: a.active, state: a.state, x: a.x, y: a.y + a.lift, z: a.z,
        heading: a.heading, visit: a.visit, incidents: a.incidents, flying: a.flying,
        physicalClearance: (habitat.obstacles || []).reduce((d, o) => Math.min(d, Math.hypot(a.x - o.x, a.z - o.z) - o.r), Infinity) })),
    };
  }
  function dispose() { if (disposed) return; disposed = true; models.dispose(); }
  return { update, serialize, snapshot, invite, dispose, rigs };
}
