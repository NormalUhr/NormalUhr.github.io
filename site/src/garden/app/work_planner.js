import { Navigation } from './navigation.js';
import { TAU, wrapAngle } from './math.js';

// Planning is read-only. These helpers return a real stance or itinerary; only
// the worker's ordinary walking and brush states may change the garden.
const SMOOTH = new Set(['clean', 'cycle', 'erase', 'chore']);
const DRAW = new Set(['restore', 'time']);
const point = p => ({ x: p.x, z: p.z });
const distance = (a, b) => Math.hypot(a.x - b.x, a.z - b.z);
const homeFor = world => world.habitat.home.threshold || { x: -3.2, z: -4.12 };
const keyFor = p => `${Math.round(p.x * 10000)}:${Math.round(p.z * 10000)}`;
const egressCaches = new WeakMap();
const growthCaches = new WeakMap();
const actualRouteCaches = new WeakMap();
const surfaceRevision = world => `${world.lastMutation}:${world.sand.field?.revision ?? world.sand.revision ?? 0}`;

function verifiedFootRoute(world, start, route) {
  if (!route.length) return [];
  let a = start;
  for (const b of route) {
    if (!world.feetVisible(a, b)) return [];
    a = b;
  }
  return route.map(point);
}

/** A precisely sampled fallback uses the real feet, never a smaller body. */
export function actualFootRoute(world, a, b) {
  const revision = surfaceRevision(world);
  let cache = actualRouteCaches.get(world);
  if (!cache || cache.revision !== revision) {
    cache = { revision, navigator: new Navigation(world.habitat.obstacles, {
      surface: (x, z) => world.surfaceFree(x, z), cell: .16, fineCell: .05, surfaceStep: .005,
    }), routes: new Map() };
    actualRouteCaches.set(world, cache);
  }
  const key = `${a.x}:${a.z}>${b.x}:${b.z}`;
  let route = cache.routes.get(key);
  if (!route) {
    route = verifiedFootRoute(world, a, cache.navigator.route(a, b));
    if (cache.routes.size >= 64) cache.routes.delete(cache.routes.keys().next().value);
    cache.routes.set(key, route);
  }
  return route.map(point);
}

function segmentsFor(task) {
  const segments = [];
  for (let i = 1; i < task.path.length; i++) {
    if (task.path[i].working === false) continue;
    const a = task.path[i - 1], b = task.path[i], dx = b.x - a.x, dz = b.z - a.z;
    segments.push({ a, b, dx, dz, lengthSq: dx * dx + dz * dz,
      minX: Math.min(a.x, b.x), maxX: Math.max(a.x, b.x), minZ: Math.min(a.z, b.z), maxZ: Math.max(a.z, b.z) });
  }
  return segments;
}

function futureHazard(segments, radius) {
  const radiusSq = radius * radius;
  return (x, z) => {
    for (const s of segments) {
      if (x < s.minX - radius || x > s.maxX + radius || z < s.minZ - radius || z > s.maxZ + radius) continue;
      const t = s.lengthSq ? Math.max(0, Math.min(1, ((x - s.a.x) * s.dx + (z - s.a.z) * s.dz) / s.lengthSq)) : 0;
      if ((x - s.a.x - s.dx * t) ** 2 + (z - s.a.z - s.dz * t) ** 2 < radiusSq) return true;
    }
    return false;
  };
}

function routeStaysOpen(world, body, target, task, knownRoute = null) {
  if (!DRAW.has(task.kind) || task.path.length < 2) return true;
  // The candidate is the same worker at a future stance. Preserve identity so
  // the current body is not mistaken for another person blocking the home or
  // paused-worksite endpoint of this prospective route.
  const traveller = { ...body, id: world.active?.id };
  const route = knownRoute || world.travelRoute(traveller, target);
  if (!route.length && distance(body, target) > .03) return false;
  const segments = segmentsFor(task), hazard = futureHazard(segments, world.rakeFootClearance());
  let a = body, intersects = hazard(body.x, body.z);
  for (const b of route) {
    const n = Math.max(1, Math.ceil(distance(a, b) / .025));
    for (let k = 1; !intersects && k <= n; k++)
      intersects = hazard(a.x + (b.x - a.x) * k / n, a.z + (b.z - a.z) * k / n);
    a = b;
  }
  if (!intersects) return true;

  // A shortest route crossing the future stroke does not prove that every
  // exit is closed. Search the actual remaining component before rejecting it.
  const revision = surfaceRevision(world);
  let cache = egressCaches.get(world);
  if (!cache || cache.revision !== revision) {
    cache = { revision, routes: new Map() }; egressCaches.set(world, cache);
  }
  const pathKey = task.path.map(p => `${keyFor(p)}:${p.working === false ? 0 : 1}`).join('/');
  let navigators = cache.routes.get(pathKey);
  if (!navigators) {
    navigators = { standard: new Navigation(world.habitat.obstacles, {
      surface: (x, z) => !hazard(x, z) && world.walkNav.surfaceAt(x, z), cell: .24, fineCell: .10,
    }) };
    if (cache.routes.size >= 8) cache.routes.delete(cache.routes.keys().next().value);
    cache.routes.set(pathKey, navigators);
  }
  if (futureRouteOpen(world, navigators.standard, body, target)) return true;
  // The extra planning margin can disappear at a swept junction even though
  // the worker's real feet can retrace their way out. Check that remaining
  // route with finer samples and the same future-rake exclusion, then verify
  // every edge against the unchanged physical foot and obstacle checks.
  navigators.actual ||= new Navigation(world.habitat.obstacles, {
    surface: (x, z) => !hazard(x, z) && world.surfaceFree(x, z), cell: .16, fineCell: .05, surfaceStep: .005,
  });
  return futureRouteOpen(world, navigators.actual, body, target, true);
}


const COMPONENT_STEPS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]];
function attachFutureNode(nav, grid, p) {
  const cx=Math.max(0,Math.min(grid.nx-1,Math.round((p.x-nav.minX)/grid.cell))),cz=Math.max(0,Math.min(grid.nz-1,Math.round((p.z-nav.minZ)/grid.cell)));
  let best=-1,bestDistance=Infinity;
  for(let ring=0;ring<=6;ring++){
    for(let z=cz-ring;z<=cz+ring;z++)for(let x=cx-ring;x<=cx+ring;x++){
      if(x<0||z<0||x>=grid.nx||z>=grid.nz||(ring&&Math.abs(x-cx)!==ring&&Math.abs(z-cz)!==ring))continue;
      const id=z*grid.nx+x,q=grid.points[id],d=distance(p,q);
      if(d<bestDistance&&nav.nodeFree(grid,id)&&nav.visible(p,q)){best=id;bestDistance=d;}
    }
    if(best>=0&&(ring+.5)*grid.cell>bestDistance+grid.cell)break;
  }
  return best;
}
function futureRouteOpen(world,nav,body,target,actual=false){
  if(!nav.free(body.x,body.z)||!nav.free(target.x,target.z))return false;
  if(nav.visible(body,target))return !actual||world.feetVisible(body,target);
  nav.workComponents ||= new Map();
  for(const cell of [nav.cell,nav.fineCell].filter(Boolean)){
    const grid=nav.gridFor(cell),from=attachFutureNode(nav,grid,body),to=attachFutureNode(nav,grid,target);
    if(from<0||to<0)continue;
    let info=nav.workComponents.get(cell);if(!info){info={labels:new Uint16Array(grid.count),next:1};nav.workComponents.set(cell,info);}
    const label=info.labels[from],other=info.labels[to];
    if(label&&other===label)return true;
    if(label||other)continue;
    const route=nav.routeOnGrid(body,target,grid);
    if(route.length&&(!actual||verifiedFootRoute(world,body,route).length))return true;
    const seen=new Set([from]),queue=[from];let at=0;
    while(at<queue.length&&queue.length<=2500){
      const id=queue[at++],cx=id%grid.nx,cz=Math.floor(id/grid.nx);
      for(let d=0;d<8;d++){
        const x=cx+COMPONENT_STEPS[d][0],z=cz+COMPONENT_STEPS[d][1];
        if(x<0||z<0||x>=grid.nx||z>=grid.nz)continue;
        const next=z*grid.nx+x;if(seen.has(next)||!nav.gridEdge(grid,id,next,d))continue;
        seen.add(next);queue.push(next);
      }
    }
    if(at===queue.length&&queue.length<=2500){const next=info.next++;for(const id of queue)info.labels[id]=next;}
  }
  return false;
}

/** Keep both home and any paused worksite reachable after this brush stroke. */
export function exitStaysOpen(world, body, task, additionalTargets = []) {
  return [homeFor(world), ...additionalTargets].every(target => routeStaysOpen(world, body, target, task));
}

/** Find a useful stance on the worker's reachable clear component first. */
export function findWorkAnchor(world, w, task, { clearOnly = false, directOnly = false } = {}) {
  if (!task.path?.length) return null;
  const restoring = task.kind === 'restore', drawing = DRAW.has(task.kind), child = w.id >= 4;
  const p = restoring ? task.path[Math.floor((task.path.length - 1) / 2)] : task.path[0];
  const heading = world.taskHeading(task), smooth = SMOOTH.has(task.kind);
  const relocatingClean = task.kind === 'clean' && w.task?.kind === 'clean' && task !== w.task &&
    ['working', 'relocating', 'waitingForAccess'].includes(w.state);
  const home = homeFor(world);
  const preferred = task.kind === 'cycle' ? Math.atan2(-p.x, -p.z) :
    restoring ? (w.accessSerial > 0 ? Math.atan2(home.x - p.x, home.z - p.z) : Math.atan2(w.x - p.x, w.z - p.z)) :
    drawing ? Math.atan2(p.x, p.z) : smooth ? heading + Math.PI : Math.atan2(w.x - p.x, w.z - p.z);
  const minimum = drawing ? world.rakeFootClearance() : child ? .28 : .31;
  const maximum = child ? (restoring ? .575 : .56) : (restoring ? .835 : .82);
  // Restore fragments must fit one stance. A longer new clock stroke will
  // move its stance later, but the approaching tip must already be foot-safe.
  const checkPoints = restoring ? task.path : drawing ? task.path.slice(0, Math.min(4, task.path.length)) : [p];
  const options = [], seen = new Set();
  const add = (body, preference = 0) => {
    const key = keyFor(body); if (seen.has(key)) return; seen.add(key);
    if (!world.nav.free(body.x, body.z, .21) || world.occupiedAnchor(w, body)) return;
    if (checkPoints.some(q => { const d = distance(body, q); return d < minimum || d > maximum; })) return;
    const angle = Math.atan2(body.x - p.x, body.z - p.z);
    // A stroke that has exhausted its arm travel needs room to continue.
    // Merely choosing the closest long-reach pose moves the feet by a few
    // centimetres, then raises the rake again almost immediately. Once work
    // is underway, favor the normal trailing pose that can follow its sweep.
    // Initial jobs retain extended reach from the clear outer border.
    const continuationCost = relocatingClean ? Math.abs(distance(body, p) - (child ? .36 : .46)) * 2.5 +
      Math.abs(wrapAngle(angle - preferred)) * .30 : 0;
    options.push({ ...point(body), clear: world.walkNav.free(body.x, body.z),
      score: distance(w, body) + Math.abs(wrapAngle(angle - preferred)) * .025 + continuationCost - (distance(w, body) < .001 ? .10 : 0) - preference });
  };
  // A clock stroke begins at the stance it keeps while walking backward ahead
  // of the rake, so that it can follow the whole stroke without a new stance.
  const walkingStance = task.walking ? world.clockStance(p, heading) : null;
  if (walkingStance) add(walkingStance, 10);
  add(w);
  // A nearby obstacle can leave a small useful stance between the polar
  // samples around the tool. Try modest local foot placements too, so a whole
  // reachable arc does not get split solely by the candidate grid alignment.
  if (restoring && distance(w, p) <= maximum + .30) for (const radius of [.06, .12, .20, .30]) for (let k = 0; k < 16; k++) {
    const angle = k * TAU / 16;
    add({ x: w.x + Math.sin(angle) * radius, z: w.z + Math.cos(angle) * radius });
  }
  for (const q of w.accessTrail || []) if (distance(q, p) <= maximum + .025) add(q);
  // The dependency proof may have found a precise clear stance between two
  // finished grooves. Reuse that real witness instead of losing it when the
  // regular polar candidate grid is centred on a different repair fragment.
  const workAccess = unfinishedAccessCaches.get(world);
  if (workAccess?.worker === w) for (const q of workAccess.witnesses)
    if (distance(q, p) <= maximum + .025) add(q);
  const radii = child ? (relocatingClean ? [.36, .40, .49, .55, .34] : [.40, .49, .55, .34]) :
    restoring ? [.64, .73, .55, .81, .43] : relocatingClean ? [.46, .40, .55, .67, .77, .81, .37] : [.55, .67, .77, .46, .81, .37];
  for (const radius of radii) for (let k = 0; k < 24; k++) {
    const angle = preferred + k * TAU / 24;
    add({ x: p.x + Math.sin(angle) * radius, z: p.z + Math.cos(angle) * radius });
  }
  const clear = options.filter(o => o.clear).sort((a, b) => a.score - b.score);
  // Reject an unreachable arm span before the more expensive connectivity
  // proof. This changes no viable plan: no worker could perform that stroke.
  if (!options.length || ((clearOnly || directOnly) && !clear.length)) return null;
  if (!unfinishedWorkAccess(world, w, task).open) return null;
  const egress = new Map();
  const acceptable = body => {
    if (!drawing) return true;
    const key = keyFor(body);
    // The walking stance does not stay where the stroke begins. Its groove
    // trails behind the feet, so the way home starts where the stroke ends.
    const from = walkingStance && key === keyFor(walkingStance) ?
      world.clockStance(task.path.at(-1), world.taskHeading(task, task.path.length - 1)) : body;
    if (!egress.has(key)) egress.set(key, exitStaysOpen(world, from, task, task.keepAccess || []));
    return egress.get(key);
  };
  // The walking stance is worth a walk round the clock. From any other stance
  // the stroke would stop for a new one as soon as it began.
  const start = walkingStance && clear.find(o => keyFor(o) === keyFor(walkingStance));
  if (start && acceptable(start)) {
    if (world.walkNav.visible(w, start) && !world.personBlocker(w, w, start, .34)) return { ...start, route: [point(start)] };
    const route = directOnly ? [] : world.travelRoute(w, start);
    if (route.length) return { ...start, route };
  }
  for (const body of clear) {
    if (!world.walkNav.visible(w, body) || world.personBlocker(w, w, body, .34) || !acceptable(body)) continue;
    return { ...body, route: [point(body)] };
  }
  if (directOnly) return null;
  // Do not confuse the five closest disconnected pockets with all possible
  // stances. A peripheral repair often needs a longer walk around the clock.
  for (const body of clear) {
    const route = world.travelRoute(w, body);
    if (!route.length && distance(w, body) > .015) continue;
    if (!acceptable(body)) continue;
    return { ...body, route };
  }
  if (clearOnly) return null;
  // A rejected future exit cannot become permissible just because this is the
  // fallback pass. Only an as-yet unreachable stance can request a new entry.
  // Keep a normal working reach when making a new floor. The extended arm
  // candidates are valuable on existing clear ground; cutting to the very
  // limit of reach leaves no tolerance for a blade's first few centimetres.
  const desiredReach = child ? .36 : restoring ? .64 : .46;
  const fallbackScore = o => o.score + Math.abs(distance(o, p) - desiredReach) * 2;
  const blocked = options.filter(o => egress.get(keyFor(o)) !== false);
  if (!blocked.length) return null;
  const normalReach = blocked.filter(o => Math.abs(distance(o, p) - desiredReach) < .115);
  const pool = normalReach.length ? normalReach : blocked;
  const outer = .62 + 2.33 * Math.min(1, Math.max(world.minute, world.recordedMinute) / 720) + .36;
  const lane = (w.accessTrail || []).filter(q => world.walkNav.free(q.x, q.z));
  const estimatedCut = o => {
    let gap = Math.min(distance(w, o), Math.max(0, outer - Math.hypot(o.x, o.z)));
    for (const q of lane) gap = Math.min(gap, distance(q, o));
    return gap * 8 + Math.abs(distance(o, p) - desiredReach) * 2 + distance(w, o) * .08;
  };
  pool.sort((a, b) => estimatedCut(a) - estimatedCut(b));
  const shortlist = [];
  for (const body of pool) {
    if (shortlist.every(other => distance(body, other) > .14)) shortlist.push(body);
    if (shortlist.length === 3) break;
  }
  const near = [...pool].sort((a, b) => fallbackScore(a) - fallbackScore(b))[0];
  if (near && shortlist.every(other => distance(near, other) > .14)) shortlist.push(near);
  let best = null;
  for (const body of shortlist) {
    const accessPlan = planAccessApproach(world, w, body);
    if (!accessPlan) continue;
    if (!best || accessPlan.score < best.accessPlan.score) best = { ...body, route: null, accessPlan };
  }
  // Select the stance and its entrance together. Choosing the stance nearest
  // the person first would favor the inner side of a scratch even when the
  // actual approach sensibly walks around to the opposite outer border.
  return best || { ...pool[0], route: null };
}

function trimClearPrefix(world, seed, route) {
  let entry = point(seed), previous = point(seed);
  for (let i = 0; i < route.length; i++) {
    const b = route[i], n = Math.max(1, Math.ceil(distance(previous, b) / .045));
    for (let k = 1; k <= n; k++) {
      const q = { x: previous.x + (b.x - previous.x) * k / n, z: previous.z + (b.z - previous.z) * k / n };
      if (!world.walkNav.visible(entry, q)) return { entry, route: route.slice(i).map(point) };
      entry = q;
    }
    previous = b;
  }
  return { entry, route: [] };
}

function growthHazard(world) {
  const from = Math.max(0, world.recordedMinute), to = Math.min(720, Math.max(from, world.minute) + 6);
  const radius = world.rakeFootClearance() + .055, key = `${from}:${to}:${radius}`;
  let cached = growthCaches.get(world);
  if (cached?.key === key) return cached.hazard;
  const path = [], count = Math.max(1, Math.ceil((to - from) / .125));
  for (let i = 0; i <= count; i++) {
    const minute = from + (to - from) * i / count, r = .62 + 2.33 * minute / 720, angle = minute * TAU / 60;
    path.push({ x: Math.sin(angle) * r, z: -Math.cos(angle) * r, working: i > 0 });
  }
  const hazard = futureHazard(segmentsFor({ path }), radius);
  growthCaches.set(world, { key, hazard });
  return hazard;
}

function cutCost(world, entry, route) {
  const touched = new Set(), upcoming = growthHazard(world); let a = entry, length = 0, growthConflict = 0;
  const scoringRoute = route.map(point);
  if (scoringRoute.length) {
    const last = scoringRoute.at(-1), prior = scoringRoute.at(-2) || entry, d = distance(prior, last);
    if (d > 1e-6) {
      const beyond = { x: last.x + (last.x - prior.x) / d * .24, z: last.z + (last.z - prior.z) / d * .24 };
      if (world.nav.visible(last, beyond)) scoringRoute.push(beyond);
    }
  }
  for (const b of scoringRoute) {
    const part = distance(a, b); length += part;
    const count = Math.max(1, Math.ceil(part / .05));
    for (let k = 0; k < count; k++) {
      const t = (k + .5) / count;
      if (upcoming(a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t)) growthConflict += part / count;
    }
    for (const index of world.affected(a, b, .68)) if (world.coverage[index]) touched.add(index);
    a = b;
  }
  let intactLength = 0;
  for (const index of touched) intactLength += distance(world.points[index], world.points[index + 1]);
  // Do not put the only entrance across the clock's growing tip. Otherwise
  // the first due minute would close the corridor and must wait for every
  // interior repair. Pay for a slightly different mouth now, before cutting.
  const growthCost = growthConflict > 1e-6 ? 90 + growthConflict * 400 : 0;
  return { length, intactLength, growthConflict, weighted: intactLength * 18 + length * 2.5 + growthCost };
}

/** Walk around intact work before making the shortest useful local entry. */
export function planAccessApproach(world, w, body) {
  const seeds = [], seedKeys = new Set();
  const add = (p, current = false) => {
    if (!current && (Math.abs(p.x) > 5.34 || p.z < -4.02 || p.z > 3.90)) return;
    const key = keyFor(p); if (seedKeys.has(key)) return; seedKeys.add(key);
    if (!world.nav.free(p.x, p.z) || !world.walkNav.free(p.x, p.z)) return;
    seeds.push(point(p));
  };
  add(w, true);
  for (const q of w.accessTrail || []) add(q);
  const outward = Math.atan2(body.x, body.z);
  for (const radius of [.36, .60, .90, 1.25, 1.70, 2.25, 2.90, 3.60]) for (let k = 0; k < 24; k++) {
    const angle = outward + k * TAU / 24;
    add({ x: body.x + Math.sin(angle) * radius, z: body.z + Math.cos(angle) * radius });
  }
  const clockRadius = .62 + 2.33 * Math.min(1, Math.max(world.minute, world.recordedMinute) / 720);
  for (const radius of [clockRadius + .36, clockRadius + .55, 3.34]) for (let k = 0; k < 32; k++) {
    const angle = outward + k * TAU / 32;
    add({ x: Math.sin(angle) * radius, z: Math.cos(angle) * radius });
  }
  // The present position is always considered, but a diameter-sized cut is
  // deliberately much more expensive than a clear walk to the other side.
  const candidates = [], entryKeys = new Set();
  let obstacleDetours = 0;
  seeds.sort((a, b) => distance(a, body) - distance(b, body));
  for (const seed of seeds) {
    let physical;
    if (world.nav.visible(seed, body)) physical = [point(body)];
    else if (distance(seed, w) < .001 || obstacleDetours++ < 16) physical = world.nav.route(seed, body);
    else continue;
    if (!physical.length && distance(seed, body) > .025) continue;
    const cut = trimClearPrefix(world, seed, physical), key = keyFor(cut.entry);
    if (entryKeys.has(key)) continue; entryKeys.add(key);
    const cost = cutCost(world, cut.entry, cut.route);
    candidates.push({ ...cut, cost, lowerBound: cost.weighted + distance(w, cut.entry) * .35 });
  }
  candidates.sort((a, b) => a.lowerBound - b.lowerBound);
  let best = null, queries = 0;
  for (const candidate of candidates) {
    if (best && candidate.lowerBound >= best.score) break;
    // Bounded expensive searches; direct candidates and the existing passage
    // have already been scored. An unselected candidate never changes sand.
    if (++queries > 48 && best) break;
    const approach = world.travelRoute(w, candidate.entry);
    if (!approach.length && distance(w, candidate.entry) > .025) continue;
    const approachLength = world.routeLength(w, approach), score = candidate.cost.weighted + approachLength * .35;
    if (!best || score < best.score) best = { approach, entry: point(candidate.entry), route: candidate.route,
      score, intactLength: candidate.cost.intactLength, cutLength: candidate.cost.length, approachLength,
      growthConflict: candidate.cost.growthConflict };
  }
  return best;
}

const unfinishedAccessCaches = new WeakMap();
/** Preserve a feasible stance for work reachable before this stroke. */
export function unfinishedWorkAccess(world, w, task) {
  if(task.kind!=='restore'||!w?.repairs||!task.path?.length||!task.path.every(p=>Number.isFinite(p.m)))return {open:true,blocked:[]};
  const revision=surfaceRevision(world), key=task.path.map(p=>`${keyFor(p)}:${p.m}`).join('/');
  let cache=unfinishedAccessCaches.get(world);
  if(!cache||cache.worker!==w){cache={worker:w,witnesses:[],keys:new Set()};unfinishedAccessCaches.set(world,cache);}
  if(cache.revision!==revision){cache.revision=revision;cache.previousCurrent=cache.current;cache.current=new Map();cache.results=new Map();cache.homeRoutes=[];cache.homeKeys=new Set();}
  if(cache.results.has(key))return cache.results.get(key);
  const home=homeFor(world),minimum=world.rakeFootClearance(), maximum=w.id>=4?.575:.835;
  const from=Math.min(task.path[0].m,task.path.at(-1).m),to=Math.max(task.path[0].m,task.path.at(-1).m);
  const future=new Map(), owned=Object.keys(w.repairs).map(Number).filter(i=>!world.coverage[i]&&world.points[i].m<Math.min(world.minute,world.recordedMinute)&&!(world.points[i].m>=from-1e-7&&world.points[i].m<to-1e-7));
  const add=body=>{const k=keyFor(body);if(cache.keys.has(k))return;cache.keys.add(k);cache.witnesses.push(point(body));if(cache.witnesses.length>256){const old=cache.witnesses.shift();cache.keys.delete(keyFor(old));}};
  add(w);if(w.workAnchor)add(w.workAnchor);add({x:0,z:0});for(const body of w.accessTrail||[])add(body);
  const rememberRoute=(body,route)=>{
    const path=[body,...route];
    for(let i=0;i<path.length;i++){
      const p=path[i],k=keyFor(p);if(cache.homeKeys.has(k))continue;
      cache.homeKeys.add(k);cache.homeRoutes.push({body:point(p),route:path.slice(i+1).map(point)});
      if(cache.homeRoutes.length>192){const old=cache.homeRoutes.shift();cache.homeKeys.delete(keyFor(old.body));}
    }
  };
  const current=body=>{
    const k=keyFor(body);if(cache.current.has(k))return cache.current.get(k);
    let clear=world.nav.free(body.x,body.z,.21)&&world.walkNav.free(body.x,body.z),route=[];
    if(clear){
      const prior=cache.previousCurrent?.get(k);
      if(prior){let a=body,safe=true;for(const b of prior){if(!world.feetVisible(a,b)||world.personBlocker(w,a,b,.34)){safe=false;break;}a=b;}if(safe)route=prior;}
      const joins=cache.homeRoutes.filter(join=>distance(body,join.body)<1.25).sort((a,b)=>distance(body,a.body)-distance(body,b.body));
      if(!route.length)for(const join of joins.slice(0,12))if(world.feetVisible(body,join.body)&&!world.personBlocker(w,body,join.body,.34)){route=[point(join.body),...join.route];break;}
      if(!route.length){const traveller={...body,id:w.id};route=world.travelRoute(traveller,home);}
      clear=distance(body,home)<.03||route.length>0;if(clear)rememberRoute(body,route);
    }
    cache.current.set(k,clear?route:null);return clear?route:null;
  };
  const after=body=>{const k=keyFor(body);if(!future.has(k)){const route=current(body);future.set(k,!!route&&routeStaysOpen(world,body,home,task,route));}return future.get(k);};
  const reaches=(body,a,b)=>[a,b].every(p=>{const d=distance(body,p);return d>=minimum&&d<=maximum;});
  for(const i of owned){
    const a=world.points[i],b=world.points[i+1];if(!b)continue;
    let hadAccess=false,kept=false;
    for(const body of cache.witnesses){
      if(!reaches(body,a,b)||!current(body))continue;
      hadAccess=true;if(after(body)){kept=true;break;}
    }
    if(kept)continue;
    const p={x:(a.x+b.x)/2,z:(a.z+b.z)/2},preferred=Math.atan2(home.x-p.x,home.z-p.z),options=[];
    for(const radius of w.id>=4?[.40,.49,.55,.34]:[.64,.73,.55,.81,.43])for(let j=0;j<24;j++){
      const angle=preferred+j*TAU/24,body={x:p.x+Math.sin(angle)*radius,z:p.z+Math.cos(angle)*radius};
      if(reaches(body,a,b))options.push(body);
    }
    options.sort((a,b)=>distance(a,home)-distance(b,home));
    for(const body of options){
      if(!current(body))continue;hadAccess=true;add(body);
      if(after(body)){kept=true;break;}
    }
    if(hadAccess&&!kept){const result={open:false,blocked:[i]};cache.results.set(key,result);return result;}
  }
  const result={open:true,blocked:[]};cache.results.set(key,result);return result;
}
