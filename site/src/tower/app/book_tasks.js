import { TAU, clamp, polar, shortest } from './mechanism.js';
import { angleOf, appendLine, appendRing, arc, copyPoint, distance, floorY, outerRoute,
  innerRadius, segmentPoint, walkRadius } from './navigation.js';
import { words } from './words.js';

export const BOOK_TASK_TIMINGS = Object.freeze({ notice: 1.35, fall: 1.25, pickup: 1.8, shelve: 2.4, complete: 4 });
export const BOOK_SPILL_LIMITS = Object.freeze({ maxJobs: 3, perFloor: 1, cooldown: 2, helperRest: 45 });
const clone = value => JSON.parse(JSON.stringify(value));
const finitePoint = p => p && [p.x, p.y, p.z].every(Number.isFinite);
const active = job => job.phase !== 'complete';
const PHASES = ['notice', 'approach', 'pickup', 'carry', 'shelve', 'complete'];
const BOOK_STATES = ['falling', 'ground', 'picking', 'carried', 'shelving', 'shelved'];
const initialStats = () => ({ requested: 0, accepted: 0, rejected: {}, completed: 0, collected: 0, returned: 0,
  maxActive: 0, longestCompletion: 0, longestApproach: 0, longestReturn: 0, recent: [] });

/** Stable world-space landing locations; no renderer or global random state. */
export function bookLandingLayout(position, seed = 1) {
  const angle = angleOf(position), radial = { x: Math.sin(angle), z: -Math.cos(angle) };
  const tangent = { x: Math.cos(angle), z: Math.sin(angle) };
  let state = seed >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  return [[-.02, -.26], [.10, 0], [-.02, .26]].map(([r, t], index) => ({ index,
    position: { x: position.x + radial.x * r + tangent.x * t, y: position.y,
      z: position.z + radial.z * r + tangent.z * t },
    yaw: Math.PI / 2 - angle + (random() - .5) * .16 }));
}

function tail(path, fallback) { return path.length ? segmentPoint(path[path.length - 1], path[path.length - 1].length) : fallback; }
const pathLength = path => path.reduce((sum, s) => sum + s.length, 0);

/** Depart furniture through its existing approach, and approach the pile from
 * its outward side. The final arc never traverses any still-present book. */
export function bookPickupRoute(actor, position) {
  const level = actor.level, support = { type: 'floor', level }, path = [];
  let p = copyPoint(actor.position);
  const origin = actor._atDestination;
  if (origin) for (const next of [...origin.approach].reverse()) p = appendLine(path, p, next, support, .85);
  const targetAngle = angleOf(position), sign = Math.sign(shortest(angleOf(p), targetAngle)) || 1;
  const approachAngle = targetAngle - sign * 1.05 / walkRadius(level);
  p = appendRing(path, p, level, approachAngle);
  const radius = Math.hypot(position.x, position.z) + (actor.age === 'child' ? .28 : .37);
  p = appendLine(path, p, polar(radius, approachAngle, floorY(level)), support, .85);
  path.push(arc(radius, approachAngle, shortest(approachAngle, targetAngle), level, support, .85));
  return { path, stance: tail(path, p), length: pathLength(path) };
}

function returnRoute(from, level, site) {
  const path = outerRoute(from, level, level, site.angle), support = { type: 'floor', level };
  let p = tail(path, from);
  for (const next of site.approach) p = appendLine(path, p, next, support, .85);
  appendLine(path, p, site.position, support, .78);
  return path;
}

/** A small, bounded interruption of idle residents, never a second scheduler
 * for bridge passengers. All movement still uses the production path walker. */
export function createBookTasks({ actors, returns = [], getTime, setPath, setState, still, turn, isLifeFloorBusy = null }) {
  const returnSites = returns.filter(s => typeof s.id === 'string' && Number.isInteger(s.level) && s.level >= 0 && s.level <= 6 &&
    Number.isFinite(s.angle) && finitePoint(s.position) && finitePoint(s.shelfPosition) && Array.isArray(s.approach) && s.approach.every(finitePoint))
    .map(s => ({ ...s, position: { ...s.position, y: floorY(s.level) },
      approach: s.approach.map(p => ({ ...p, y: floorY(s.level) })), shelfPosition: { ...s.shelfPosition } }));
  const returnById = new Map(returnSites.map(s => [s.id, s]));
  const actorById = new Map(actors.map(a => [a.id, a]));
  let jobs = [], sequence = 0, lastAcceptedAt = -1e6, stats = initialStats();
  function reject(reason) { stats.rejected[reason] = (stats.rejected[reason] || 0) + 1; return { ok: false, reason }; }
  function findJob(actor) { return actor._bookJobId ? jobs.find(j => j.id === actor._bookJobId) : null; }
  function presentation(job, actor) {
    const bookIndex = Math.min(2, job.collected), target = job.phase === 'pickup' ? job.books[bookIndex].position
      : job.phase === 'carry' || job.phase === 'shelve' ? job.shelfPosition : job.position;
    actor.bookTask ||= {};
    Object.assign(actor.bookTask, { jobId: job.id, phase: job.phase, progress: job.progress, bookIndex,
      carriedCount: job.collected, target, stance: job.stance, shelfPosition: job.shelfPosition, shelfYaw: job.shelfYaw, shelfFacing: job.shelfFacing,
      aligning: Boolean(job.aligning) });
  }
  function phase(job, actor, name) {
    job.phase = name; job.phaseStartedAt = getTime(); job.progress = 0; presentation(job, actor);
  }
  function request({ level, position, seed = null } = {}) {
    stats.requested++;
    const radius = finitePoint(position) ? Math.hypot(position.x, position.z) : NaN;
    if (!Number.isInteger(level) || level < 0 || level > 6 || !finitePoint(position) ||
      Math.abs(position.y - floorY(level)) > .06 || radius < innerRadius(level) + .80 - 1e-6 || radius > 29.20 + 1e-6) return reject('invalid-place');
    const now = getTime();
    if (isLifeFloorBusy?.(level) || jobs.some(j => j.level === level && active(j))) return reject('floor-busy');
    if (jobs.length >= BOOK_SPILL_LIMITS.maxJobs) return reject('busy');
    if (now - lastAcceptedAt < BOOK_SPILL_LIMITS.cooldown) return reject('cooldown');
    const availableReturns = returnSites.filter(s => s.level === level);
    if (!availableReturns.length) return reject('no-return');
    const landing = { ...position, y: floorY(level) };
    const candidates = actors.filter(a => a.active !== false && !['arriving', 'departing'].includes(a.populationPhase) &&
      a._state === 'idle' && a.support?.type === 'floor' && a.level === level &&
      !a._bookJobId && !a._lifeJobId && !a._weatherJobId && a._atDestination && !a._bridge && !a._exitLease && !a._liftId &&
      !(a._bookCooldownUntil > now)).map(a => ({ actor: a, ...bookPickupRoute(a, landing) }))
      .sort((a, b) => a.length - b.length || a.actor.index - b.actor.index);
    if (!candidates.length) return reject('no-helper');
    const chosen = candidates[0], actor = chosen.actor;
    const destinations = availableReturns.map(site => ({ site, path: returnRoute(chosen.stance, level, site) }))
      .sort((a, b) => pathLength(a.path) - pathLength(b.path) || a.site.id.localeCompare(b.site.id));
    const destination = destinations[0], site = destination.site;
    const id = `book-spill-${++sequence}`, stableSeed = Number.isFinite(seed) ? seed >>> 0 : (Math.imul(sequence, 2654435761) ^ Math.round(landing.x * 997) ^ Math.round(landing.z * 991)) >>> 0;
    const job = { id, level, position: landing, createdAt: now, phase: 'notice', phaseStartedAt: now, progress: 0,
      actorId: actor.id, stance: chosen.stance, collected: 0, returnSiteId: site.id, shelfPosition: { ...site.shelfPosition },
      shelfYaw: Number.isFinite(site.shelfYaw) ? site.shelfYaw : -site.angle,
      shelfFacing: Number.isFinite(site.facing) ? site.facing : Math.atan2(site.shelfPosition.x - site.position.x, site.shelfPosition.z - site.position.z),
      books: bookLandingLayout(landing, stableSeed).map(b => ({ ...b, id: `${id}-${b.index}`, state: 'falling', progress: 0, phaseStartedAt: now })),
      approachPath: chosen.path, returnPath: destination.path, approachLength: chosen.length, returnLength: pathLength(destination.path),
      resumeDestinationId: actor.destination.id, finishedAt: null };
    jobs.push(job); lastAcceptedAt = now; actor._bookJobId = job.id;
    // Keep the old destination reserved until the helper returns to ordinary
    // life. It cannot be occupied underneath their resumed destination.
    actor._atDestination = null; actor._initialCenter = false;
    actor.purpose = words.purposes.noticeBooks; setState(actor, 'book_notice'); still(actor, 0, 'notice'); presentation(job, actor);
    stats.accepted++; stats.maxActive = Math.max(stats.maxActive, jobs.filter(active).length);
    stats.longestApproach = Math.max(stats.longestApproach, job.approachLength);
    stats.longestReturn = Math.max(stats.longestReturn, job.returnLength);
    return { ok: true, jobId: id, actorId: actor.id };
  }
  function beginPickup(job, actor) {
    job.aligning = true; phase(job, actor, 'pickup'); setState(actor, 'book_pickup');
    const book = job.books[job.collected]; book.state = 'picking'; book.progress = 0; book.phaseStartedAt = getTime();
    actor.purpose = words.purposes.pickingUp(job.collected + 1); still(actor, 0, 'pickup');
  }
  function pathDone(actor, after) {
    if (typeof after !== 'string' || !after.startsWith('book_')) return false;
    const job = findJob(actor); if (!job) return false;
    if (after === 'book_pickup') beginPickup(job, actor);
    else if (after === 'book_shelve') {
      job.aligning = true; phase(job, actor, 'shelve'); setState(actor, 'book_shelve');
      for (const book of job.books) { book.state = 'shelving'; book.progress = 0; book.phaseStartedAt = getTime(); }
      actor.purpose = words.purposes.shelving; still(actor, 0, 'shelve');
    } else return false;
    return true;
  }
  function finish(job, actor) {
    const now = getTime(), site = returnById.get(job.returnSiteId);
    for (const book of job.books) { book.state = 'shelved'; book.progress = 1; book.phaseStartedAt = now; }
    job.phase = 'complete'; job.progress = 1; job.phaseStartedAt = now; job.finishedAt = now;
    stats.completed++; stats.returned += 3; stats.longestCompletion = Math.max(stats.longestCompletion, now - job.createdAt);
    stats.recent.push({ id: job.id, actorId: actor.id, level: job.level, seconds: now - job.createdAt, returned: 3 });
    if (stats.recent.length > 12) stats.recent.shift();
    delete actor.bookTask; delete actor._bookJobId; actor._bookCooldownUntil = now + BOOK_SPILL_LIMITS.helperRest;
    actor._journeyStarted ??= now; actor.purpose = words.purposes.tidied;
    // Exit the furnishing bay by the same clear approach used to enter. The
    // production planner then resumes the original, still-reserved destination.
    const path = []; let p = copyPoint(actor.position);
    for (const next of [...site.approach].reverse()) p = appendLine(path, p, next, { type: 'floor', level: actor.level }, .85);
    setPath(actor, path, 'plan');
  }
  function tickActor(actor, dt) {
    const job = findJob(actor); if (!job || !active(job)) return false;
    const elapsed = Math.max(0, getTime() - job.phaseStartedAt);
    if (job.phase === 'notice') {
      job.progress = clamp(elapsed / BOOK_TASK_TIMINGS.notice); presentation(job, actor);
      still(actor, dt, 'notice'); turn(actor, Math.atan2(job.position.x - actor.position.x, job.position.z - actor.position.z), dt);
      if (elapsed >= BOOK_TASK_TIMINGS.notice) {
        phase(job, actor, 'approach'); actor.purpose = words.purposes.toBooks; setPath(actor, job.approachPath, 'book_pickup');
      }
      return true;
    }
    if (job.phase === 'pickup') {
      const book = job.books[job.collected], facing = Math.atan2(book.position.x - actor.position.x, book.position.z - actor.position.z);
      still(actor, dt, 'pickup'); turn(actor, facing, dt);
      if (job.aligning) {
        job.progress = 0;
        if (Math.abs(shortest(actor.yaw, facing)) <= .015) { job.aligning = false; job.phaseStartedAt = getTime(); book.phaseStartedAt = getTime(); }
        presentation(job, actor); return true;
      }
      job.progress = clamp(elapsed / BOOK_TASK_TIMINGS.pickup); book.progress = job.progress; presentation(job, actor);
      if (elapsed >= BOOK_TASK_TIMINGS.pickup) {
        book.state = 'carried'; book.progress = 1; book.phaseStartedAt = getTime(); job.collected++; stats.collected++;
        if (job.collected < 3) beginPickup(job, actor);
        else { phase(job, actor, 'carry'); actor.purpose = words.purposes.carrying; setPath(actor, job.returnPath, 'book_shelve'); }
      }
      return true;
    }
    if (job.phase === 'shelve') {
      still(actor, dt, 'shelve'); turn(actor, job.shelfFacing, dt);
      if (job.aligning) {
        job.progress = 0;
        if (Math.abs(shortest(actor.yaw, job.shelfFacing)) <= .015) {
          job.aligning = false; job.phaseStartedAt = getTime(); for (const book of job.books) book.phaseStartedAt = getTime();
        }
        presentation(job, actor); return true;
      }
      job.progress = clamp(elapsed / BOOK_TASK_TIMINGS.shelve); presentation(job, actor);
      for (const book of job.books) book.progress = job.progress;
      if (elapsed >= BOOK_TASK_TIMINGS.shelve) finish(job, actor);
      return true;
    }
    presentation(job, actor); return false;
  }
  function update() {
    const now = getTime();
    for (const job of jobs) for (const book of job.books) if (book.state === 'falling') {
      book.progress = clamp((now - job.createdAt) / BOOK_TASK_TIMINGS.fall);
      if (book.progress >= 1) { book.state = 'ground'; book.phaseStartedAt = now; }
    }
    jobs = jobs.filter(j => active(j) || now - j.finishedAt < BOOK_TASK_TIMINGS.complete);
  }
  function snapshot() { return clone({ version: 1, sequence, lastAcceptedAt, jobs, stats }); }
  function validPath(path, level) {
    return Array.isArray(path) && path.every(s => s && ['line', 'arc'].includes(s.kind) && s.support?.type === 'floor' && s.support.level === level &&
      Number.isFinite(s.length) && s.length >= 0 && Number.isFinite(s.progress) && s.progress >= 0 && s.progress <= s.length + 1e-7 &&
      Number.isFinite(s.speed) && s.speed > 0 && (s.kind === 'line' ? finitePoint(s.from) && finitePoint(s.to)
        : Number.isFinite(s.radius) && s.radius >= innerRadius(level) && s.radius <= 33 && Number.isFinite(s.fromAngle) && Number.isFinite(s.delta)));
  }
  function validate(data, savedActors, savedReserved = []) {
    if (!data) return !savedActors.some(a => a._bookJobId || a._state?.startsWith('book_'));
    if (data.version !== 1 || !Number.isInteger(data.sequence) || data.sequence < 0 || !Number.isFinite(data.lastAcceptedAt) ||
      !Array.isArray(data.jobs) || data.jobs.length > BOOK_SPILL_LIMITS.maxJobs || !data.stats || typeof data.stats !== 'object') return false;
    const seen = new Set(), assigned = new Set(), floors = new Set(), reservations = new Map(savedReserved);
    for (const job of data.jobs) {
      if (!job || typeof job.id !== 'string' || seen.has(job.id) || !PHASES.includes(job.phase) || !Number.isInteger(job.level) || job.level < 0 || job.level > 6 || !finitePoint(job.position) || !finitePoint(job.stance) ||
        !finitePoint(job.shelfPosition) || !Number.isFinite(job.shelfYaw) || !Number.isFinite(job.shelfFacing) || !Number.isFinite(job.createdAt) || !Number.isFinite(job.phaseStartedAt) ||
        !Number.isFinite(job.progress) || !Number.isInteger(job.collected) || job.collected < 0 || job.collected > 3 || !returnById.has(job.returnSiteId) ||
        !Array.isArray(job.books) || job.books.length !== 3 || !validPath(job.approachPath, job.level) || !validPath(job.returnPath, job.level)) return false;
      seen.add(job.id);
      const actor = savedActors.find(a => a.id === job.actorId);
      if (!actor) return false;
      if (active(job)) {
        if (assigned.has(job.actorId) || floors.has(job.level) || actor._lifeJobId || actor._weatherJobId || actor._bookJobId !== job.id || actor.bookTask?.jobId !== job.id || actor.level !== job.level || actor.support?.type !== 'floor' ||
          actor.destination?.id !== job.resumeDestinationId || reservations.get(job.resumeDestinationId) !== job.actorId) return false;
        if (['notice', 'pickup', 'shelve'].includes(job.phase) && actor._state !== `book_${job.phase}`) return false;
        if (['approach', 'carry'].includes(job.phase) && (actor._state !== 'path' || !validPath(actor._path, job.level) ||
          actor._after !== (job.phase === 'approach' ? 'book_pickup' : 'book_shelve'))) return false;
        assigned.add(job.actorId); floors.add(job.level);
      } else if (!Number.isFinite(job.finishedAt)) return false;
      if (job.books.some((b, index) => b.index !== index || typeof b.id !== 'string' || !BOOK_STATES.includes(b.state) || !finitePoint(b.position) ||
        !Number.isFinite(b.yaw) || !Number.isFinite(b.progress) || !Number.isFinite(b.phaseStartedAt))) return false;
      if (job.phase === 'pickup' && job.collected >= 3) return false;
      if (['notice', 'approach'].includes(job.phase) && job.collected !== 0) return false;
      if (['carry', 'shelve', 'complete'].includes(job.phase) && job.collected !== 3) return false;
      const stateFor = index => job.phase === 'pickup' ? index < job.collected ? 'carried' : index === job.collected ? 'picking' : 'ground'
        : job.phase === 'carry' ? 'carried' : job.phase === 'shelve' ? 'shelving' : job.phase === 'complete' ? 'shelved' : null;
      if (job.books.some((b, index) => stateFor(index) ? b.state !== stateFor(index) : !['falling', 'ground'].includes(b.state))) return false;
    }
    return savedActors.every(a => !a._bookJobId || assigned.has(a.id));
  }
  function restore(data) {
    if (!data) { jobs = []; sequence = 0; lastAcceptedAt = -1e6; stats = initialStats(); return; }
    const saved = clone(data); jobs = saved.jobs; sequence = saved.sequence; lastAcceptedAt = saved.lastAcceptedAt;
    stats = { ...initialStats(), ...saved.stats };
    for (const job of jobs) if (active(job)) {
      const actor = actorById.get(job.actorId);
      // setPath keeps the job's actual segment objects. JSON intentionally
      // duplicates them, so restore that relationship before the next step.
      if (job.phase === 'approach') job.approachPath = actor._path;
      if (job.phase === 'carry') job.returnPath = actor._path;
      presentation(job, actor);
    }
  }
  return { request, tickActor, pathDone, update, snapshot, validate, restore,
    get jobs() { return jobs; }, get stats() { return stats; }, get returnSites() { return returnSites; } };
}
