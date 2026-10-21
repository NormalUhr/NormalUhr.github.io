import { isCameraHitVisible } from './camera_collision.js';
import { clamp01, createPaperView } from './paper_view.js';

const smooth = t => { t = clamp01(t); return t * t * (3 - 2 * t); };
const copy = p => ({ x: p.x, y: p.y, z: p.z });
const finitePoint = p => !!p && [p.x, p.y, p.z].every(Number.isFinite);
const clone = value => JSON.parse(JSON.stringify(value));
const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
const live = job => job && job.phase !== 'complete' && !job.cancelled;

/**
 * Right-click or long press, judged by the first visible surface under the pointer. Open gallery floor
 * keeps the book spill, a return tray or a desk takes a folded letter, and the water court takes a paper
 * boat. Anything else sends a paper plane. A plane leaves the hand at once. When a reader on a nearby
 * floor is free, it is guided to that reader's desk, and otherwise it glides freely, turns away from
 * walls and lands softly. A letter is made only after a resident has been booked to read it. At most
 * four papers rest at once, two of them boats, and six planes fly at once.
 */
export function createPaperInteractions(THREE, scene, { camera, element, collision, flightCollision = null, simulation, stations = [],
  bookTrays = null, readingDesks = [], garden = null, groundSurface = null, blockers = [], clippingPlanes = [], capacity = 4 } = {}) {
  const group = new THREE.Group();
  group.name = 'Finite paper interactions'; scene.add(group);
  const raycaster = new THREE.Raycaster(), groundRay = new THREE.Raycaster(), pointer = new THREE.Vector2();
  const down = new THREE.Vector3(0, -1, 0), up = new THREE.Vector3(0, 1, 0);
  const cornerA = new THREE.Vector3(), cornerB = new THREE.Vector3(), cornerC = new THREE.Vector3(), normal = new THREE.Vector3();
  const objectMatrix = new THREE.Matrix4(), instanceMatrix = new THREE.Matrix4(), probe = new THREE.Vector3(), direction = new THREE.Vector3();
  const turn = new THREE.Quaternion(), orientation = new THREE.Quaternion(), euler = new THREE.Euler();
  const sceneMaterials = new Set(), materials = [], views = new Map();
  scene.traverse(object => {
    for (const mat of Array.isArray(object.material) ? object.material : object.material ? [object.material] : []) sceneMaterials.add(mat);
  });
  const water = (garden?.meshes || []).filter(mesh => String(mesh.material?.name).includes('/ quiet water'));
  const groundY = groundSurface?.getWorldPosition(new THREE.Vector3()).y;
  const gardenRoots = garden?.group ? [garden.group] : water, trayRoots = bookTrays?.group ? [bookTrays.group] : [];
  const pool = garden?.features?.pool || { innerRadius: 5.95, outerRadius: 8.45, waterY: -1.58 };
  let objects = [], time = 0, nextAt = -1, nextPlaneAt = -1, serial = 0, section = false, disposed = false, actors = [], pending = null;
  let curve = null, results = [];
  const paperCapacity = Math.max(1, Math.min(4, capacity)), planeCapacity = 6, noPlanes = [];
  const stats = { attempts: 0, accepted: 0, boats: 0, letters: 0, planes: 0, completed: 0, stored: 0, collisionLandings: 0, maximumObjects: 0,
    rejected: {}, cancelledJobs: 0, physicalQueries: 0, planningFrames: 0, maxPlanningMilliseconds: 0, pendingCancelled: 0,
    freeFlights: 0, guidedFlights: 0, fallbackFlights: 0, softLandings: 0, retiredPlanes: 0 };
  const stationById = new Map(stations.map(station => [station.id, station]));
  const paperJobs = () => simulation?.lifeState?.paperJobs || [], jobOf = id => paperJobs().find(job => (job.jobId || job.id) === id);

  function refuse(reason, kind = null, handled = true) {
    return { handled, ok: false, kind, reason };
  }
  /** Collision checks must not depend on the section view, so clipping is lifted from every scene material while they run. */
  function unclipped(check) {
    const saved = [];
    for (const mat of sceneMaterials) if (mat.clippingPlanes?.length) { saved.push([mat, mat.clippingPlanes]); mat.clippingPlanes = []; }
    try {
      return check();
    } finally {
      for (const [mat, planes] of saved) mat.clippingPlanes = planes;
    }
  }
  /** The nearest surface along a ray, from the tower's collision geometry, the city's surfaces or the ground. */
  function nearest(origin, toward, far, near = .001) {
    stats.physicalQueries++;
    let hit = collision?.nearest(origin, toward, near, far) || null;
    const cityHit = flightCollision?.nearest(origin, toward, near, hit?.distance ?? far);
    if (cityHit && (!hit || cityHit.distance < hit.distance)) hit = cityHit;
    // The ground is one large mesh, so it is tested only when the ray reaches the ground's height within range.
    const groundDistance = groundSurface && Math.abs(toward.y) > 1e-9 ? (groundY - origin.y) / toward.y : Infinity;
    if (groundSurface && groundDistance >= near && groundDistance <= (hit?.distance ?? far)) {
      groundRay.set(origin, toward); groundRay.near = near; groundRay.far = hit?.distance ?? far;
      const groundHit = groundRay.intersectObject(groundSurface, false).find(isCameraHitVisible);
      if (groundHit && (!hit || groundHit.distance < hit.distance)) hit = groundHit;
    }
    return hit;
  }
  /** The world normal of the hit triangle, so that only upward faces count as a desk top or a floor. */
  function faceNormal(hit) {
    const geometry = hit.object?.geometry, position = geometry?.attributes.position, index = geometry?.index;
    if (!position || !Number.isInteger(hit.faceIndex)) return normal.set(0, 0, 0);
    objectMatrix.copy(hit.object.matrixWorld);
    if (hit.instanceId !== undefined) { hit.object.getMatrixAt(hit.instanceId, instanceMatrix); objectMatrix.multiply(instanceMatrix); }
    const first = hit.faceIndex * 3;
    for (const [corner, k] of [[cornerA, 0], [cornerB, 1], [cornerC, 2]]) {
      corner.fromBufferAttribute(position, index ? index.getX(first + k) : first + k).applyMatrix4(objectMatrix);
    }
    return normal.subVectors(cornerB, cornerA).cross(cornerC.sub(cornerA)).normalize();
  }
  function isInside(object, root) {
    for (let node = object; node; node = node.parent) if (node === root) return true;
    return false;
  }
  /** The first surface along a ray, from the collision geometry or from any nearer visible garden, tray, blocker or paper. */
  function firstVisible(origin, toward, far = 900) {
    let hit = nearest(origin, toward, far, .015);
    raycaster.set(origin, toward.clone().normalize()); raycaster.near = .015; raycaster.far = hit?.distance ?? far;
    const seen = raycaster.intersectObjects([...gardenRoots, ...trayRoots, ...blockers, ...[...views.values()].map(view => view.group)], true)
      .find(isCameraHitVisible);
    if (seen && (!hit || seen.distance < hit.distance + 1e-7)) hit = seen;
    return hit;
  }
  function pickRay(origin, toward, far = 900) {
    if (!finitePoint(origin) || !finitePoint(toward) || !(toward.length?.() > 1e-8 || Math.hypot(toward.x, toward.y, toward.z) > 1e-8)) {
      return refuse('invalid-pointer');
    }
    const from = new THREE.Vector3(origin.x, origin.y, origin.z), ray = new THREE.Vector3(toward.x, toward.y, toward.z).normalize();
    const hit = firstVisible(from, ray, far);
    const plane = () => ({ handled: true, ok: true, kind: 'plane', origin: copy(from), direction: copy(ray), hitDistance: hit?.distance ?? null });
    if (!hit) return plane();
    if (water.includes(hit.object)) {
      // A click near the pool's rim puts the boat on the water just inside it.
      const radius = Math.hypot(hit.point.x, hit.point.z), afloat = clamp(radius, pool.innerRadius + .24, pool.outerRadius - .24),
        scale = afloat / Math.max(1e-6, radius);
      return { handled: true, ok: true, kind: 'boat', position: { x: hit.point.x * scale, y: pool.waterY, z: hit.point.z * scale }, distance: hit.distance };
    }
    if (bookTrays?.group && isInside(hit.object, bookTrays.group) && faceNormal(hit).y > .985) {
      for (const station of stations) {
        const dx = hit.point.x - station.shelfPosition.x, dz = hit.point.z - station.shelfPosition.z;
        const along = dx * Math.cos(station.mountingAngle) + dz * Math.sin(station.mountingAngle);
        const across = dx * Math.sin(station.mountingAngle) - dz * Math.cos(station.mountingAngle);
        if (Math.abs(hit.point.y - station.shelfPosition.y) < .012 && Math.abs(along) < .37 && Math.abs(across) < .33) {
          // Anywhere on the tray, the letter is laid down near its middle.
          const x = clamp(along, -.2, .2), z = clamp(across, -.15, .15), angle = station.mountingAngle;
          return { handled: true, ok: true, kind: 'letter', stationId: station.id,
            position: { x: station.shelfPosition.x + Math.cos(angle) * x + Math.sin(angle) * z, y: hit.point.y + .002,
              z: station.shelfPosition.z + Math.sin(angle) * x - Math.cos(angle) * z },
            distance: hit.distance };
        }
      }
      return plane();
    }
    if (faceNormal(hit).y > .985) for (const desk of readingDesks) {
      const center = desk.center, angle = desk.mountingAngle ?? 0;
      if (!center || Math.abs(hit.point.y - (desk.topY ?? center.y)) > .014 ||
        desk.surfaceMaterialName && (hit.material || hit.object?.material)?.name !== desk.surfaceMaterialName) continue;
      const dx = hit.point.x - center.x, dz = hit.point.z - center.z;
      const along = dx * Math.cos(angle) + dz * Math.sin(angle), across = dx * Math.sin(angle) - dz * Math.cos(angle);
      const onTop = Number.isFinite(desk.radius) ? Math.hypot(along, across) <= desk.radius - .008
        : Math.abs(along) <= desk.width / 2 - .008 && Math.abs(across) <= desk.depth / 2 - .008;
      if (onTop && stationById.has(desk.stationId)) {
        return { handled: true, ok: true, kind: 'letter', stationId: desk.stationId,
          position: copy(desk.paperPosition || { x: hit.point.x, y: hit.point.y + .002, z: hit.point.z }), distance: hit.distance, source: 'reading-desk' };
      }
    }
    const level = Math.round(hit.point.y / 6);
    if (level >= 0 && level <= 6 && Math.abs(hit.point.y - level * 6) < .045 && faceNormal(hit).y > .985) {
      return { handled: false, ok: false, kind: 'books', reason: 'gallery-floor' };
    }
    return plane();
  }
  function pick(clientX, clientY) {
    const rect = element?.getBoundingClientRect(), view = typeof camera === 'function' ? camera() : camera;
    if (!rect || !view || !(rect.width > 0 && rect.height > 0) || !Number.isFinite(clientX + clientY) || clientX < rect.left ||
      clientY < rect.top || clientX > rect.left + rect.width || clientY > rect.top + rect.height) return refuse('invalid-pointer');
    pointer.set((clientX - rect.left) / rect.width * 2 - 1, -(clientY - rect.top) / rect.height * 2 + 1);
    view.updateMatrixWorld(); raycaster.setFromCamera(pointer, view);
    return pickRay(raycaster.ray.origin, raycaster.ray.direction, view.far);
  }
  /** A plane can be thrown along any ray, whatever it points at. */
  function planeRay(origin, toward, far = 900) {
    if (!finitePoint(origin) || !finitePoint(toward) || Math.hypot(toward.x, toward.y, toward.z) < 1e-8) return refuse('invalid-pointer');
    const from = new THREE.Vector3(origin.x, origin.y, origin.z), ray = new THREE.Vector3(toward.x, toward.y, toward.z).normalize();
    const hit = firstVisible(from, ray, far);
    return { handled: true, ok: true, kind: 'plane', origin: copy(from), direction: copy(ray), hitDistance: hit?.distance ?? null };
  }
  function pickPlane(clientX, clientY) {
    const rect = element?.getBoundingClientRect(), view = typeof camera === 'function' ? camera() : camera;
    if (!rect || !view || !(rect.width > 0 && rect.height > 0) || !Number.isFinite(clientX + clientY) || clientX < rect.left ||
      clientY < rect.top || clientX > rect.left + rect.width || clientY > rect.top + rect.height) return refuse('invalid-pointer');
    pointer.set((clientX - rect.left) / rect.width * 2 - 1, -(clientY - rect.top) / rect.height * 2 + 1);
    view.updateMatrixWorld(); raycaster.setFromCamera(pointer, view);
    return planeRay(raycaster.ray.origin, raycaster.ray.direction, view.far);
  }

  const curveThrough = points => new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p.x, p.y, p.z)), false, 'centripetal');
  /** The top of the landing desk, near its middle and just under a plane coming down onto it, is where the flight ends, not an obstacle. */
  function landsOnDesk(hit, landing, to) {
    return !!(hit && landing && Math.hypot(hit.point.x - landing.x, hit.point.z - landing.z) < .3 && Math.abs(hit.point.y - (landing.y - .002)) < .015 &&
      Math.hypot(to.x - landing.x, to.z - landing.z) < .55 && to.y >= landing.y - .006 && faceNormal(hit).y > .985);
  }
  /** The first obstacle between two points of a flight, along seven parallel lines that span the plane. */
  function blockedBetween(from, to, landing = null) {
    direction.subVectors(to, from);
    const length = direction.length();
    if (length < 1e-8) return null;
    direction.divideScalar(length);
    for (const offset of [[0, 0, 0], [.1, 0, 0], [-.1, 0, 0], [0, .04, 0], [0, -.025, 0], [0, 0, .1], [0, 0, -.1]]) {
      probe.copy(from).add(new THREE.Vector3(...offset));
      const hit = nearest(probe, direction, length + .018);
      if (hit && !landsOnDesk(hit, landing, to)) return hit;
    }
    return null;
  }
  function clearFlight(path, landing) {
    const length = path.getLength();
    if (!(length > .2 && length < 250)) return false;
    const steps = Math.max(8, Math.ceil(length / .36));
    let previous = path.getPointAt(0);
    for (let i = 1; i <= steps; i++) {
      const next = path.getPointAt(i / steps);
      if (blockedBetween(previous, next, landing)) return false;
      previous = next;
    }
    return true;
  }
  /** Desks ordered by how near their floor is to the point 1.4 m along the ray, then by distance across the floor. */
  function stationsByReach(origin, toward) {
    const reach = new THREE.Vector3(origin.x, origin.y, origin.z).addScaledVector(new THREE.Vector3(toward.x, toward.y, toward.z).normalize(), 1.4);
    const level = clamp(Math.floor((reach.y - 1.4) / 6), 0, 6);
    return stations.slice().sort((a, b) => Math.abs(a.level - level) - Math.abs(b.level - level) ||
      Math.hypot(a.shelfPosition.x - reach.x, a.shelfPosition.z - reach.z) - Math.hypot(b.shelfPosition.x - reach.x, b.shelfPosition.z - reach.z));
  }
  /**
   * Three candidate flights to one desk, straight or swinging a little to either side. A plane thrown from
   * outside first climbs above the tower and drops into the open centre. Every flight ends above the place
   * where the reader will stand and comes straight down onto the desk.
   */
  function flightVariants(origin, toward, station, launch = null) {
    const start = launch ? new THREE.Vector3(launch.x, launch.y, launch.z)
      : new THREE.Vector3(origin.x, origin.y, origin.z).addScaledVector(new THREE.Vector3(toward.x, toward.y, toward.z).normalize(), 1.4);
    const ahead = start.clone().addScaledVector(new THREE.Vector3(toward.x, toward.y, toward.z).normalize(), 2.1);
    const outside = Math.hypot(start.x, start.z) > 25;
    const desk = new THREE.Vector3(station.shelfPosition.x, station.shelfPosition.y + .002, station.shelfPosition.z);
    const deskAngle = Math.atan2(desk.x, -desk.z), cruise = station.level * 6 + 2.3;
    const ring = (radius, angle, y) => new THREE.Vector3(Math.sin(angle) * radius, y, -Math.cos(angle) * radius);
    return [0, .36, -.36].map(swing => {
      const points = [start, ahead];
      if (outside) {
        const high = Math.max(58, ahead.y + 3), wide = Math.max(39, Math.hypot(ahead.x, ahead.z)), aheadAngle = Math.atan2(ahead.x, -ahead.z);
        points.push(ring(wide, aheadAngle, high), ring(17.8, deskAngle + swing, high), ring(17.8, deskAngle + swing, cruise + 3));
      } else if (Math.abs(start.y - cruise) > 4.5) {
        points.push(ring(17.8, deskAngle + swing, start.y + .6), ring(17.8, deskAngle + swing, cruise + 2.8));
      }
      points.push(ring(19.8, deskAngle, cruise), ring(26.9, deskAngle, cruise), ring(29.8, deskAngle, cruise));
      const stand = station.approach?.at(-1);
      if (stand) points.push(new THREE.Vector3(stand.x, cruise, stand.z));
      points.push(desk.clone().add(new THREE.Vector3(0, .42, 0)), desk.clone().add(new THREE.Vector3(0, .13, 0)), desk);
      return points.map(copy);
    });
  }
  /** Plans flights in one go. Clicks never use this. It serves checks of the route planner. */
  function planFlights(origin, toward, only = null) {
    const plans = [];
    for (const station of only ? [only] : stationsByReach(origin, toward).slice(0, 9)) {
      for (const points of flightVariants(origin, toward, station)) {
        const path = curveThrough(points), length = path.getLength(), landing = points.at(-1);
        if (unclipped(() => clearFlight(path, landing))) {
          plans.push({ station, points, length, duration: Math.max(3.5, length / 7.4) });
          break;
        }
      }
      if (plans.length >= 3) break;
    }
    return plans;
  }

  function finishPlanning(result) {
    if (!pending) return;
    const plane = objects.find(object => object.id === pending.objectId);
    if (!result.ok && live(jobOf(pending.jobId))) {
      simulation.cancelPaperInteraction?.(pending.jobId); stats.cancelledJobs++; stats.pendingCancelled++;
    }
    // A plane whose guided flight was not found in time flies freely instead.
    if (!result.ok && plane && !disposed) { startFreeFlight(plane); stats.fallbackFlights++; result = { ok: true, id: plane.id, freeFlight: true }; }
    const outcome = { handled: true, kind: 'plane', pending: false, requestId: pending.id, actorId: plane?.actorId ?? null, jobId: plane?.jobId ?? null, ...result };
    results.push(outcome);
    if (results.length > 8) results.shift();
    pending = null; curve = null;
  }
  /**
   * Where a plane leaves the hand, 1.4 m along the ray or a little before the surface it points at. The
   * plane is made smaller there in proportion to the room around it, so it starts clear of walls.
   */
  function launchPose(choice) {
    const ray = new THREE.Vector3(choice.direction.x, choice.direction.y, choice.direction.z).normalize();
    const reach = Number.isFinite(choice.hitDistance) ? Math.max(.018, Math.min(1.4, choice.hitDistance * .6)) : 1.4;
    const position = new THREE.Vector3(choice.origin.x, choice.origin.y, choice.origin.z).addScaledVector(ray, reach);
    let room = .28;
    for (const axis of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      const hit = nearest(position, new THREE.Vector3(...axis), .28, .001);
      if (hit) room = Math.min(room, hit.distance);
    }
    return { position: copy(position), direction: copy(ray), scale: clamp((room - .009) / .245, .06, 1) };
  }
  function startFreeFlight(plane) {
    plane.phase = 'free-flight'; plane.freeFlight = true; plane.points = null; plane.jobId = null; plane.actorId = null; plane.stationId = null;
    plane.position = copy(plane.lastPosition || plane.launchPosition || plane.position); plane.cancelAge = null; plane.freeStartedAt = plane.age;
    plane.freeDuration = 11 + plane.seed % 4; plane.heading = copy(plane.heading || plane.launchDirection || { x: 0, y: 0, z: 1 });
    plane.speed = .65; plane.turnSign = Math.sin(plane.seed) > 0 ? 1 : -1;
    const view = views.get(plane.id);
    if (view) view.curve = null;
    stats.freeFlights++;
  }
  /** Makes room for a new plane by fading out one already in the air. */
  function retire(plane) {
    if (plane.retireAge !== undefined) return;
    if (pending?.objectId === plane.id) finishPlanning({ ok: false, reason: 'newer-plane' });
    if (plane.jobId && live(jobOf(plane.jobId))) { simulation.cancelPaperInteraction?.(plane.jobId); stats.cancelledJobs++; startFreeFlight(plane); }
    plane.retireAge = 0; stats.retiredPlanes++;
  }
  /**
   * A plane leaves the hand at once. When the throw is not cramped and a reader on a nearby floor is free,
   * the reader is booked and a guided flight to that desk is planned during the next second. Otherwise the
   * plane flies freely.
   */
  function launchPlane(choice) {
    if (time < nextPlaneAt) return refuse('paper-cooldown', 'plane');
    if (objects.filter(object => object.kind === 'plane').length >= planeCapacity + 1) return refuse('paper-capacity', 'plane');
    const flying = objects.filter(object => object.kind === 'plane' && object.retireAge === undefined);
    if (flying.length >= planeCapacity) {
      const older = flying.find(plane => plane.freeFlight || ['launching', 'flying'].includes(plane.phase));
      if (!older) return refuse('paper-capacity', 'plane');
      retire(older);
    }
    const launch = launchPose(choice);
    const plane = { id: `paper-${++serial}`, kind: 'plane', createdAt: time, age: 0, position: copy(launch.position), phase: 'launching',
      launchPosition: copy(launch.position), launchDirection: copy(launch.direction), heading: copy(launch.direction), launchScale: launch.scale,
      seed: serial * 2.399963, freeFlight: false, lastPosition: copy(launch.position),
      lastQuaternion: new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.asin(clamp(launch.direction.y, -1, 1)),
        Math.atan2(launch.direction.x, launch.direction.z), 0)).toArray(),
      cancelAge: null, jobId: null, actorId: null, stationId: null, points: null, duration: 1, flight: 0, ready: false };
    objects.push(plane); addView(plane); stats.planes++; stats.accepted++;
    stats.maximumObjects = Math.max(stats.maximumObjects, objects.length); nextPlaneAt = time + .32;
    let booking = null, station = null, reason = 'no-idle-reader';
    const busyFloors = new Set();
    for (const job of [...paperJobs(), ...simulation?.bookJobs || []]) if (live(job)) {
      const level = job.level ?? stationById.get(job.stationId)?.level;
      if (Number.isFinite(level)) busyFloors.add(level);
    }
    const guided = !pending && launch.scale > .96 && (!Number.isFinite(choice.hitDistance) || choice.hitDistance > 4);
    for (const candidate of guided ? stationsByReach(choice.origin, choice.direction) : []) {
      if (busyFloors.has(candidate.level)) continue;
      const position = { ...candidate.shelfPosition, y: candidate.shelfPosition.y + .002 };
      booking = simulation?.requestPaperInteraction?.({ kind: 'plane', stationId: candidate.id, position });
      if (booking?.ok) { station = candidate; break; }
      reason = booking?.reason || reason;
      if (['busy', 'cooldown'].includes(reason)) break;
      busyFloors.add(candidate.level);
    }
    if (!station) {
      startFreeFlight(plane); poseObject(plane, 0, null);
      return { handled: true, ok: true, kind: 'plane', id: plane.id, freeFlight: true, actorId: null, jobId: null };
    }
    plane.jobId = booking.jobId; plane.actorId = booking.actorId; plane.stationId = station.id;
    pending = { id: `paper-request-${++serial}`, jobId: booking.jobId, actorId: booking.actorId, stationId: station.id, objectId: plane.id,
      launchPosition: copy(launch.position), origin: copy(choice.origin), direction: copy(choice.direction), createdAt: time, elapsed: 0,
      variants: flightVariants(choice.origin, choice.direction, station, launch.position), candidate: 0, step: 1, offset: 0, length: 0, count: 0 };
    poseObject(plane, 0, null);
    return { handled: true, ok: true, pending: true, kind: 'plane', id: plane.id, requestId: pending.id, actorId: booking.actorId, jobId: booking.jobId };
  }
  const planOffsets = [[0, 0, 0], [.15, 0, 0], [-.15, 0, 0], [0, .055, 0], [0, -.025, 0], [0, 0, .15], [0, 0, -.15]];
  /** One collision query of the pending flight. A single query cannot be split, so it is the unit of planning work. */
  function planStep() {
    if (!pending) return;
    const job = jobOf(pending.jobId);
    // The plane is already in the hand, so a guided flight gets one second of planning before it flies freely.
    if (!live(job) || job.phase !== 'await' || pending.elapsed > 1) { finishPlanning({ ok: false, reason: 'paper-plan-expired' }); return; }
    if (!curve) {
      if (pending.candidate >= pending.variants.length) { finishPlanning({ ok: false, reason: 'no-clear-flight' }); return; }
      curve = curveThrough(pending.variants[pending.candidate]); pending.length = curve.getLength();
      if (!(pending.length > .2 && pending.length < 200)) { pending.candidate++; curve = null; return; }
      pending.count = Math.max(8, Math.ceil(pending.length / .36));
    }
    if (pending.step > pending.count) {
      const points = pending.variants[pending.candidate], landing = copy(points.at(-1));
      // The booked reader waits only so long, so a flight that would land too late is not started.
      if (time - job.createdAt + Math.max(3.5, pending.length / 7.4) + .75 > 43) { finishPlanning({ ok: false, reason: 'paper-plan-expired' }); return; }
      const plane = objects.find(object => object.id === pending.objectId);
      if (!plane) { finishPlanning({ ok: false, reason: 'missing-launch' }); return; }
      Object.assign(plane, { position: landing, phase: 'flying', points: clone(points), duration: Math.max(3.5, pending.length / 7.4), flight: 0,
        ready: false, freeFlight: false });
      views.get(plane.id).curve = curveThrough(points); stats.guidedFlights++;
      poseObject(plane, 0, null); finishPlanning({ ok: true, id: plane.id });
      return;
    }
    const from = curve.getPointAt((pending.step - 1) / pending.count), to = curve.getPointAt(pending.step / pending.count);
    const landing = pending.variants[pending.candidate].at(-1), offset = planOffsets[pending.offset];
    direction.subVectors(to, from);
    const length = direction.length();
    direction.divideScalar(Math.max(1e-9, length)); probe.copy(from).add(new THREE.Vector3(...offset));
    const hit = nearest(probe, direction, length + .025), onDesk = landsOnDesk(hit, landing, to);
    if (hit && !onDesk) { pending.candidate++; pending.step = 1; pending.offset = 0; curve = null; return; }
    pending.offset++;
    if (pending.offset >= planOffsets.length) { pending.offset = 0; pending.step++; }
  }
  /** At most 16 queries and two milliseconds of planning in a frame. */
  function plan(dt) {
    if (!pending) return;
    pending.elapsed += dt;
    const started = performance.now();
    unclipped(() => {
      for (let i = 0; i < 16 && pending; i++) {
        planStep();
        if (performance.now() - started >= 2) break;
      }
    });
    const spent = performance.now() - started;
    stats.planningFrames++; stats.maxPlanningMilliseconds = Math.max(stats.maxPlanningMilliseconds, spent);
  }

  function addView(object) {
    const view = createPaperView(THREE, object.kind, { clippingPlanes: clippingFor(object) });
    if (object.points) view.curve = curveThrough(object.points);
    views.set(object.id, view); group.add(view.group); materials.push(...view.materials); view.group.visible = true;
    return view;
  }
  function removeObject(object) {
    const view = views.get(object.id);
    if (view) {
      view.dispose(); views.delete(object.id);
      for (const mat of view.materials) {
        const index = materials.indexOf(mat);
        if (index >= 0) materials.splice(index, 1);
      }
    }
    objects = objects.filter(other => other !== object); stats.completed++;
  }
  function accept(choice) {
    stats.attempts++;
    if (!choice.ok) return choice;
    if (choice.kind === 'plane') return launchPlane(choice);
    if (time < nextAt) return refuse('paper-cooldown', choice.kind);
    if (objects.filter(object => object.kind !== 'plane').length >= paperCapacity) return refuse('paper-capacity', choice.kind);
    let object;
    if (choice.kind === 'boat') {
      if (objects.filter(other => other.kind === 'boat').length >= 2) return refuse('boat-capacity', 'boat');
      object = { id: `paper-${++serial}`, kind: 'boat', createdAt: time, age: 0, position: copy(choice.position),
        radius: Math.hypot(choice.position.x, choice.position.z), angle: Math.atan2(choice.position.x, -choice.position.z),
        seed: serial * 2.399963, duration: 49 + serial % 3 * 5 };
      stats.boats++;
    } else {
      let booking, landing;
      if (choice.kind === 'letter') {
        const station = stationById.get(choice.stationId);
        if (!station) return refuse('missing-paper-station', 'letter');
        booking = simulation?.requestPaperInteraction?.({ kind: 'letter', stationId: choice.stationId, position: copy(choice.position) });
        landing = { station, position: copy(choice.position) };
      }
      if (!booking?.ok) return refuse(booking?.reason || 'no-idle-reader', choice.kind);
      object = { id: `paper-${++serial}`, kind: choice.kind, createdAt: time, age: 0, jobId: booking.jobId, actorId: booking.actorId,
        stationId: landing.station.id, position: copy(landing.position), phase: choice.kind === 'plane' ? 'flying' : 'settling',
        points: landing.points || null, duration: landing.duration || .65, flight: 0, ready: false, seed: serial * 2.399963,
        lastPosition: choice.kind === 'plane' ? copy(landing.points[0]) : { ...copy(landing.position), y: landing.position.y + .15 },
        lastQuaternion: [0, 0, 0, 1], cancelAge: null };
      if (choice.kind === 'plane') stats.planes++;
      else stats.letters++;
    }
    objects.push(object); addView(object); nextAt = time + 1.8; stats.accepted++;
    stats.maximumObjects = Math.max(stats.maximumObjects, objects.length); poseObject(object, 0, null);
    return { handled: true, ok: true, kind: object.kind, actorId: object.actorId || null, jobId: object.jobId || null, id: object.id };
  }
  function request(clientX, clientY) {
    const result = accept(pick(clientX, clientY));
    if (!result.ok && result.handled) stats.rejected[result.reason] = (stats.rejected[result.reason] || 0) + 1;
    return result;
  }
  /** Lets go of a paper whose reader is gone or never came. Its job is cancelled and the paper fades out. */
  function release(object) {
    if (object.jobId && live(jobOf(object.jobId))) { simulation.cancelPaperInteraction?.(object.jobId); stats.cancelledJobs++; }
    object.cancelAge ??= 0;
  }
  /** Boats and planes in the air are never cut by the section view. Papers at rest indoors are. */
  function clippingFor(object) {
    return section && object.kind !== 'boat' && !object.freeFlight && !['launching', 'flying'].includes(object.phase) ? clippingPlanes : noPlanes;
  }
  /** One step of a free flight, of a soft landing or of the rest after it. Returns where to draw the plane, or null once it has faded. */
  function freeFlightPose(plane, dt) {
    const position = new THREE.Vector3(plane.position.x, plane.position.y, plane.position.z);
    const rotation = new THREE.Quaternion().fromArray(plane.lastQuaternion);
    const size = plane.launchScale ?? 1, flown = plane.age - (plane.freeStartedAt ?? 0);
    let opacity = 1;
    if (plane.phase === 'free-landing') {
      const settled = smooth((plane.age - plane.landAt) / .62);
      position.copy(new THREE.Vector3(plane.landFrom.x, plane.landFrom.y, plane.landFrom.z))
        .lerp(new THREE.Vector3(plane.landTo.x, plane.landTo.y, plane.landTo.z), settled);
      rotation.fromArray(plane.landRotation).slerp(new THREE.Quaternion().fromArray(plane.restRotation), settled);
      if (settled >= 1) plane.phase = 'free-rest';
    } else if (plane.phase === 'free-flight' && dt > 0) {
      const step = Math.min(.1, dt), heading = new THREE.Vector3(plane.heading.x, plane.heading.y, plane.heading.z).normalize();
      // The plane keeps to the direction it was thrown in, sinking a little more as the flight goes on.
      const thrown = plane.launchDirection || plane.heading, glide = new THREE.Vector3(thrown.x, 0, thrown.z);
      if (glide.lengthSq() < .02) glide.set(Math.cos(plane.seed), 0, Math.sin(plane.seed));
      glide.normalize(); glide.y = -.05 - .14 * clamp(flown / 8); glide.normalize();
      heading.lerp(glide, 1 - Math.exp(-step * .7)).normalize();
      const cruise = 3.4, lookAhead = Math.max(1.3, (plane.speed || .65) * .72), ahead = nearest(position, heading, lookAhead);
      let wanted = cruise, landing = null;
      if (ahead) {
        const facing = faceNormal(ahead).clone();
        if (facing.dot(heading) > 0) facing.negate();
        if (facing.y > .6 && heading.y < 0) {
          // Coming down onto a floor or a table, it slows and lands once close.
          wanted = Math.min(wanted, Math.max(.18, (ahead.distance - .08) * 2.3));
          if (ahead.distance < .52) landing = { hit: ahead, normal: facing };
        } else {
          // A wall ahead turns it along the wall and away from it.
          const along = heading.clone().addScaledVector(facing, -heading.dot(facing));
          if (along.lengthSq() < .035) along.crossVectors(up, facing).multiplyScalar(plane.turnSign);
          if (along.lengthSq() < .035) along.set(Math.cos(plane.seed), 0, Math.sin(plane.seed));
          along.normalize().addScaledVector(facing, .95).normalize();
          heading.lerp(along, 1 - Math.exp(-step * 6)).normalize();
          wanted = Math.min(wanted, Math.max(.3, (ahead.distance - .31 * size) * 1.5));
        }
      }
      plane.speed += (wanted - (plane.speed || .65)) * (1 - Math.exp(-step * 3));
      const travel = plane.speed * step, margin = .3 * size;
      let move = travel;
      if (!landing) {
        for (const offset of [[0, 0, 0], [.17, 0, 0], [-.17, 0, 0], [0, .045, 0], [0, -.025, 0], [0, 0, .17], [0, 0, -.17]]) {
          const from = position.clone().addScaledVector(new THREE.Vector3(...offset), size), hit = nearest(from, heading, travel + margin + .012);
          if (hit) move = Math.min(move, Math.max(0, hit.distance - margin - .012));
        }
      }
      if (landing) {
        const rest = landing.hit.point.clone().addScaledVector(landing.normal, .035 * size);
        const restRotation = new THREE.Quaternion().setFromUnitVectors(up, landing.normal)
          .multiply(new THREE.Quaternion().setFromAxisAngle(up, Math.atan2(heading.x, heading.z)));
        plane.phase = 'free-landing'; plane.landAt = plane.age; plane.landFrom = copy(position); plane.landTo = copy(rest);
        plane.landRotation = rotation.toArray(); plane.restRotation = restRotation.toArray(); stats.softLandings++;
      } else {
        position.addScaledVector(heading, move);
        const yaw = Math.atan2(heading.x, heading.z), pitch = -Math.asin(clamp(heading.y, -1, 1));
        const target = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, Math.sin(flown * 1.45 + plane.seed) * .075));
        rotation.slerp(target, 1 - Math.exp(-step * 7));
      }
      plane.heading = copy(heading);
    }
    // A plane fades a few seconds after it lands, or near the end of a flight that never lands.
    if (plane.phase === 'free-rest') opacity = 1 - smooth((plane.age - plane.landAt - 3.3) / 1.25);
    else opacity = 1 - smooth((flown - (plane.freeDuration - 1.8)) / 1.8);
    plane.position = copy(position); plane.lastPosition = copy(position); plane.lastQuaternion = rotation.toArray();
    if (opacity <= 0) { removeObject(plane); return null; }
    return { position, rotation, scale: size, opacity };
  }
  function poseObject(object, dt, residents) {
    const view = views.get(object.id);
    if (!view) return;
    let open = 0, wetness = 0, scale = 1, opacity = 1;
    orientation.identity();
    const position = new THREE.Vector3(object.position.x, object.position.y, object.position.z);
    if (object.kind === 'boat') {
      // A boat drifts on the still water, soaks through once its time is up, and is removed only below the surface.
      const sinking = smooth((object.age - object.duration) / 8), angle = object.angle + Math.sin(object.age * .035) * .055;
      const radius = object.radius + Math.sin(object.age * .09) * .025;
      position.set(Math.sin(angle) * radius, pool.waterY + Math.sin(object.age * 1.15) * .005 - sinking * .25, -Math.cos(angle) * radius);
      orientation.setFromEuler(euler.set(Math.sin(object.age * .8) * .018, angle + Math.PI / 2 + Math.sin(object.age * .14) * .06,
        Math.sin(object.age * .65) * .021));
      wetness = smooth((object.age - object.duration + 12) / 17); scale = smooth(object.age / .22);
      if (object.age > object.duration + 8.3) { removeObject(object); return; }
    } else if (object.freeFlight || object.phase === 'launching') {
      if (object.phase === 'launching') {
        // While its flight is planned, the plane waits where it leaves the hand.
        position.set(object.launchPosition.x, object.launchPosition.y, object.launchPosition.z);
        const thrown = object.launchDirection;
        orientation.setFromEuler(euler.set(-Math.asin(clamp(thrown.y, -1, 1)), Math.atan2(thrown.x, thrown.z), 0)); scale = object.launchScale ?? 1;
      } else {
        const free = freeFlightPose(object, dt);
        if (!free) return;
        position.copy(free.position); orientation.copy(free.rotation); scale = free.scale; opacity = free.opacity;
      }
      scale *= smooth(object.age / .16); object.lastPosition = copy(position); object.lastQuaternion = orientation.toArray();
    } else {
      const job = jobOf(object.jobId), hands = residents?.getLifeAnchors?.(object.actorId), station = stationById.get(object.stationId);
      orientation.setFromEuler(euler.set(0, station?.shelfYaw || 0, 0));
      if (object.phase === 'flying') {
        const path = view.curve, flight = Math.min(1, object.flight + dt / object.duration);
        const from = path.getPointAt(object.flight), to = path.getPointAt(flight);
        // Something moved into the planned flight, so the plane comes down where it is.
        const blocked = dt > 0 ? unclipped(() => blockedBetween(from, to, object.position)) : null;
        if (blocked) {
          stats.lastCollision = { mesh: blocked.object.name, material: blocked.material?.name || blocked.object.material?.name, point: copy(blocked.point),
            normal: copy(faceNormal(blocked)), flight: object.flight, target: copy(object.position), before: copy(from), after: copy(to),
            stationId: object.stationId };
          object.phase = 'grounded'; object.lastPosition = copy(from); object.cancelAge = 0; release(object); stats.collisionLandings++;
          position.copy(from);
        } else {
          object.flight = flight; position.copy(to);
          const tangent = path.getTangentAt(object.flight), heading = Math.atan2(tangent.x, tangent.z);
          const pitch = -Math.asin(clamp(tangent.y, -1, 1)) * smooth((1 - object.flight) * path.getLength() / .9);
          orientation.setFromEuler(euler.set(pitch, heading, Math.sin(object.flight * Math.PI * 4) * .08 * (1 - object.flight)));
          if (flight >= 1) { object.phase = 'settling'; object.settleAt = object.age; object.lastQuaternion = orientation.toArray(); }
        }
        scale = smooth(object.age / .16);
      }
      if (object.phase === 'settling') {
        const since = object.kind === 'letter' ? object.age : object.age - (object.settleAt ?? object.age), settled = smooth(since / .65);
        // A letter drops its last 15 cm onto the desk. A plane turns to lie square on it.
        if (object.kind === 'letter') position.y += .15 * (1 - settled);
        else if (object.lastQuaternion) {
          orientation.fromArray(object.lastQuaternion).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, station?.shelfYaw || 0, 0)), settled);
        }
        if (settled >= 1 && !object.ready) {
          object.ready = !!simulation.markPaperReady?.(object.jobId);
          if (object.ready) object.phase = 'on-table';
          else release(object);
        }
        scale = smooth(object.age / .16);
      }
      if (object.phase === 'grounded') {
        // A plane that came down falls to the first surface below it.
        position.set(object.lastPosition.x, object.lastPosition.y, object.lastPosition.z);
        const below = unclipped(() => nearest(position, down, 80));
        if (below && below.distance > .025) position.y -= Math.min(below.distance - .025, Math.max(.01, dt) * (1 + (object.cancelAge || 0) * 2));
      }
      const phase = job?.phase, progress = clamp(job?.progress || 0);
      if (hands?.center && ['pickup', 'open', 'read', 'fold', 'store', 'return', 'complete'].includes(phase)) {
        // From the pickup on, the paper is held at the reader's solved hands.
        if (phase === 'pickup') {
          const grip = smooth((progress - .4) / .08);
          position.lerp(hands.center.position, grip); orientation.slerp(hands.center.quaternion, grip);
        } else if (['open', 'read', 'fold', 'store'].includes(phase)) {
          position.copy(hands.center.position); orientation.copy(hands.center.quaternion);
        }
        if (phase === 'open') open = smooth(progress);
        else if (phase === 'read') open = 1;
        else if (phase === 'fold') open = 1 - smooth(progress);
        if (phase === 'store') {
          const away = smooth((progress - .55) / .35);
          position.lerp(hands.stash.position, away); scale = 1 - .96 * away; opacity = 1 - smooth((progress - .84) / .13);
          if (progress > .98 && !object.stored) { object.stored = true; stats.stored++; }
        }
        if (['return', 'complete'].includes(phase) && !job.cancelled) { object.stored = true; removeObject(object); return; }
      }
      if ((!job || job.cancelled) && !object.stored && !(hands && ['pickup', 'open', 'read', 'fold', 'store'].includes(phase))) release(object);
      if (object.cancelAge !== null) {
        object.cancelAge += dt;
        if (object.phase === 'flying' && object.age > 38) object.phase = 'grounded';
        if (object.phase !== 'flying') {
          wetness = smooth((object.cancelAge - 3) / 4); opacity = 1 - smooth((object.cancelAge - 4) / 3);
          if (object.cancelAge > 7.1) { removeObject(object); return; }
        }
      }
      object.lastPosition = copy(position); object.lastQuaternion = orientation.toArray();
    }
    if (object.retireAge !== undefined) {
      object.retireAge += dt; opacity *= 1 - smooth(object.retireAge / .25);
      if (object.retireAge >= .25) { removeObject(object); return; }
    }
    view.group.position.copy(position); view.group.quaternion.copy(orientation); view.group.scale.setScalar(Math.max(1e-6, scale));
    view.pose({ open, wetness, flutter: ['flying', 'free-flight', 'launching'].includes(object.phase) ? Math.sin(object.age * 7) * .4 : 0 });
    for (const mat of view.materials) {
      mat.clippingPlanes = clippingFor(object);
      const fading = opacity < .99999;
      if (mat.transparent !== fading) { mat.transparent = fading; mat.needsUpdate = true; }
      mat.opacity = opacity; mat.depthWrite = !fading;
    }
    view.group.updateMatrixWorld(true);
  }
  function update(dt, { time: now, actors: present = [], residents = null } = {}) {
    if (disposed || !Number.isFinite(dt) || dt < 0) return;
    time = Number.isFinite(now) ? now : time + dt; actors = present;
    const step = Math.min(dt, 2);
    plan(step);
    for (const object of [...objects]) { object.age += step; poseObject(object, step, residents); }
  }
  function snapshot() {
    return { version: 1, time, nextAt, nextPlaneAt, serial, objects: clone(objects), pending: pending ? clone(pending) : null, stats: clone(stats) };
  }
  /** Cancels paper jobs that no paper here belongs to, and lets go of papers whose job is gone. */
  function reconcile() {
    const owned = new Set(objects.filter(object => object.kind !== 'boat').map(object => object.jobId));
    if (pending) owned.add(pending.jobId);
    for (const job of paperJobs()) {
      if (live(job) && !owned.has(job.jobId || job.id)) { simulation.cancelPaperInteraction?.(job.jobId || job.id); stats.cancelledJobs++; }
    }
    for (const object of objects) if (object.jobId && !jobOf(object.jobId)) release(object);
  }
  function restore(data) {
    if (!data || data.version !== 1 || !Array.isArray(data.objects) || data.objects.length > paperCapacity + planeCapacity + 1) {
      reconcile(); return false;
    }
    const kept = [], ids = new Set();
    let resumed = null;
    for (const saved of data.objects) {
      if (!saved?.id || ids.has(saved.id) || !['boat', 'letter', 'plane'].includes(saved.kind) || !finitePoint(saved.position) ||
        !Number.isFinite(saved.age) || saved.age < 0) continue;
      const object = clone(saved);
      if (object.kind === 'boat') {
        if (![object.radius, object.angle, object.duration].every(Number.isFinite) || object.radius < pool.innerRadius + .2 ||
          object.radius > pool.outerRadius - .2 || object.age > object.duration + 8.3) continue;
      } else if (object.kind === 'plane' && (object.freeFlight || object.phase === 'launching')) {
        if (!['free-flight', 'free-landing', 'free-rest', 'launching'].includes(object.phase) || !finitePoint(object.lastPosition) ||
          !finitePoint(object.launchPosition) || !finitePoint(object.launchDirection) || !finitePoint(object.heading) ||
          !Number.isFinite(object.launchScale) || object.launchScale < .04 || object.launchScale > 1 || !Array.isArray(object.lastQuaternion) ||
          object.lastQuaternion.length !== 4 || object.lastQuaternion.some(value => !Number.isFinite(value))) continue;
        if (object.freeFlight) {
          if (object.jobId || object.actorId || object.stationId || !Number.isFinite(object.freeStartedAt) || !Number.isFinite(object.freeDuration) ||
            object.freeDuration < 3 || object.freeDuration > 30 || !Number.isFinite(object.speed) ||
            ['free-landing', 'free-rest'].includes(object.phase) && (!finitePoint(object.landFrom) || !finitePoint(object.landTo) ||
              !Number.isFinite(object.landAt) ||
              ![object.landRotation, object.restRotation].every(q => Array.isArray(q) && q.length === 4 && q.every(Number.isFinite)))) continue;
        } else {
          const job = jobOf(object.jobId);
          if (!job || job.actorId !== object.actorId || job.stationId !== object.stationId || !stationById.has(object.stationId)) continue;
        }
      } else {
        const station = stationById.get(object.stationId), job = jobOf(object.jobId);
        if (!station || !job || job.actorId !== object.actorId || job.stationId !== object.stationId ||
          Math.hypot(object.position.x - station.shelfPosition.x, object.position.z - station.shelfPosition.z) > .4 ||
          Math.abs(object.position.y - station.shelfPosition.y) > .12 || !['flying', 'settling', 'on-table', 'grounded'].includes(object.phase) ||
          !Number.isFinite(object.duration) || object.duration <= 0 || object.duration > 40 || !finitePoint(object.lastPosition) ||
          !Array.isArray(object.lastQuaternion) || object.lastQuaternion.length !== 4 || object.lastQuaternion.some(value => !Number.isFinite(value)) ||
          object.kind === 'plane' && (!Array.isArray(object.points) || object.points.length < 3 || object.points.length > 14 ||
            object.points.some(p => !finitePoint(p)) || !Number.isFinite(object.flight) || object.flight < 0 || object.flight > 1)) continue;
      }
      if (object.retireAge !== undefined && (!Number.isFinite(object.retireAge) || object.retireAge < 0 || object.retireAge > .3)) continue;
      kept.push(object); ids.add(object.id);
    }
    if (data.pending) {
      const saved = data.pending, job = jobOf(saved.jobId);
      if (live(job) && job.phase === 'await' && job.actorId === saved.actorId && stationById.has(saved.stationId) && finitePoint(saved.origin) &&
        finitePoint(saved.direction) && Number.isFinite(saved.elapsed) && saved.elapsed < 15) {
        let plane = kept.find(object => object.id === saved.objectId && object.jobId === saved.jobId && object.phase === 'launching');
        if (!saved.objectId) {
          // A request saved before planes left the hand at once gets its waiting plane now.
          const launch = launchPose({ origin: saved.origin, direction: saved.direction, hitDistance: null });
          plane = { id: `paper-restored-${saved.id}`, kind: 'plane', createdAt: data.time || 0, age: 0, position: copy(launch.position), phase: 'launching',
            launchPosition: copy(launch.position), launchDirection: copy(launch.direction), heading: copy(launch.direction), launchScale: launch.scale,
            seed: (data.serial || 1) * 2.399963, freeFlight: false, lastPosition: copy(launch.position), lastQuaternion: [0, 0, 0, 1], cancelAge: null,
            jobId: saved.jobId, actorId: saved.actorId, stationId: saved.stationId, points: null, duration: 1, flight: 0, ready: false };
          kept.push(plane);
        }
        if (plane) {
          resumed = clone(saved);
          Object.assign(resumed, { objectId: plane.id, launchPosition: copy(plane.launchPosition), elapsed: saved.objectId ? Math.min(1.1, saved.elapsed) : 0,
            candidate: saved.objectId ? clamp(Math.floor(saved.candidate || 0), 0, 2) : 0,
            step: saved.objectId ? clamp(Math.floor(saved.step || 1), 1, 3000) : 1,
            offset: saved.objectId ? clamp(Math.floor(saved.offset || 0), 0, 6) : 0 });
          resumed.variants = flightVariants(saved.origin, saved.direction, stationById.get(saved.stationId), plane.launchPosition);
        }
      }
    }
    if (kept.filter(object => object.kind !== 'plane').length > paperCapacity || kept.filter(object => object.kind === 'plane').length > planeCapacity + 1) {
      reconcile(); return false;
    }
    for (const view of views.values()) view.dispose();
    views.clear(); materials.length = 0; objects = kept; pending = resumed; curve = null; results = [];
    time = Number.isFinite(data.time) ? data.time : 0; nextAt = Number.isFinite(data.nextAt) ? data.nextAt : time;
    nextPlaneAt = Number.isFinite(data.nextPlaneAt) ? data.nextPlaneAt : time;
    serial = Math.max(0, data.serial || 0); Object.assign(stats, data.stats || {});
    for (const object of objects) {
      addView(object);
      // A plane that was waiting for a plan that did not come back with it flies freely.
      if (object.phase === 'launching' && pending?.objectId !== object.id) {
        if (live(jobOf(object.jobId))) { simulation.cancelPaperInteraction?.(object.jobId); stats.cancelledJobs++; }
        startFreeFlight(object);
      }
    }
    reconcile();
    return kept.length === data.objects.length + (!data.pending?.objectId && resumed ? 1 : 0);
  }
  return { group, materials, pick, probe: pick, pickRay, request,
    requestRay(origin, toward, far) { return accept(pickRay(origin, toward, far)); },
    requestFallback(clientX, clientY) { return accept(pickPlane(clientX, clientY)); },
    requestFallbackRay(origin, toward, far) { return accept(planeRay(origin, toward, far)); },
    update, snapshot, restore, reconcile,
    takeResults() {
      const done = results;
      results = [];
      return done;
    },
    get pending() { return pending ? clone(pending) : null; },
    setSection(enabled) {
      section = !!enabled;
      for (const object of objects) {
        const view = views.get(object.id);
        for (const mat of view?.materials || []) mat.clippingPlanes = clippingFor(object);
      }
    },
    planPlane(origin, toward) { return planFlights(origin, toward).map(flight => ({ ...flight, station: flight.station.id })); },
    get diagnostics() {
      return objects.map(object => ({ ...clone(object), modelPosition: views.get(object.id)?.group.position.toArray(),
        modelQuaternion: views.get(object.id)?.group.quaternion.toArray() }));
    },
    stats, metrics: { capacity: paperCapacity, boatCapacity: 2, planeCapacity, maximumPlanesIncludingRetirement: planeCapacity + 1, textures: 0, pointLights: 0 },
    dispose() {
      if (disposed) return;
      disposed = true;
      if (pending) finishPlanning({ ok: false, reason: 'disposed' });
      for (const object of objects) release(object);
      for (const view of views.values()) view.dispose();
      views.clear(); objects = []; materials.length = 0; group.removeFromParent();
    } };
}
