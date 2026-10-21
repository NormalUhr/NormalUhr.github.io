import { createCameraMover } from './explore_camera.js';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));

/**
 * A quiet observer, not a camera fixed to a resident's face or walking heading.
 * Only the resident's support position is tracked. A turn, gait phase or step
 * animation never changes the world-space bearing selected by the viewer.
 *
 * Collision recovery has memory: a momentary gap between railing bars cannot
 * repeatedly extend and collapse the camera arm. Very cramped targets retain
 * the observer's safe offset; real swept movement may stop against an obstacle.
 * No alternate side is picked automatically and nothing teleports out of walls.
 */
export function createFollowCamera(THREE, camera, options = {}) {
  const collision = options.collision;
  const mover = createCameraMover(THREE, options);
  const minDistance = Math.max(1.6, finite(options.minDistance, 1.8));
  const maxDistance = Math.max(minDistance, finite(options.maxDistance, 14));
  const minY = finite(options.minY, .65);
  const minElevation = -.18, maxElevation = 1.2;
  const anchor = new THREE.Vector3(), rawAnchor = new THREE.Vector3();
  const oldAnchor = new THREE.Vector3(), anchorTravel = new THREE.Vector3();
  const offset = new THREE.Vector3(), desired = new THREE.Vector3(), movement = new THREE.Vector3();
  const viewDirection = new THREE.Vector3(), cameraRight = new THREE.Vector3();
  let active = false, held = false, obstructed = false, lookHeld = false;
  let yaw = 0, elevation = .28, wantedYaw = 0, wantedElevation = .28;
  let arm = 6, requestedDistance = 6, clearance = maxDistance, clearFor = 0, uncrampedFor = 0;
  let aimYaw = 0, aimPitch = 0, steps = 0, blockedSteps = 0;

  function readAnchor(actor) {
    const p = actor?.position;
    if (!p || ![p.x, p.y, p.z].every(Number.isFinite)) return false;
    // A subject that knows its own eye height, such as the cat, gives it. Residents use their age.
    const height = Number.isFinite(actor.viewHeight) ? clamp(actor.viewHeight, .12, 2.2) : actor.age === 'child' ? 1.02 : 1.48;
    // Seat transitions are filtered with the position; they cannot snap the aim.
    const seated = Number.isFinite(actor.lifeTask?.seated) ? clamp(actor.lifeTask.seated, 0, 1) : actor.activity === 'sit' ? 1 : 0;
    const drop = Number.isFinite(actor.viewHeight) ? 0 : seated * (actor.age === 'child' ? .24 : .42);
    rawAnchor.set(p.x, p.y + height - drop, p.z); return true;
  }

  function findFollowPose(actor, observerPosition) {
    if (!readAnchor(actor)) return null;
    const focus = rawAnchor.clone();
    const ox = finite(observerPosition?.x, focus.x), oz = finite(observerPosition?.z, focus.z + 1);
    const bearing = Math.atan2(ox - focus.x, oz - focus.z);
    const probe = new THREE.Vector3(), position = new THREE.Vector3();
    const directions = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0),
      new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1)];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) directions.push(new THREE.Vector3(x, y, z).normalize());
    const offsets = [0, Math.PI / 4, -Math.PI / 4, Math.PI / 2, -Math.PI / 2, Math.PI, Math.PI * .75, -Math.PI * .75];
    // This discrete search happens only at a deliberate mode transition, never
    // while following. It cannot trigger a succession of automatic angle cuts.
    for (const distance of [5.5, 3.7, 2.6, 1.9]) for (const angle of offsets) for (const tilt of [.16, .06, .38]) {
      const a = bearing + angle, flat = Math.cos(tilt);
      probe.set(Math.sin(a) * flat, Math.sin(tilt), Math.cos(a) * flat);
      position.copy(focus).addScaledVector(probe, distance);
      if (position.y < minY || position.y > 90 || Math.hypot(position.x, position.z) > 150) continue;
      // Starting in the resident's known free space ensures a candidate on the
      // far side of a wall cannot be mistaken for a useful observation point.
      if (collision?.nearest?.(focus, probe, .045, distance + mover.radius + .09)) continue;
      let safe = true;
      for (const direction of directions) if (collision?.nearest?.(position, direction, .001, mover.radius + .035)) { safe = false; break; }
      if (safe) return { position: position.clone(), target: focus };
    }
    return null;
  }

  function begin(actor) {
    if (!readAnchor(actor)) return false;
    camera.updateMatrixWorld(); camera.getWorldDirection(viewDirection);
    const horizontal = Math.hypot(viewDirection.x, viewDirection.z);
    if (horizontal > 1e-7) {
      aimYaw = Math.atan2(-viewDirection.x, -viewDirection.z);
    } else {
      // A vertical free-camera view still has a compass heading. Recover it
      // from the camera's right axis rather than amplified direction noise.
      cameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
      if (Math.hypot(cameraRight.x, cameraRight.z) > 1e-7) {
        aimYaw = Math.atan2(-cameraRight.z, cameraRight.x);
      }
    }
    aimPitch = Math.atan2(viewDirection.y, horizontal);
    anchor.copy(rawAnchor); offset.subVectors(camera.position, anchor);
    arm = Math.max(.01, offset.length());
    yaw = wantedYaw = Math.atan2(offset.x, offset.z);
    elevation = Math.atan2(offset.y, Math.hypot(offset.x, offset.z));
    wantedElevation = clamp(elevation, minElevation, maxElevation);
    requestedDistance = clamp(arm, Math.max(2.6, minDistance), Math.min(9, maxDistance));
    clearance = Math.max(arm, requestedDistance); clearFor = uncrampedFor = 0;
    held = obstructed = lookHeld = false; active = true; steps = blockedSteps = 0;
    // Deliberately do not modify camera position, orientation or FOV here.
    return true;
  }

  function stop() { active = false; held = obstructed = lookHeld = false; }
  function look(deltaYaw, deltaElevation) {
    wantedYaw = wrap(wantedYaw + finite(deltaYaw));
    wantedElevation = clamp(wantedElevation + finite(deltaElevation), minElevation, maxElevation);
  }
  // Positive dolly moves towards the subject, matching free exploration.
  function dolly(metres) { requestedDistance = clamp(requestedDistance - finite(metres), minDistance, maxDistance); }

  function step(dt) {
    steps++;
    oldAnchor.copy(anchor);
    anchor.lerp(rawAnchor, 1 - Math.exp(-dt * 5));
    anchorTravel.subVectors(anchor, oldAnchor);
    const angleBlend = 1 - Math.exp(-dt * 8);
    yaw = wrap(yaw + wrap(wantedYaw - yaw) * angleBlend);
    elevation += (wantedElevation - elevation) * angleBlend;
    const flat = Math.cos(elevation);
    offset.set(Math.sin(yaw) * flat, Math.sin(elevation), Math.cos(yaw) * flat);

    const probeLength = Math.max(requestedDistance, arm) + mover.radius + .08;
    const hit = collision?.nearest?.(anchor, offset, .045, probeLength);
    const available = hit ? Math.max(0, hit.distance - mover.radius - .09) : maxDistance;
    const cramped = !!hit && available < minDistance;
    obstructed = !!hit && available < requestedDistance;
    if (cramped) {
      if (!held) arm = Math.max(minDistance, camera.position.distanceTo(anchor));
      held = true; uncrampedFor = 0;
    } else {
      uncrampedFor += dt;
      if (held && uncrampedFor >= .6) { held = false; arm = Math.max(minDistance, camera.position.distanceTo(anchor)); }
    }

    const candidate = Math.max(minDistance, Math.min(maxDistance, available));
    if (candidate < clearance) {
      clearance = candidate; clearFor = 0;
    } else if (candidate > clearance + .35) {
      clearFor += dt;
      // Wait for a durable opening, then restore distance gently. A single
      // unobstructed frame amongst fine railing bars does not move the camera.
      if (clearFor >= .7) clearance = Math.min(candidate, clearance + dt * .85);
    } else clearFor = 0;

    if (held) {
      // Keep the current framing offset while a resident is in an alcove/cabin.
      // Following support translation still lets a lift carry the observer;
      // blocked movement stops at its last safe point instead of switching side.
      desired.copy(camera.position).add(anchorTravel);
    } else {
      const goal = Math.min(requestedDistance, clearance);
      const recovering = goal > arm;
      const speed = recovering ? 1.2 : Math.min(16, Math.max(3.5, (arm - goal) * 1.1));
      const delta = (goal - arm) * (1 - Math.exp(-dt * (recovering ? 2 : 3.5)));
      arm += clamp(delta, -speed * dt, speed * dt);
      desired.copy(anchor).addScaledVector(offset, arm);
    }

    movement.subVectors(desired, camera.position);
    if (!held) movement.multiplyScalar(1 - Math.exp(-dt * 4));
    const length = movement.length(), speedLimit = Math.min(16, Math.max(7, arm * .45));
    if (length > speedLimit * dt) movement.multiplyScalar(speedLimit * dt / length);
    if (movement.lengthSq() > 1e-14) {
      mover.move(camera.position, movement);
      if (mover.last.collided) blockedSteps++;
    }

    viewDirection.subVectors(anchor, camera.position);
    const separation = viewDirection.length();
    if (separation < 1.05) lookHeld = true;
    else if (separation > 1.45) lookHeld = false;
    if (!lookHeld && separation > .001) {
      const horizontal = Math.hypot(viewDirection.x, viewDirection.z);
      const targetYaw = horizontal > .3 ? Math.atan2(-viewDirection.x, -viewDirection.z) : aimYaw;
      const targetPitch = clamp(Math.atan2(viewDirection.y, horizontal), -1.3, 1.3);
      const k = 1 - Math.exp(-dt * 5);
      let dy = wrap(targetYaw - aimYaw) * k, dp = (targetPitch - aimPitch) * k;
      // A person passing very close or underneath cannot whip the camera round.
      const angular = Math.hypot(dy, dp), maxStep = dt * 1.3;
      if (angular > maxStep) { dy *= maxStep / angular; dp *= maxStep / angular; }
      aimYaw = wrap(aimYaw + dy); aimPitch += dp;
    }
    camera.up.set(0, 1, 0); camera.rotation.set(aimPitch, aimYaw, 0, 'YXZ');
  }

  function update(seconds, actor) {
    if (!active) return;
    if (!readAnchor(actor)) { stop(); return; }
    const dt = clamp(finite(seconds), 0, .05); if (!dt) return;
    // Substeps bound close-range motion equally at 30, 60 and 120 Hz.
    const count = Math.ceil(dt * 120), substep = dt / count;
    for (let i = 0; i < count; i++) step(substep);
  }

  return { begin, findFollowPose, look, dolly, update, stop, mover,
    setRadius: value => mover.setRadius(value),
    get snapshot() { return { active, held, obstructed, lookHeld, requestedDistance, arm, clearance,
      worldBearing: yaw, elevation, position: camera.position.toArray(), focus: anchor.toArray(),
      orientation: [aimYaw, aimPitch], clearFor, steps, blockedSteps,
      collision: { ...mover.last } }; },
  };
}
