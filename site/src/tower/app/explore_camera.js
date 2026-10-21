/**
 * Free exploration is translation plus yaw/pitch, never an orbit around a target.
 * This module has no DOM/input listeners so keyboard, touch and follow cameras
 * can share the same movement envelope without fighting over a camera pose.
 */
const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

/**
 * A conservative, finite-volume swept camera with sliding contacts. The centre
 * and twelve samples of its cross-section query real, currently visible meshes.
 * Six short axial probes preserve clearance while sliding at grazing angles.
 * Triangle plane normals supply the full radius padding even between samples.
 * This is a camera envelope, not a character/rigid-body physics simulation.
 * Existing overlap (for example a moving lift) never triggers a teleport/pushout.
 */
export function createCameraMover(THREE, {
  collision, radius = .32, minY = .65, maxY = 90, maxRadius = 150,
} = {}) {
  radius = clamp(finite(radius, .32), .16, 1.25);
  minY = finite(minY, .65); maxY = Math.max(minY, finite(maxY, 90));
  maxRadius = Math.max(1, finite(maxRadius, 150));
  const skin = .008, epsilon = 1e-7;
  const start = new THREE.Vector3(), remaining = new THREE.Vector3();
  const direction = new THREE.Vector3(), side = new THREE.Vector3(), up = new THREE.Vector3();
  const origin = new THREE.Vector3(), offset = new THREE.Vector3(), normal = new THREE.Vector3();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const edge = new THREE.Vector3(), world = new THREE.Matrix4(), instance = new THREE.Matrix4();
  const bestNormal = new THREE.Vector3(), worldUp = new THREE.Vector3(0, 1, 0);
  const axes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    .map(v => new THREE.Vector3(...v));
  const contacts = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  const probes = [[0, 0]];
  for (let n = 0; n < 8; n++) {
    const theta = n * Math.PI / 4;
    probes.push([Math.cos(theta), Math.sin(theta)]);
  }
  for (let n = 0; n < 4; n++) {
    const theta = n * Math.PI / 2 + Math.PI / 4;
    probes.push([Math.cos(theta) * .48, Math.sin(theta) * .48]);
  }
  let contactCount = 0;
  const last = { collided: false, clamped: false, queries: 0, requestedDistance: 0, movedDistance: 0 };

  function hitNormal(hit, motion) {
    // The accelerated collision API returns native geometry face indices. Use
    // actual transformed triangle vertices, including live instance matrices.
    const object = hit.object, geometry = object?.geometry, position = geometry?.attributes.position;
    const face = hit.faceIndex;
    if (position && Number.isInteger(face) && face >= 0) {
      const index = geometry.index, i = face * 3;
      if (i + 2 < (index?.count ?? position.count)) {
        a.fromBufferAttribute(position, index ? index.getX(i) : i);
        b.fromBufferAttribute(position, index ? index.getX(i + 1) : i + 1);
        c.fromBufferAttribute(position, index ? index.getX(i + 2) : i + 2);
        world.copy(object.matrixWorld);
        if (object.isInstancedMesh && Number.isInteger(hit.instanceId)) {
          object.getMatrixAt(hit.instanceId, instance); world.multiply(instance);
        }
        a.applyMatrix4(world); b.applyMatrix4(world); c.applyMatrix4(world);
        normal.subVectors(b, a); edge.subVectors(c, a); normal.cross(edge);
        if (normal.lengthSq() > 1e-15) {
          normal.normalize(); if (normal.dot(motion) > 0) normal.negate();
          return normal;
        }
      }
    }
    // A simple callback-only fixture may provide an explicit world-space normal.
    if (hit.normal?.isVector3) {
      normal.copy(hit.normal).normalize(); if (normal.dot(motion) > 0) normal.negate();
      return normal;
    }
    return normal.copy(motion).negate();
  }

  function sweep(position, delta) {
    const distance = delta.length();
    direction.copy(delta).multiplyScalar(1 / distance);
    if (Math.abs(direction.y) < .9) side.crossVectors(direction, worldUp).normalize();
    else side.set(1, 0, 0).addScaledVector(direction, -direction.x).normalize();
    up.crossVectors(side, direction).normalize();
    let travel = distance, blocked = false;
    if (!collision?.nearest) return travel;
    // A near-parallel movement ray may not reach a wall at all during a short
    // frame, even though the camera envelope already touches it. Short radial
    // probes retain that contact without caching normals from moving geometry.
    for (const axis of axes) {
      const hit = collision.nearest(position, axis, .0001, (radius + skin) * Math.sqrt(3) + Math.min(distance, .25));
      last.queries++;
      if (!hit) continue;
      const n = hitNormal(hit, axis), closing = -n.dot(direction);
      if (closing < 1e-8) continue;
      const advance = (hit.distance * -n.dot(axis) - radius - skin) / closing;
      if (advance < travel - epsilon) {
        travel = Math.max(0, advance); bestNormal.copy(n); blocked = true;
      }
    }
    for (const [sx, sy] of probes) {
      offset.copy(side).multiplyScalar(sx * radius).addScaledVector(up, sy * radius);
      origin.copy(position).add(offset);
      // At a shallow approach the parallel probe may reach the triangle farther
      // along the ray than the sphere would. Keep a bounded look-ahead for that
      // contact, while the plane calculation still clips to this frame's motion.
      const hit = collision.nearest(origin, direction, .0001, distance + radius * 5 + skin);
      last.queries++;
      if (!hit) continue;
      const n = hitNormal(hit, direction), closing = -n.dot(direction);
      if (closing < .025) continue; // Grazing triangle edges should not snag a camera.
      const advance = hit.distance - (radius + skin + n.dot(offset)) / closing;
      if (advance < travel - epsilon) {
        travel = Math.max(0, advance); bestNormal.copy(n); blocked = true;
      }
    }
    if (blocked) {
      last.collided = true;
      contacts[contactCount++].copy(bestNormal);
    }
    return travel;
  }

  function bound(position) {
    const y = clamp(position.y, minY, maxY), length = Math.hypot(position.x, position.z);
    if (y !== position.y) { position.y = y; last.clamped = true; }
    if (length > maxRadius) {
      position.x *= maxRadius / length; position.z *= maxRadius / length; last.clamped = true;
    }
    return position;
  }

  function move(position, delta, out = position) {
    Object.assign(last, { collided: false, clamped: false, queries: 0, requestedDistance: 0, movedDistance: 0 });
    contactCount = 0;
    if (![position.x, position.y, position.z, delta.x, delta.y, delta.z].every(Number.isFinite)) return out.copy(position);
    start.copy(position); remaining.copy(delta); out.copy(position);
    last.requestedDistance = remaining.length();
    // No corrections at rest: stopping near a wall stays pixel-stable, even when
    // the geometry around the camera subsequently moves.
    if (last.requestedDistance < epsilon) return out;
    for (let iteration = 0; iteration < 3; iteration++) {
      const distance = remaining.length(); if (distance < epsilon) break;
      const before = contactCount, travel = sweep(out, remaining);
      out.addScaledVector(direction, travel);
      if (contactCount === before) break;
      remaining.addScaledVector(direction, -travel);
      // Respect all accumulated planes, so a corner does not alternate between
      // two wall normals and vibrate on subsequent iterations.
      for (let n = 0; n < contactCount; n++) {
        const into = remaining.dot(contacts[n]);
        if (into < 0) remaining.addScaledVector(contacts[n], -into);
      }
    }
    bound(out); last.movedDistance = out.distanceTo(start); return out;
  }
  function constrainVelocity(velocity) {
    for (let n = 0; n < contactCount; n++) {
      const into = velocity.dot(contacts[n]);
      if (into < 0) velocity.addScaledVector(contacts[n], -into);
    }
    return velocity;
  }
  return { move, constrainVelocity, bound, last,
    setRadius(value) { radius = clamp(finite(value, radius), .16, 1.25); },
    get radius() { return radius; },
  };
}

export function createExploreCamera(THREE, camera, options = {}) {
  const mover = createCameraMover(THREE, options);
  const speed = Math.max(.25, finite(options.speed, 3.6));
  const fastSpeed = Math.max(speed, finite(options.fastSpeed, 9));
  const dollySpeed = Math.max(.25, finite(options.dollySpeed, 6));
  // A top-view preset is an exact downward view, then remains freely movable.
  // Inclusive poles avoid tilting that preset before the user touches a control.
  const maxPitch = Math.PI / 2;
  const velocity = new THREE.Vector3(), desired = new THREE.Vector3(), displacement = new THREE.Vector3();
  const wheelStep = new THREE.Vector3(), direction = new THREE.Vector3(), target = new THREE.Vector3();
  const cameraRight = new THREE.Vector3();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  const input = { right: 0, forward: 0, up: 0, fast: false };
  let yaw = 0, pitch = 0, wheelRemaining = 0;

  function orient() {
    euler.set(pitch, yaw, 0, 'YXZ'); camera.quaternion.setFromEuler(euler); camera.up.set(0, 1, 0);
  }
  function stop() { velocity.set(0, 0, 0); wheelRemaining = 0; setMove({}); }
  function adopt() {
    camera.updateMatrixWorld(); camera.getWorldDirection(direction);
    const horizontal = Math.hypot(direction.x, direction.z);
    if (horizontal > 1e-7) yaw = Math.atan2(-direction.x, -direction.z);
    else {
      // Looking straight up/down has no horizontal forward direction. The
      // camera's right axis still carries its heading, including north-up.
      cameraRight.set(1, 0, 0).applyQuaternion(camera.quaternion);
      if (Math.hypot(cameraRight.x, cameraRight.z) > 1e-7) yaw = Math.atan2(-cameraRight.z, cameraRight.x);
    }
    pitch = clamp(Math.atan2(direction.y, horizontal), -maxPitch, maxPitch);
    mover.bound(camera.position); stop(); orient(); return api;
  }
  function setPose(pose = {}) {
    if (pose.position) {
      const p = pose.position;
      camera.position.set(finite(p.x ?? p[0], camera.position.x), finite(p.y ?? p[1], camera.position.y), finite(p.z ?? p[2], camera.position.z));
    }
    if (pose.target) {
      const p = pose.target;
      target.set(finite(p.x ?? p[0]), finite(p.y ?? p[1]), finite(p.z ?? p[2]));
      direction.subVectors(target, camera.position);
      if (direction.lengthSq() > 1e-12) {
        direction.normalize(); const horizontal = Math.hypot(direction.x, direction.z);
        // A target directly above/below does not define a new compass heading.
        if (horizontal > 1e-7) yaw = Math.atan2(-direction.x, -direction.z);
        pitch = Math.atan2(direction.y, horizontal);
      }
    }
    yaw = finite(pose.yaw, yaw); pitch = clamp(finite(pose.pitch, pitch), -maxPitch, maxPitch);
    mover.bound(camera.position); stop(); orient(); return api;
  }
  function look(deltaYaw, deltaPitch) {
    yaw += finite(deltaYaw); yaw = Math.atan2(Math.sin(yaw), Math.cos(yaw));
    pitch = clamp(pitch + finite(deltaPitch), -maxPitch, maxPitch); orient();
  }
  function setMove(next = {}) {
    input.right = clamp(finite(next.right), -1, 1); input.forward = clamp(finite(next.forward), -1, 1);
    input.up = clamp(finite(next.up), -1, 1); input.fast = Boolean(next.fast);
  }
  function dolly(meters) { wheelRemaining = clamp(wheelRemaining + finite(meters), -5, 5); }
  function update(seconds) {
    // Hidden tabs / debugger pauses never produce a single giant movement step.
    const dt = clamp(finite(seconds), 0, .05); if (!dt) return;
    const sin = Math.sin(yaw), cos = Math.cos(yaw);
    desired.set(cos * input.right - sin * input.forward, input.up, -sin * input.right - cos * input.forward);
    const moving = desired.lengthSq() > 1e-10;
    if (moving) desired.clampLength(0, 1).multiplyScalar(input.fast ? fastSpeed : speed);
    const rate = moving ? 9 : 20, decay = Math.exp(-rate * dt);
    // Analytic velocity integration keeps equal travel at 30 / 60 / 120 Hz.
    displacement.copy(desired).multiplyScalar(dt).addScaledVector(velocity, (1 - decay) / rate).addScaledVector(desired, -(1 - decay) / rate);
    velocity.lerp(desired, 1 - decay);
    if (!moving && velocity.lengthSq() < 1e-7) velocity.set(0, 0, 0);
    if (Math.abs(wheelRemaining) > .00001) {
      const amount = clamp(wheelRemaining * (1 - Math.exp(-dt * 18)), -dollySpeed * dt, dollySpeed * dt);
      wheelRemaining -= amount;
      camera.getWorldDirection(wheelStep); displacement.addScaledVector(wheelStep, amount);
    } else wheelRemaining = 0;
    mover.move(camera.position, displacement); mover.constrainVelocity(velocity);
    if (mover.last.collided && mover.last.movedDistance < .00001) wheelRemaining = 0;
    orient();
  }
  const api = { adopt, setPose, look, setMove, dolly, update, stop, mover,
    setRadius: value => mover.setRadius(value),
    get snapshot() { return { position: camera.position.toArray(), yaw, pitch, velocity: velocity.toArray(),
      input: { ...input }, wheelRemaining, radius: mover.radius, collision: { ...mover.last } }; },
  };
  adopt(); return api;
}
