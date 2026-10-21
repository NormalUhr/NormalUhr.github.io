import { TAU } from './mechanism.js';

const clamp = (n, low = 0, high = 1) => Math.min(high, Math.max(low, n));
const finite = (n, fallback = 0) => Number.isFinite(n) ? n : fallback;
function randomSequence(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let t = Math.imul(value ^ value >>> 15, 1 | value);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/** Exact plan of the existing bridge deck, including its moving dock plate. */
export function bridgeDeckHeightAt(bridge, x, z) {
  const e = bridge.group.matrixWorld.elements, dx = x - e[12], dz = z - e[14];
  const lx = e[0] * dx + e[2] * dz, r = -(e[8] * dx + e[10] * dz);
  const half = bridge.config.width / 2, end = bridge.config.radius - .9;
  if (Math.abs(lx) <= half && r >= Math.sqrt(5.4 * 5.4 - lx * lx) && r <= end)
    return e[13] + bridge.profile.height(r) + .025;
  if (Math.abs(lx) <= half - .12 && r >= end && r <= end + bridge.outerPlate.scale.z)
    return e[13] + 6.005;
  return null;
}

/**
 * Stable world-space rain/snow inside the formerly empty atrium. Roof queries
 * are evaluated after bridge and canopy motion, independently of cutaway view.
 * A drop dies at its first roof contact and cannot reappear below a roof when
 * that roof retracts. Only the next deterministic fall cycle spawns it again.
 * The old exterior weather remains owned by environment.js, with its seed and
 * draw budgets unchanged. These two additional batches need no new shader.
 */
export function createAtriumWeather(THREE, scene, {
  architecture, coreRoof, canopies = [], bridges = [], lowPower = false, initialTime = 0,
} = {}) {
  if (!architecture?.floors?.[3] || !coreRoof?.roofHeightAt)
    throw new TypeError('Atrium weather needs the actual gallery and central roof');
  const group = new THREE.Group(); group.name = 'Sheltered atrium weather'; scene.add(group);
  const maxParticles = lowPower ? 220 : 440, radius = 26.65, top = 58, bottom = -2.30, span = top - bottom;
  const random = randomSequence(908173), seeds = [];
  for (let i = 0; i < maxParticles; i++) {
    const a = random() * TAU, r = Math.sqrt(random()) * radius;
    seeds.push({ a, r, phase: random() * span, snowPhase: random() * span,
      speed: 14 + random() * 8, snowSpeed: .7 + random() * .8, sway: random() * TAU,
      rainHitCycle: -Infinity, snowHitCycle: -Infinity });
  }
  const rainGeometry = new THREE.BufferGeometry();
  rainGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxParticles * 6), 3));
  rainGeometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
  const snowGeometry = new THREE.BufferGeometry();
  snowGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxParticles * 3), 3));
  snowGeometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
  const rainMaterial = new THREE.LineBasicMaterial({ color: '#c1d0d7', transparent: true, opacity: 0, depthWrite: false });
  const snowMaterial = new THREE.PointsMaterial({ color: '#fff8ec', size: .11, transparent: true, opacity: 0, depthWrite: false });
  const rain = new THREE.LineSegments(rainGeometry, rainMaterial); rain.name = 'Rain above atrium roofs';
  const snow = new THREE.Points(snowGeometry, snowMaterial); snow.name = 'Snow above atrium roofs';
  rain.frustumCulled = snow.frustumCulled = false;
  rain.visible = snow.visible = false; group.add(rain, snow);
  rainGeometry.setDrawRange(0, 0); snowGeometry.setDrawRange(0, 0);
  const activeRainIds = new Uint16Array(maxParticles), activeSnowIds = new Uint16Array(maxParticles);
  let elapsed = Math.max(0, finite(initialTime)), quality = 1, disposed = false;
  const state = { rain: 0, snow: 0, visibleRain: 0, visibleSnow: 0, stoppedRain: 0, stoppedSnow: 0, elapsed };
  const upperGallery = architecture.floors[3], segment = TAU / 180;

  function shelterHeightAt(x, z) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return null;
    // Actual L3 annulus has 180 sides. Lower galleries are covered by this
    // slab; the higher galleries start outside this particle field at R27.
    const r = Math.hypot(x, z), angle = ((Math.atan2(x, -z) % TAU) + TAU) % TAU;
    const d = angle - (Math.floor(angle / segment) + .5) * segment;
    const boundary = upperGallery.innerRadius * Math.cos(segment / 2) / Math.cos(d);
    let highest = -1.665; // Three centimetres above the visible atrium court.
    if (r >= boundary && r <= radius + .001)
      highest = upperGallery.y + (r <= upperGallery.innerRadius + .27 ? .060 : .025);
    const core = coreRoof.roofHeightAt(x, z);
    if (Number.isFinite(core)) highest = Math.max(highest, core);
    for (const canopy of canopies) {
      const height = canopy.roofHeightAt(x, z);
      if (Number.isFinite(height)) highest = Math.max(highest, height);
    }
    for (const bridge of bridges) {
      const height = bridgeDeckHeightAt(bridge, x, z);
      if (Number.isFinite(height)) highest = Math.max(highest, height);
    }
    return highest;
  }

  function update(dt, environment = {}) {
    if (disposed) return state;
    elapsed += Math.max(0, finite(dt)); state.elapsed = elapsed;
    state.rain = clamp(finite(environment.rain)); state.snow = clamp(finite(environment.snow));
    const day = clamp(finite(environment.day, 1 - finite(environment.night)));
    rainMaterial.opacity = state.rain * (.23 + day * .11); snowMaterial.opacity = state.snow * .80;
    rain.visible = state.rain > .004; snow.visible = state.snow > .004;
    const n = Math.floor(maxParticles * quality), rp = rainGeometry.attributes.position.array, sp = snowGeometry.attributes.position.array;
    let nr = 0, ns = 0, stoppedRain = 0, stoppedSnow = 0;
    // Quality only changes drawing. All seeded drops still collide, including
    // those currently omitted from the low-quality draw range.
    for (let i = 0; i < maxParticles; i++) {
      const p = seeds[i];
      if (rain.visible) {
        const fall = p.phase + elapsed * p.speed, cycle = Math.floor(fall / span);
        const y = top - (fall - cycle * span), x = Math.sin(p.a) * p.r, z = -Math.cos(p.a) * p.r;
        const hit = shelterHeightAt(x, z) + .035;
        if (y <= hit) p.rainHitCycle = cycle;
        if (p.rainHitCycle === cycle) { if (i < n) stoppedRain++; }
        else if (i < n) {
          const off = nr * 6;
          rp[off] = x; rp[off + 1] = y; rp[off + 2] = z;
          rp[off + 3] = x; rp[off + 4] = Math.min(top + .85, y + .85); rp[off + 5] = z;
          activeRainIds[nr++] = i;
        }
      }
      if (snow.visible) {
        const fall = p.snowPhase + elapsed * p.snowSpeed, cycle = Math.floor(fall / span);
        const y = top - (fall - cycle * span), a = p.a + Math.sin(elapsed * .18 + p.sway) * .008;
        const x = Math.sin(a) * p.r, z = -Math.cos(a) * p.r;
        if (y <= shelterHeightAt(x, z) + .09) p.snowHitCycle = cycle;
        if (p.snowHitCycle === cycle) { if (i < n) stoppedSnow++; }
        else if (i < n) { sp[ns * 3] = x; sp[ns * 3 + 1] = y; sp[ns * 3 + 2] = z; activeSnowIds[ns++] = i; }
      }
    }
    rainGeometry.setDrawRange(0, nr * 2); snowGeometry.setDrawRange(0, ns);
    if (nr) rainGeometry.attributes.position.needsUpdate = true;
    if (ns) snowGeometry.attributes.position.needsUpdate = true;
    Object.assign(state, { visibleRain: nr, visibleSnow: ns, stoppedRain, stoppedSnow });
    return state;
  }
  function setQuality(value) { quality = clamp(finite(value, quality), .3, 1); }
  function dispose() {
    if (disposed) return;
    disposed = true; group.removeFromParent();
    rainGeometry.dispose(); snowGeometry.dispose(); rainMaterial.dispose(); snowMaterial.dispose();
  }
  return { group, rain, snow, rainGeometry, snowGeometry, materials: [rainMaterial, snowMaterial],
    activeRainIds, activeSnowIds, state, update, setQuality, shelterHeightAt, dispose,
    metrics: { maxParticles, drawCalls: 2, textures: 0, lights: 0, radius, top, bottom,
      stableSeed: 908173, occlusion: 'Current core roof, actual deployed bridge canopies, bridge decks and L3 gallery slab' } };
}
