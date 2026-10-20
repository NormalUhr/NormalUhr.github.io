import * as THREE from 'three';
import {createEnvironment} from './environment.js';
export {prepareEnvironment} from './environment.js';

// All distances are in the habitat's world coordinates. Weather is a seeded
// local simulation, not a forecast. Nothing here recreates scene inhabitants.
const TAU = Math.PI * 2;
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const mix = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a)); return t * t * (3 - 2 * t); };
const damp = (value, target, rate, dt) => mix(value, target, 1 - Math.exp(-rate * dt));
const mod = (n, d) => ((n % d) + d) % d;
const labels = { clear: '晴朗', cloudy: '多云', rain: '细雨', storm: '阵雨', snow: '落雪', mist: '薄雾' };
const seasons = ['spring', 'summer', 'autumn', 'winter'];
const climate = {
  clear:  { cloud: .10, rain: 0, snowfall: 0, mist: .025, wind: .25 },
  cloudy: { cloud: .78, rain: 0, snowfall: 0, mist: .17, wind: .52 },
  rain:   { cloud: .87, rain: .53, snowfall: 0, mist: .40, wind: .65 },
  storm:  { cloud: 1, rain: 1, snowfall: 0, mist: .54, wind: 1 },
  snow:   { cloud: .82, rain: 0, snowfall: .83, mist: .39, wind: .38 },
  mist:   { cloud: .43, rain: 0, snowfall: 0, mist: .94, wind: .12 }
};
const seasonalWeather = [
  [['clear', .32], ['cloudy', .24], ['rain', .23], ['storm', .03], ['mist', .18]],
  [['clear', .55], ['cloudy', .18], ['rain', .13], ['storm', .08], ['mist', .06]],
  [['clear', .36], ['cloudy', .29], ['rain', .18], ['storm', .05], ['mist', .12]],
  [['clear', .25], ['cloudy', .27], ['rain', .16], ['snow', .20], ['mist', .12]]
];

function randomGenerator(seed) {
  let a = seed >>> 0;
  return () => {
    a += 0x6D2B79F5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(seed, value) {
  let h = Math.imul((seed ^ (value | 0)) >>> 0, 0x45d9f3b);
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function chooseWeather(seed, bucket, season) {
  let p = hash(seed + season * 1299721, bucket);
  for (const [kind, weight] of seasonalWeather[season]) {
    p -= weight;
    if (p < 0) return kind;
  }
  return 'clear';
}

function normalAt(heightAt, x, z, out) {
  const d = .06;
  return out.set(heightAt(x - d, z) - heightAt(x + d, z), d * 2,
    heightAt(x, z - d) - heightAt(x, z + d)).normalize();
}

const skyVertex = /* glsl */`
  varying vec3 vDirection;
  void main() {
    vDirection = position;
    vec4 p = projectionMatrix * mat4(mat3(viewMatrix)) * vec4(position, 1.0);
    gl_Position = p.xyww;
  }
`;

const skyFragment = /* glsl */`
  precision highp float;
  varying vec3 vDirection;
  uniform vec3 uSun;
  uniform vec3 uMoon;
  uniform float uDay;
  uniform float uTwilight;
  uniform float uCloud;
  uniform float uTime;
  uniform vec2 uWind;
  uniform vec3 uHorizon;
  float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.,0.)), f.x),
      mix(hash21(i + vec2(0.,1.)), hash21(i + 1.), f.x), f.y);
  }
  float fbm(vec2 p) {
    float value = 0.0;
    value += .53 * noise(p); p = mat2(1.7,1.2,-1.2,1.7) * p + 12.3;
    value += .27 * noise(p); p = mat2(1.7,1.2,-1.2,1.7) * p + 31.1;
    value += .13 * noise(p); p = mat2(1.7,1.2,-1.2,1.7) * p + 6.4;
    value += .07 * noise(p);
    return value;
  }
  void main() {
    vec3 d = normalize(vDirection);
    float elevation = max(d.y, 0.0);
    vec3 zenith = mix(vec3(.006,.014,.031), vec3(.22,.39,.53), uDay);
    vec3 sky = mix(uHorizon, zenith, pow(elevation, .42));
    float solarFacing = pow(max(dot(d, uSun), 0.0), 6.0);
    float horizonBand = exp(-abs(d.y - .03) * 5.8);
    sky += vec3(.35,.12,.026) * uTwilight * solarFacing * horizonBand;
    float sunDot = dot(d, uSun);
    float sunDisk = smoothstep(.99978, .99994, sunDot);
    float sunAura = pow(max(sunDot, 0.0), 250.0) * .18
      + pow(max(sunDot, 0.0), 35.0) * .06;
    vec3 sunColor = mix(vec3(1.0,.42,.12), vec3(1.0,.88,.68), smoothstep(0.,.55,uSun.y));
    sky += sunColor * (sunDisk * 9.0 + sunAura) * smoothstep(-.06,.02,uSun.y);
    float moonDot = dot(d,uMoon);
    float moonDisk = smoothstep(.99986,.999965,moonDot);
    // The small tonal variation keeps the moon from looking like a UI dot.
    float lunarSurface = .82 + .18 * noise(d.xz * 650.);
    sky += vec3(.42,.52,.64) * (moonDisk * lunarSurface + pow(max(moonDot,0.),170.)*.015) * (1.-uDay);
    vec2 uv = vec2(atan(d.z,d.x)/6.2831853+.5, asin(clamp(d.y,-1.,1.))/3.14159265+.5);
    vec2 starGrid = uv * vec2(680.,340.);
    vec2 cell = floor(starGrid);
    float starPick = hash21(cell);
    float star = smoothstep(.9975,.9999,starPick)
      * exp(-dot(fract(starGrid)-.5,fract(starGrid)-.5)*135.);
    sky += vec3(.55,.64,.78) * star * (1.-uDay) * smoothstep(.01,.25,d.y)
      * (.86+.14*sin(uTime*.33+starPick*500.));
    vec2 cloudUV = d.xz / max(d.y + .18,.10) * 1.32 + uWind * uTime * .0025;
    float cloudField = fbm(cloudUV);
    float coverage = smoothstep(mix(.79,.33,uCloud),mix(.91,.55,uCloud),cloudField);
    coverage *= smoothstep(-.06,.20,d.y) * smoothstep(.02,.26,uCloud);
    float edgeLight = fbm(cloudUV + vec2(.2,.11));
    vec3 cloudColor = mix(vec3(.016,.023,.036), vec3(.45,.48,.47),uDay);
    cloudColor += vec3(.35,.30,.22) * edgeLight * uDay * (1.-uCloud*.55);
    cloudColor += vec3(.25,.105,.025) * uTwilight * solarFacing;
    sky = mix(sky,cloudColor,coverage * .95);
    // The horizon fades into the same air color used by scene fog.
    sky = mix(uHorizon,sky,smoothstep(-.18,.01,d.y));
    gl_FragColor = vec4(sky,1.);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

function createRain(group, random, count, heightAt) {
  const positions = new Float32Array(count * 6);
  const strengths = new Float32Array(count * 2);
  const tails = new Float32Array(count * 2);
  const particles = Array.from({ length: count }, (_, i) => {
    const x = (random() - .5) * 40, z = (random() - .5) * 32;
    const particle = { x, y: heightAt(x, z) + random() * 18 + .2, z,
      speed: 11 + random() * 8, length: .19 + random() * .26, threshold: random() };
    strengths[i * 2] = strengths[i * 2 + 1] = particle.threshold;
    tails[i * 2] = .72; tails[i * 2 + 1] = .07;
    return particle;
  });
  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', positionAttribute);
  geometry.setAttribute('aThreshold', new THREE.BufferAttribute(strengths, 1));
  geometry.setAttribute('aTail', new THREE.BufferAttribute(tails, 1));
  const material = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, toneMapped: false,
    uniforms: { uRain: { value: 0 }, uLight: { value: 1 }, uFog: { value: .008 } },
    vertexShader: `
      attribute float aThreshold; attribute float aTail;
      uniform float uRain; uniform float uFog;
      varying float vAlpha;
      void main(){
        vec4 p=modelViewMatrix*vec4(position,1.);
        vAlpha=smoothstep(aThreshold-.10,aThreshold+.10,uRain)*aTail
          *smoothstep(0.,.07,uRain)*exp(-uFog*uFog*p.z*p.z);
        gl_Position=projectionMatrix*p;
      }`,
    fragmentShader: `
      uniform float uLight; varying float vAlpha;
      void main(){gl_FragColor=vec4(mix(vec3(.33,.49,.60),vec3(.65,.73,.75),uLight),vAlpha*.38);
        #include <colorspace_fragment>
      }`
  });
  const mesh = new THREE.LineSegments(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 5;
  group.add(mesh);
  return { mesh, geometry, material, positions, particles, positionAttribute };
}

function createSnow(group, random, count, heightAt) {
  const positions = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const thresholds = new Float32Array(count);
  const particles = Array.from({ length: count }, (_, i) => {
    const x = (random() - .5) * 40, z = (random() - .5) * 32;
    sizes[i] = .025 + random() * .035; thresholds[i] = random();
    return { x, y: heightAt(x, z) + random() * 15 + .2, z, speed: .72 + random() * .63,
      phase: random() * TAU, swing: .14 + random() * .23 };
  });
  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', positionAttribute);
  geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute('aThreshold', new THREE.BufferAttribute(thresholds, 1));
  const material = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, toneMapped: false,
    uniforms: { uSnow: { value: 0 }, uLight: { value: 1 }, uPixels: { value: 700 } },
    vertexShader: `
      attribute float aSize; attribute float aThreshold;
      uniform float uSnow; uniform float uPixels; varying float vAlpha;
      void main(){
        vec4 p=modelViewMatrix*vec4(position,1.);
        gl_PointSize=clamp(aSize*uPixels/max(1.,-p.z),1.,7.);
        vAlpha=smoothstep(aThreshold-.12,aThreshold+.12,uSnow)*smoothstep(0.,.08,uSnow);
        gl_Position=projectionMatrix*p;
      }`,
    fragmentShader: `
      uniform float uLight; varying float vAlpha;
      void main(){
        float radius=length(gl_PointCoord-.5)*2.;
        float a=(1.-smoothstep(.24,1.,radius))*vAlpha;
        if(a<.005)discard;
        gl_FragColor=vec4(mix(vec3(.32,.43,.57),vec3(.83,.87,.88),uLight),a*.75);
        #include <colorspace_fragment>
      }`
  });
  const mesh = new THREE.Points(geometry, material);
  mesh.frustumCulled = false; mesh.renderOrder = 5;
  group.add(mesh);
  return { mesh, geometry, material, positions, particles, positionAttribute };
}

function createRipples(group, count) {
  const geometry = new THREE.RingGeometry(.77, 1, 20);
  const life = new Float32Array(count).fill(1);
  geometry.setAttribute('aLife', new THREE.InstancedBufferAttribute(life, 1).setUsage(THREE.DynamicDrawUsage));
  const material = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    uniforms: { uLight: { value: 1 }, uWet: { value: 0 } },
    vertexShader: `
      attribute float aLife; varying float vAlpha;
      void main(){
        vAlpha=sin(aLife*3.14159265)*pow(1.-aLife,1.5);
        gl_Position=projectionMatrix*modelViewMatrix*instanceMatrix*vec4(position,1.);
      }`,
    fragmentShader: `
      uniform float uLight; uniform float uWet; varying float vAlpha;
      void main(){gl_FragColor=vec4(mix(vec3(.22,.35,.46),vec3(.57,.67,.68),uLight),vAlpha*(.15+uWet*.25));
        #include <colorspace_fragment>
      }`
  });
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  group.add(mesh);
  const entries = Array.from({ length: count }, () => ({ life: 1, x: 0, y: 0, z: 0, radius: .1,
    rotation: new THREE.Quaternion() }));
  const zero = new THREE.Matrix4().makeScale(0, 0, 0);
  for (let i = 0; i < count; i++) mesh.setMatrixAt(i, zero);
  return { mesh, geometry, material, life, entries, cursor: 0 };
}

function createDew(group, random, count, heightAt) {
  const geometry = new THREE.SphereGeometry(1, 8, 6);
  const material = new THREE.MeshPhysicalMaterial({ color: 0xc5dbd7, roughness: .055,
    metalness: .025, clearcoat: 1, clearcoatRoughness: .025,
    transparent: true, opacity: 0, envMapIntensity: 1.05, depthWrite: false });
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.frustumCulled = false;
  mesh.renderOrder = 2;
  const dummy = new THREE.Object3D(), up = new THREE.Vector3(0, 1, 0), normal = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    const x = (random() - .5) * 33, z = (random() - .5) * 25;
    const radius = .018 + Math.pow(random(), 1.9) * .046;
    normalAt(heightAt, x, z, normal);
    dummy.position.set(x, heightAt(x, z) + radius * .38, z);
    dummy.quaternion.setFromUnitVectors(up, normal);
    dummy.scale.set(radius, radius * .71, radius);
    dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
  }
  group.add(mesh);
  return { mesh, geometry, material };
}

function createFireflies(group, random, count, heightAt) {
  const positions = new Float32Array(count * 3), seeds = new Float32Array(count);
  const particles = Array.from({ length: count }, (_, i) => {
    const side = random() < .5 ? -1 : 1;
    const x = side * (8 + random() * 7), z = (random() - .5) * 20;
    seeds[i] = random();
    return { x, z, y: heightAt(x, z) + .18 + random() * 1.1, phase: random() * TAU, speed: .15 + random() * .18 };
  });
  const geometry = new THREE.BufferGeometry();
  const positionAttribute = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', positionAttribute);
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
  const material = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false,
    uniforms: { uNight: { value: 0 }, uTime: { value: 0 }, uPixels: { value: 700 } },
    vertexShader: `
      attribute float aSeed; uniform float uNight; uniform float uTime; uniform float uPixels;
      varying float vGlow;
      void main(){
        vec4 p=modelViewMatrix*vec4(position,1.);
        gl_PointSize=clamp((.11+aSeed*.05)*uPixels/max(1.,-p.z),2.,11.);
        float pulse=max(0.,sin(uTime*(.7+aSeed*.65)+aSeed*36.));
        vGlow=uNight*pow(pulse,5.)*(.35+aSeed*.35);
        gl_Position=projectionMatrix*p;
      }`,
    fragmentShader: `
      varying float vGlow;
      void main(){
        vec2 p=gl_PointCoord-.5; float d=dot(p,p);
        float core=exp(-d*120.); float halo=exp(-d*18.)*.15;
        gl_FragColor=vec4(vec3(.68,.87,.23),(core+halo)*vGlow);
        #include <colorspace_fragment>
      }`
  });
  const mesh = new THREE.Points(geometry, material);
  mesh.frustumCulled = false; mesh.renderOrder = 6;
  group.add(mesh);
  return { mesh, geometry, material, positions, particles, positionAttribute };
}

/**
 * A single directional shadow source, physically placed precipitation and a
 * fixed PMREM reflection field. Root owns renderer tone mapping and the clock.
 * state.wind is a Vector3; scalar weather quantities are normalized 0..1.
 * state.snow is accumulated cover, while state.snowfall is falling particles.
 */
export function createAtmosphere(scene, renderer, camera, options = {}) {
  const { heightAt = () => 0, seed = 428713, quality = 'high' } = options;
  const random = randomGenerator(seed ^ 0x724167);
  const low = quality === 'low', medium = quality === 'medium';
  const group = new THREE.Group(); group.name = '庭院的风雨与天光'; scene.add(group);
  const oldFog = scene.fog, oldEnvironment = scene.environment, oldEnvironmentIntensity = scene.environmentIntensity;
  const fog = new THREE.FogExp2(0x9eaaa0, .01); scene.fog = fog;
  const sun = new THREE.DirectionalLight(0xffebce, 3);
  sun.name = 'Continuous sun and moon shadow';
  sun.castShadow = true;
  sun.shadow.mapSize.set(low ? 1024 : 2048, low ? 1024 : 2048);
  Object.assign(sun.shadow.camera, { left: -13, right: 13, top: 13, bottom: -13, near: .5, far: 145 });
  sun.shadow.bias = -.000045; sun.shadow.normalBias = .009; sun.shadow.radius = 2;
  sun.shadow.camera.updateProjectionMatrix();
  sun.target.position.set(0, 0, 0); group.add(sun, sun.target);
  const hemisphere = new THREE.HemisphereLight(0xd4e4e7, 0x554d32, .7);
  hemisphere.name = '天空与庭院的反射光'; group.add(hemisphere);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const skyUniforms = {
    uSun: { value: new THREE.Vector3(.7, .6, -.3).normalize() },
    uMoon: { value: new THREE.Vector3(-.7, .5, .3).normalize() },
    uDay: { value: 1 }, uTwilight: { value: 0 }, uCloud: { value: .1 }, uTime: { value: 0 },
    uWind: { value: new THREE.Vector2(1, .2) }, uHorizon: { value: new THREE.Color(.45, .49, .41) }
  };
  const skyGeometry = new THREE.SphereGeometry(1, 32, 20);
  const skyMaterial = new THREE.ShaderMaterial({ uniforms: skyUniforms, vertexShader: skyVertex,
    fragmentShader: skyFragment, side: THREE.BackSide, depthWrite: false, depthTest: true });
  const sky = new THREE.Mesh(skyGeometry, skyMaterial);
  sky.name = 'Continuous procedural sky'; sky.frustumCulled = false; sky.renderOrder = -1000; group.add(sky);
  const environmentTarget = createEnvironment(renderer, options.environmentData);
  scene.environment = environmentTarget.texture;
  scene.environmentIntensity = .42;
  const rain = createRain(group, random, low ? 560 : medium ? 980 : 1600, heightAt);
  const snow = createSnow(group, random, low ? 170 : medium ? 300 : 480, heightAt);
  const ripples = createRipples(group, low ? 40 : 88);
  const dew = createDew(group, random, low ? 110 : 240, heightAt);
  const fireflies = createFireflies(group, random, low ? 16 : 32, heightAt);
  const now = new Date();
  const firstDay = new Date(now.getFullYear(), 0, 0);
  const state = {
    time: now.getHours() + now.getMinutes() / 60, dayOfYear: Math.floor((now - firstDay) / 86400000),
    elapsed: 0, wind: new THREE.Vector3(.35, 0, .08), wetness: .06, snow: 0, season: 2,
    seasonBlend: [0, 0, 1, 0], dew: 0, daylight: 1, rain: 0, snowfall: 0,
    cloud: .1, mist: .05, fogDensity: .01, temperature: 17, nightGlow: 0,
    sunElevation: .7, sunDirection: new THREE.Vector3(), moonDirection: new THREE.Vector3(),
    weatherLabel: labels.clear, weatherKey: 'clear', weatherMode: 'auto', nextWeather: 'clear',
    simulation: true
  };
  const targetSun = new THREE.Vector3(), targetMoon = new THREE.Vector3(), lightDirection = new THREE.Vector3();
  const normal = new THREE.Vector3(), zAxis = new THREE.Vector3(0, 0, 1);
  const targetLight = new THREE.Color(), targetSky = new THREE.Color(), targetBounce = new THREE.Color();
  const targetHorizon = new THREE.Color(), size = new THREE.Vector2();
  const lunarColor = new THREE.Color(.46, .61, .88), overcastColor = new THREE.Color(.53, .56, .57);
  const warmHorizon = new THREE.Color(.56, .33, .17), cloudyHorizon = new THREE.Color(.31, .35, .35);
  const dummy = new THREE.Object3D();
  let initialized = false, disposed = false;

  function splash(particle) {
    const entry = ripples.entries[ripples.cursor++ % ripples.entries.length];
    entry.x = particle.x; entry.z = particle.z;
    entry.y = heightAt(particle.x, particle.z) + .012;
    entry.life = 0; entry.radius = .085 + random() * .06 + state.wetness * .07;
    normalAt(heightAt, entry.x, entry.z, normal);
    entry.rotation.setFromUnitVectors(zAxis, normal);
  }

  function update(dt, input = {}) {
    if (disposed) return state;
    dt = clamp(Number.isFinite(dt) ? dt : 0, 0, .12);
    state.elapsed += dt;
    const alpha = initialized ? 1 - Math.exp(-dt * .8) : 1;
    const skyAlpha = initialized ? 1 - Math.exp(-dt * 2.5) : 1;
    const hours = mod(Number.isFinite(input.hours) ? input.hours : state.time, 24);
    const dayOfYear = clamp(Number.isFinite(input.dayOfYear) ? input.dayOfYear : state.dayOfYear, 1, 366);
    let season = Number.isFinite(input.season) ? Math.round(input.season) : seasons.indexOf(input.season);
    if (season < 0 || season > 3) season = mod(Math.floor((dayOfYear - 80) / 91.25), 4);
    state.time = hours; state.dayOfYear = dayOfYear; state.season = season;
    for (let i = 0; i < 4; i++) state.seasonBlend[i] = mix(state.seasonBlend[i], i === season ? 1 : 0, alpha);
    let mode = input.weather || 'auto'; if (mode !== 'auto' && !climate[mode]) mode = 'auto';
    const epoch = dayOfYear * 24 + hours;
    const bucket = Math.floor(epoch / 6), phase = mod(epoch / 6, 1);
    const currentKey = mode === 'auto' ? chooseWeather(seed, bucket, season) : mode;
    const nextKey = mode === 'auto' ? chooseWeather(seed, bucket + 1, season) : mode;
    const weatherBlend = mode === 'auto' ? smooth(.72, 1, phase) : 0;
    const a = climate[currentKey], b = climate[nextKey];
    const next = {};
    for (const field of ['cloud', 'rain', 'snowfall', 'mist', 'wind']) next[field] = mix(a[field], b[field], weatherBlend);
    state.weatherMode = mode; state.weatherKey = weatherBlend > .5 ? nextKey : currentKey;
    state.weatherLabel = labels[state.weatherKey]; state.nextWeather = nextKey;
    for (const field of ['cloud', 'rain', 'snowfall', 'mist']) state[field] = mix(state[field], next[field], alpha);

    // Solar declination and hour angle for a temperate northern latitude. The
    // root clock passes a continuous hour value, including smooth midnight wrap.
    const latitude = .647, declination = -.4091 * Math.cos(TAU * (dayOfYear + 10) / 365);
    const hourAngle = (hours - 12) * Math.PI / 12;
    const sinElevation = Math.sin(latitude) * Math.sin(declination)
      + Math.cos(latitude) * Math.cos(declination) * Math.cos(hourAngle);
    targetSun.set(-Math.sin(hourAngle) * Math.cos(declination), sinElevation,
      Math.cos(latitude) * Math.sin(declination) - Math.sin(latitude) * Math.cos(declination) * Math.cos(hourAngle)).normalize();
    targetMoon.set(-targetSun.x * .91 + .18, -targetSun.y * .87 + .10, -targetSun.z * .91 + .21).normalize();
    skyUniforms.uSun.value.lerp(targetSun, skyAlpha).normalize();
    skyUniforms.uMoon.value.lerp(targetMoon, skyAlpha).normalize();
    state.sunDirection.copy(skyUniforms.uSun.value); state.moonDirection.copy(skyUniforms.uMoon.value);
    state.sunElevation = Math.asin(clamp(state.sunDirection.y, -1, 1));
    const daylight = smooth(-.14, .15, state.sunDirection.y);
    const twilight = Math.exp(-Math.pow((state.sunDirection.y - .025) / .19, 2)) * (1 - state.cloud * .7);
    state.daylight = mix(state.daylight, daylight, skyAlpha);
    const windAngle = .47 + .20 * Math.sin(state.elapsed * .027 + seed * .001)
      + .11 * Math.sin(state.elapsed * .071);
    const gust = 1 + .18 * Math.sin(state.elapsed * .54) + .08 * Math.sin(state.elapsed * 1.21 + 2);
    const windSpeed = next.wind * gust * (1.4 + state.rain * 2.7);
    const windAlpha = initialized ? 1 - Math.exp(-dt * .4) : 1;
    state.wind.x = mix(state.wind.x, Math.cos(windAngle) * windSpeed, windAlpha);
    state.wind.z = mix(state.wind.z, Math.sin(windAngle) * windSpeed, windAlpha);
    const seasonalTemperature = state.seasonBlend.reduce((sum, value, i) => sum + value * [14, 24, 15, 5][i], 0);
    let targetTemperature = seasonalTemperature + Math.sin((hours - 9) / 24 * TAU) * 5.5 - state.rain * 3.5 - state.cloud * 1.6;
    if (mode === 'snow') targetTemperature = Math.min(targetTemperature, -1.4);
    state.temperature = mix(state.temperature, targetTemperature, alpha * .5 + (initialized ? 0 : .5));
    const dawnWindow = hours < 12 ? 1 - smooth(7.0, 11.3, hours) : smooth(20, 24, hours) * .5;
    const dewTarget = dawnWindow * (1 - state.cloud * .58) * (1 - state.rain)
      * (1 - state.seasonBlend[3] * .62) * .85;
    if (!initialized) { state.dew = dewTarget; state.wetness = .025 + dewTarget * .16; }
    const climateDt = dt * Math.min(6, Math.pow(Math.max(1, input.timeScale || 1), .25));
    state.dew = damp(state.dew, dewTarget, state.daylight > .8 ? .09 : .065, climateDt);
    const evaporate = .0015 + state.daylight * .0045 + Math.max(0, state.temperature) * .00019;
    state.wetness = clamp(state.wetness + state.rain * .065 * (1 - state.wetness) * climateDt
      - evaporate * state.wetness * climateDt);
    const melting = Math.max(0, state.temperature) * .00042 + state.rain * .005;
    state.snow = clamp(state.snow + state.snowfall * .013 * (1 - state.snow) * climateDt
      - melting * state.snow * climateDt);
    state.nightGlow = (1 - smooth(.12, .62, state.daylight))
      * (state.seasonBlend[0] * .5 + state.seasonBlend[1]) * (1 - state.rain * .9);

    const solarStrength = Math.pow(Math.max(0, state.sunDirection.y), .55);
    const dayShadow = smooth(-.045, .10, state.sunDirection.y);
    const cloudTransmission = 1 - state.cloud * .77;
    const daylightPower = solarStrength * 3.65 * cloudTransmission;
    // A readable, dark-adapted night exposure: the camera's exposure stays
    // fixed, so a modest blue sky fill preserves garden details under the trees.
    const moonPower = .58 * (1 - state.daylight) * (1 - state.cloud * .55);
    sun.intensity = mix(sun.intensity, daylightPower + moonPower, skyAlpha);
    const sunWarmth = smooth(.02, .55, state.sunDirection.y);
    targetLight.setRGB(mix(1, 1, sunWarmth), mix(.40, .91, sunWarmth), mix(.16, .73, sunWarmth));
    targetLight.lerp(lunarColor, 1 - dayShadow);
    sun.color.lerp(targetLight, skyAlpha);
    lightDirection.copy(state.moonDirection).lerp(state.sunDirection, dayShadow);
    lightDirection.y = Math.max(.085, lightDirection.y); lightDirection.normalize();
    sun.position.copy(lightDirection).multiplyScalar(65);
    targetSky.setRGB(mix(.32, .59, state.daylight), mix(.43, .72, state.daylight), mix(.66, .84, state.daylight));
    targetSky.lerp(overcastColor, state.cloud * state.daylight * .6);
    targetBounce.setRGB(mix(.10, .22, state.daylight), mix(.13, .18, state.daylight), mix(.18, .095, state.daylight));
    hemisphere.color.lerp(targetSky, skyAlpha); hemisphere.groundColor.lerp(targetBounce, skyAlpha);
    hemisphere.intensity = mix(hemisphere.intensity, .70 + state.daylight * .88 + state.cloud * state.daylight * .12, skyAlpha);
    scene.environmentIntensity = mix(scene.environmentIntensity, .25 + state.daylight * .55 - state.cloud * .055, skyAlpha);
    targetHorizon.setRGB(mix(.012, .41, state.daylight), mix(.023, .46, state.daylight), mix(.040, .43, state.daylight));
    targetHorizon.lerp(warmHorizon, twilight * .38);
    targetHorizon.lerp(cloudyHorizon, state.cloud * state.daylight * .50);
    skyUniforms.uHorizon.value.lerp(targetHorizon, skyAlpha);
    skyUniforms.uDay.value = state.daylight; skyUniforms.uTwilight.value = twilight;
    skyUniforms.uCloud.value = state.cloud; skyUniforms.uTime.value = state.elapsed;
    skyUniforms.uWind.value.set(state.wind.x, state.wind.z);
    const morningMist = dawnWindow * .0035 * (1 - state.rain);
    const targetDensity = .0055 + state.mist * .018 + morningMist + state.snowfall * .002;
    state.fogDensity = mix(state.fogDensity, targetDensity, alpha);
    fog.density = state.fogDensity; fog.color.copy(skyUniforms.uHorizon.value);

    renderer.getDrawingBufferSize(size);
    const pixels = size.y * .68;
    rain.material.uniforms.uRain.value = state.rain;
    rain.material.uniforms.uLight.value = state.daylight;
    rain.material.uniforms.uFog.value = fog.density;
    rain.mesh.visible = state.rain > .001;
    if (rain.mesh.visible || !initialized) {
      for (let i = 0; i < rain.particles.length; i++) {
        const p = rain.particles[i];
        p.x += state.wind.x * dt * .72; p.z += state.wind.z * dt * .72;
        p.y -= p.speed * dt;
        const floor = heightAt(p.x, p.z);
        if (p.y < floor || Math.abs(p.x) > 21 || Math.abs(p.z) > 17) {
          if (p.y < floor && p.threshold < state.rain && random() < .085) splash(p);
          p.x = (random() - .5) * 40; p.z = (random() - .5) * 32;
          p.y = heightAt(p.x, p.z) + 15 + random() * 4;
        }
        const j = i * 6, length = p.length * (1 + state.rain * .2);
        rain.positions[j] = p.x; rain.positions[j + 1] = p.y; rain.positions[j + 2] = p.z;
        rain.positions[j + 3] = p.x - state.wind.x * .72 * length / p.speed;
        rain.positions[j + 4] = p.y + length;
        rain.positions[j + 5] = p.z - state.wind.z * .72 * length / p.speed;
      }
      rain.positionAttribute.needsUpdate = true;
    }
    snow.material.uniforms.uSnow.value = state.snowfall;
    snow.material.uniforms.uLight.value = state.daylight; snow.material.uniforms.uPixels.value = pixels;
    snow.mesh.visible = state.snowfall > .001;
    if (snow.mesh.visible || !initialized) {
      for (let i = 0; i < snow.particles.length; i++) {
        const p = snow.particles[i];
        p.x += (state.wind.x * .4 + Math.sin(state.elapsed * .8 + p.phase) * p.swing) * dt;
        p.z += (state.wind.z * .4 + Math.cos(state.elapsed * .53 + p.phase) * p.swing) * dt;
        p.y -= p.speed * dt;
        if (p.y < heightAt(p.x, p.z) || Math.abs(p.x) > 21 || Math.abs(p.z) > 17) {
          p.x = (random() - .5) * 40; p.z = (random() - .5) * 32;
          p.y = heightAt(p.x, p.z) + 14 + random() * 3;
        }
        const j = i * 3; snow.positions[j] = p.x; snow.positions[j + 1] = p.y; snow.positions[j + 2] = p.z;
      }
      snow.positionAttribute.needsUpdate = true;
    }
    let anyRipple = false;
    for (let i = 0; i < ripples.entries.length; i++) {
      const p = ripples.entries[i];
      if (p.life >= 1) continue;
      p.life = Math.min(1, p.life + dt / .43); ripples.life[i] = p.life;
      dummy.position.set(p.x, p.y, p.z); dummy.quaternion.copy(p.rotation);
      const radius = p.life >= 1 ? 0 : p.radius * (.14 + .86 * p.life);
      dummy.scale.setScalar(radius); dummy.updateMatrix(); ripples.mesh.setMatrixAt(i, dummy.matrix);
      anyRipple = true;
    }
    if (anyRipple) { ripples.mesh.instanceMatrix.needsUpdate = true; ripples.geometry.attributes.aLife.needsUpdate = true; }
    ripples.material.uniforms.uLight.value = state.daylight; ripples.material.uniforms.uWet.value = state.wetness;
    dew.material.opacity = state.dew * (.09 + .66 * smooth(.03, .65, state.daylight)) * (1 - state.snow * .8);
    dew.mesh.visible = dew.material.opacity > .002;
    fireflies.material.uniforms.uNight.value = state.nightGlow;
    fireflies.material.uniforms.uTime.value = state.elapsed; fireflies.material.uniforms.uPixels.value = pixels;
    fireflies.mesh.visible = state.nightGlow > .002;
    if (fireflies.mesh.visible || !initialized) {
      for (let i = 0; i < fireflies.particles.length; i++) {
        const p = fireflies.particles[i], t = state.elapsed * p.speed, j = i * 3;
        fireflies.positions[j] = p.x + Math.sin(t + p.phase) * .42 + Math.sin(t * 2.3) * .07;
        fireflies.positions[j + 1] = p.y + Math.sin(t * 1.8 + p.phase) * .16;
        fireflies.positions[j + 2] = p.z + Math.cos(t * .7 + p.phase) * .38;
      }
      fireflies.positionAttribute.needsUpdate = true;
    }
    initialized = true;
    return state;
  }

  function dispose() {
    if (disposed) return;
    disposed = true; scene.remove(group);
    group.traverse(object => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) material.dispose();
      }
    });
    sun.shadow.dispose(); environmentTarget.dispose();
    if (scene.environment === environmentTarget.texture) {
      scene.environment = oldEnvironment; scene.environmentIntensity = oldEnvironmentIntensity;
    }
    if (scene.fog === fog) scene.fog = oldFog;
  }

  // First-frame lighting matches local time before the first render. Root's
  // first update is still applied immediately, so saved preview time is safe.
  update(0, { hours: state.time, dayOfYear: state.dayOfYear, weather: 'auto' });
  initialized = false;
  return { update, state, dispose, group, sun, hemisphere, sky, fog };
}
