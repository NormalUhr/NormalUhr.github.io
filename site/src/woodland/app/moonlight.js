import * as THREE from 'three';

// At night the canopy lets only part of the moonlight through, and up to all of
// it through one opening above the clock and a smaller one above the approach to
// the nest. The openings are projected along the moon's direction onto every lit
// surface, so the terrain, the stones, the ants and the leaves agree.
export const MOON_CLEARING = Object.freeze({
  version: 'moon-clearing-v1',
  outerTransmission: .4,
  overcastPeak: .62,
  clockCenter: Object.freeze([0, -1.45]),
  clockHalfSize: Object.freeze([8.5, 2.3]),
  clockCornerRadius: .8,
  clockFeather: 4.8,
  nestCenter: Object.freeze([8.8, 3.45]),
  nestRadii: Object.freeze([2.3, 2.8]),
  nestStrength: .9,
  nestInner: .72,
  nestOuter: 1.85,
  shaftLength: 19,
  shaftRadius: 1.25,
  shaftOpacity: .02
});

const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const finite = (v, fallback) => Number.isFinite(v) ? v : fallback;
const smooth = (a, b, v) => {
  const t = clamp((v - a) / (b - a));
  return t * t * (3 - 2 * t);
};

const VERTEX_DECLARATIONS = 'varying vec3 vMoonClearingPosition;';
const FRAGMENT_DECLARATIONS = `
  varying vec3 vMoonClearingPosition;
  // x = night blend, y = outer transmission, z = opening transmission,
  // w = strength of the smaller opening above the nest approach.
  uniform vec4 uMoonProfile;
  // xy = horizontal distance per vertical distance toward the moon;
  // z = the actual terrain height under the centre of the clock.
  uniform vec3 uMoonProjection;
  uniform vec4 uMoonClock;
  uniform vec2 uMoonSoftness;
  uniform vec4 uMoonNest;
  uniform vec2 uMoonNestFeather;
  float moonClearingTransmission(vec3 worldPosition) {
    vec2 p = worldPosition.xz
      - (worldPosition.y - uMoonProjection.z) * uMoonProjection.xy;
    vec2 q = abs(p - uMoonClock.xy) - uMoonClock.zw;
    float clockDistance = length(max(q, vec2(0.)))
      + min(max(q.x, q.y), 0.) - uMoonSoftness.x;
    float clockOpening = 1. - smoothstep(0., uMoonSoftness.y, clockDistance);
    float nestDistance = length((p - uMoonNest.xy) / uMoonNest.zw);
    float nestOpening = (1. - smoothstep(uMoonNestFeather.x,
      uMoonNestFeather.y, nestDistance)) * uMoonProfile.w;
    // A smooth union avoids a crease where the two canopy openings meet.
    float opening = 1. - (1. - clockOpening) * (1. - nestOpening);
    return mix(1., mix(uMoonProfile.y, uMoonProfile.z, opening), uMoonProfile.x);
  }
`;
const WORLD_POSITION = `
  // Run after wind, morphing, skinning and displacement. Instanced ant parts
  // and leaves must use the same world projection as the terrain and stones.
  vec4 moonClearingWorld = vec4(transformed, 1.);
  #ifdef USE_BATCHING
    moonClearingWorld = batchingMatrix * moonClearingWorld;
  #endif
  #ifdef USE_INSTANCING
    moonClearingWorld = instanceMatrix * moonClearingWorld;
  #endif
  vMoonClearingPosition = (modelMatrix * moonClearingWorld).xyz;
`;

function replaceOnce(source, find, replacement, label) {
  const at = source.indexOf(find);
  if (at < 0 || source.indexOf(find, at + find.length) >= 0) throw new Error(`Moonlight: unsupported ${label} shader layout.`);
  return source.slice(0, at) + replacement + source.slice(at + find.length);
}

// A faint beam of moonlit air through the opening above the clock.
function createShaft(group, uniforms) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, 0, 0, 1, 0, 0, -1, 1, 0, 1, 1, 0, 0, 0, -1, 0, 0, 1, 0, 1, -1, 0, 1, 1], 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1], 2));
  geometry.setIndex([0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7]);
  const material = new THREE.ShaderMaterial({
    name: 'Feathered lunar air scattering', uniforms, transparent: true, depthTest: true, depthWrite: false,
    side: THREE.DoubleSide, forceSinglePass: true, toneMapped: true,
    vertexShader: `
      uniform vec3 uMoonDirection;
      uniform vec3 uMoonShaftOrigin;
      uniform vec2 uMoonShaftSize;
      varying vec2 vShaftUV;
      varying float vShaftDepth;
      void main() {
        vec3 axis = normalize(uMoonDirection);
        vec3 side = normalize(cross(axis, vec3(0., 0., 1.)));
        vec3 other = cross(axis, side);
        float width = uMoonShaftSize.y * (1. + position.y * .12);
        vec3 world = uMoonShaftOrigin + axis * position.y * uMoonShaftSize.x
          + (side * position.x + other * position.z) * width;
        vec4 viewPosition = viewMatrix * vec4(world, 1.);
        vShaftDepth = max(0., -viewPosition.z);
        vShaftUV = uv;
        gl_Position = projectionMatrix * viewPosition;
      }
    `,
    fragmentShader: `
      uniform float uMoonShaftStrength;
      uniform float uMoonFogDensity;
      varying vec2 vShaftUV;
      varying float vShaftDepth;
      void main() {
        float across = abs(vShaftUV.x * 2. - 1.);
        float radial = exp(-across * across * 4.) * (1. - smoothstep(.62, 1., across));
        float axial = smoothstep(0., .16, vShaftUV.y)
          * (1. - smoothstep(.62, 1., vShaftUV.y));
        float air = exp(-uMoonFogDensity * uMoonFogDensity * vShaftDepth * vShaftDepth);
        float alpha = radial * axial * air * uMoonShaftStrength;
        if (alpha < .00001) discard;
        gl_FragColor = vec4(.36, .46, .63, alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'Moonlight through one canopy opening';
  mesh.frustumCulled = false; mesh.renderOrder = 1; mesh.userData.nonInteractive = true;
  group.add(mesh);
  return {mesh, geometry, material};
}

export function createMoonlight(scene, camera, {heightAt = () => 0, shaft = true} = {}) {
  const config = MOON_CLEARING;
  const clockHeight = finite(heightAt(...config.clockCenter), 0);
  const uniforms = {
    uMoonProfile: {value: new THREE.Vector4(0, config.outerTransmission, 1, config.nestStrength)},
    uMoonProjection: {value: new THREE.Vector3(0, 0, clockHeight)},
    uMoonClock: {value: new THREE.Vector4(...config.clockCenter, ...config.clockHalfSize)},
    uMoonSoftness: {value: new THREE.Vector2(config.clockCornerRadius, config.clockFeather)},
    uMoonNest: {value: new THREE.Vector4(...config.nestCenter, ...config.nestRadii)},
    uMoonNestFeather: {value: new THREE.Vector2(config.nestInner, config.nestOuter)},
    uMoonDirection: {value: new THREE.Vector3(-.45, .82, .35).normalize()},
    uMoonShaftOrigin: {value: new THREE.Vector3(-1.8, finite(heightAt(-1.8, -1.45), clockHeight) + .14, -1.45)},
    uMoonShaftSize: {value: new THREE.Vector2(config.shaftLength, config.shaftRadius)},
    uMoonShaftStrength: {value: 0},
    uMoonFogDensity: {value: .01}
  };
  const group = new THREE.Group();
  group.name = 'Lunar canopy opening'; group.userData.nonInteractive = true;
  scene.add(group);
  const beam = shaft ? createShaft(group, uniforms) : null;
  const installed = new Map();
  const profile = {night: 0, cloudTransmission: 1, peak: 1, outer: config.outerTransmission, shaftStrength: 0};
  let disposed = false;
  const directional = 'getDirectionalLightInfo( directionalLight, directLight );';
  const lighting = replaceOnce(THREE.ShaderChunk.lights_fragment_begin, directional, directional + `
      // The atmosphere owns the single shadow-casting sun/moon source. Keep
      // any future secondary directional lights outside this canopy mask.
      #if ( UNROLLED_LOOP_INDEX == 0 )
        if (uMoonProfile.x > 0.) {
          directLight.color *= moonClearingTransmission(vMoonClearingPosition);
        }
      #endif
    `, 'directional-light');

  // Patches every standard material in the scene, and any listed that are not
  // in it yet, such as the crumbs. A material is patched once.
  function installMaterials(extra = []) {
    if (disposed) return 0;
    const materials = new Set();
    const collect = material => {
      if (Array.isArray(material)) material.forEach(collect);
      else if (material?.isMeshStandardMaterial || material?.isMeshPhysicalMaterial) materials.add(material);
    };
    scene.traverse(object => collect(object.material));
    collect(extra);
    let count = 0;
    for (const material of materials) {
      if (installed.has(material)) continue;
      const originalCompile = material.onBeforeCompile, originalKey = material.customProgramCacheKey;
      const defaultKey = originalKey === THREE.Material.prototype.customProgramCacheKey;
      const compileSource = defaultKey ? originalCompile.toString() : null;
      const compile = function(shader, renderer) {
        originalCompile.call(this, shader, renderer);
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = replaceOnce(shader.vertexShader, '#include <common>', '#include <common>\n' + VERTEX_DECLARATIONS, 'vertex declarations');
        shader.vertexShader = replaceOnce(shader.vertexShader, '#include <worldpos_vertex>', '#include <worldpos_vertex>\n' + WORLD_POSITION, 'world-position');
        shader.fragmentShader = replaceOnce(shader.fragmentShader, '#include <common>', '#include <common>\n' + FRAGMENT_DECLARATIONS, 'fragment declarations');
        shader.fragmentShader = replaceOnce(shader.fragmentShader, '#include <lights_fragment_begin>', lighting, 'lighting');
      };
      const key = function() {
        return (defaultKey ? compileSource : originalKey.call(this)) + '|' + config.version;
      };
      installed.set(material, {originalCompile, originalKey, compile, key});
      material.onBeforeCompile = compile; material.customProgramCacheKey = key; material.needsUpdate = true;
      count++;
    }
    return count;
  }

  function update(environment = {}) {
    if (disposed) return profile;
    const daylight = clamp(finite(environment.daylight, 1)), cloud = clamp(finite(environment.cloud, 0)), rain = clamp(finite(environment.rain, 0));
    profile.night = 1 - smooth(0, .4, daylight);
    profile.cloudTransmission = clamp(1 - cloud * .7 - rain * .25);
    profile.peak = config.overcastPeak + (1 - config.overcastPeak) * profile.cloudTransmission;
    profile.shaftStrength = shaft
      ? config.shaftOpacity * profile.night * profile.cloudTransmission * profile.cloudTransmission * (1 - rain) * (1 + clamp(finite(environment.mist, 0)) * .25)
      : 0;
    const direction = uniforms.uMoonDirection.value;
    if (environment.moonDirection && Number.isFinite(environment.moonDirection.x) && Number.isFinite(environment.moonDirection.y) && Number.isFinite(environment.moonDirection.z)) direction.copy(environment.moonDirection);
    direction.y = Math.max(.12, direction.y);
    if (direction.lengthSq() < 1e-10) direction.set(0, 1, 0);
    direction.normalize();
    uniforms.uMoonProjection.value.set(direction.x / direction.y, direction.z / direction.y, clockHeight);
    uniforms.uMoonProfile.value.set(profile.night, profile.outer, profile.peak, config.nestStrength);
    uniforms.uMoonShaftStrength.value = profile.shaftStrength;
    uniforms.uMoonFogDensity.value = clamp(finite(environment.fogDensity, .01), 0, .1);
    return profile;
  }

  // The same mask as the shader, computed here for checks. It tells how much
  // moonlight reaches a point.
  function sampleAt(x, y, z) {
    const projection = uniforms.uMoonProjection.value;
    const px = x - (y - projection.z) * projection.x, pz = z - (y - projection.z) * projection.y;
    const clock = uniforms.uMoonClock.value, softness = uniforms.uMoonSoftness.value;
    const dx = Math.abs(px - clock.x) - clock.z, dz = Math.abs(pz - clock.y) - clock.w;
    const clockDistance = Math.hypot(Math.max(dx, 0), Math.max(dz, 0)) + Math.min(Math.max(dx, dz), 0) - softness.x;
    const clockOpening = 1 - smooth(0, softness.y, clockDistance);
    const nest = uniforms.uMoonNest.value, feather = uniforms.uMoonNestFeather.value;
    const nestDistance = Math.hypot((px - nest.x) / nest.z, (pz - nest.y) / nest.w);
    const moonProfile = uniforms.uMoonProfile.value;
    const nestOpening = (1 - smooth(feather.x, feather.y, nestDistance)) * moonProfile.w;
    const mask = 1 - (1 - clockOpening) * (1 - nestOpening);
    const multiplier = 1 + (moonProfile.y + (moonProfile.z - moonProfile.y) * mask - 1) * moonProfile.x;
    return {mask, multiplier, projectedX: px, projectedZ: pz};
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    uniforms.uMoonProfile.value.x = 0; uniforms.uMoonShaftStrength.value = 0;
    for (const [material, entry] of installed) {
      if (material.onBeforeCompile === entry.compile) material.onBeforeCompile = entry.originalCompile;
      if (material.customProgramCacheKey === entry.key) material.customProgramCacheKey = entry.originalKey;
      material.needsUpdate = true;
    }
    installed.clear();
    scene.remove(group);
    beam?.geometry.dispose(); beam?.material.dispose();
  }

  return {installMaterials, update, dispose, uniforms, config, debug: {get materialCount() { return installed.size; }, get profile() { return {...profile}; }, sampleAt}, group};
}
