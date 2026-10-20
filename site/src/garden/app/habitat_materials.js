import * as THREE from 'three';
import {rng, noise2, fbm, clamp} from './math.js';

// Small, deterministic material maps. The garden has no network assets and no
// per-frame texture allocation; the fine grain continues through every season.
function texture(size, kind, seed) {
  const data = new Uint8Array(size * size * 4), random = rng(seed);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = (y * size + x) * 4, u = x / size, v = y / size;
    const n = random(), broad = fbm(u * 12, v * 12), fine = noise2(u * 71, v * 71);
    let r, g, b;
    if (kind === 'wood') {
      const knot = Math.sin(u * 155 + Math.sin(v * 15) * 1.8 + Math.sin(v * 3.1) * 3.2);
      const grain = Math.pow(Math.abs(knot), 13) * 31 + noise2(u * 84, v * 3.2) * 34;
      const t = 122 + broad * 49 - grain + (n - .5) * 16;
      r = t; g = t * .78; b = t * .57;
    } else if (kind === 'bark') {
      const ridges = Math.pow(Math.abs(Math.sin(u * 57 + noise2(u * 9, v * 22) * 4.2 + Math.sin(v * 18) * .65)), 8);
      const t = 101 + broad * 38 - ridges * 41 + (n - .5) * 24;
      r = t * .98; g = t * .87; b = t * .74;
    } else if (kind === 'stone') {
      const grain = (n - .5) * 35 + (fine - .5) * 36;
      const veins = Math.pow(Math.max(0, Math.sin(u * 14.5 + v * 9 + broad * 5.5)), 29) * 20;
      const t = 121 + (broad - .5) * 67 + grain + veins;
      r = t * 1.01; g = t * 1.025; b = t * .99;
    } else if (kind === 'moss') {
      const t = 70 + broad * 96 + n * 27;
      const pin = n > .966 ? 34 : 0;
      r = t * .74 + pin; g = t * .81 + pin; b = t * .40;
    } else if (kind === 'earth') {
      const grit = n > .984 ? 34 : n < .06 ? -27 : 0;
      const t = 66 + broad * 42 + (n - .5) * 29 + grit;
      r = t * 1.03; g = t * .95; b = t * .77;
    } else if (kind === 'plaster') {
      const t = 209 + (n - .5) * 13 + (broad - .5) * 22;
      r = t * 1.04; g = t * 1.015; b = t * .91;
    } else if (kind === 'tile') {
      const t = 115 + broad * 24 + (n - .5) * 20;
      r = t * .91; g = t * .98; b = t * 1.02;
    } else {
      const mid = Math.exp(-Math.abs(u - .5) * 65);
      const veins = Math.pow(Math.max(0, Math.sin(v * 80 + Math.abs(u - .5) * 30)), 18);
      const t = 177 + broad * 37 + (n - .5) * 12 + mid * 30 + veins * 13;
      r = t * .91; g = t; b = t * .73;
    }
    data[i] = clamp(Math.round(r), 0, 255); data[i + 1] = clamp(Math.round(g), 0, 255);
    data[i + 2] = clamp(Math.round(b), 0, 255); data[i + 3] = 255;
  }
  const map = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  map.name = 'Garden procedural ' + kind;
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = map.wrapT = THREE.RepeatWrapping;
  map.magFilter = THREE.LinearFilter; map.minFilter = THREE.LinearMipmapLinearFilter;
  map.generateMipmaps = true; map.anisotropy = 4; map.needsUpdate = true;
  return map;
}

export function createGardenMaterials() {
  const maps = Object.fromEntries(['wood', 'bark', 'stone', 'moss', 'earth', 'plaster', 'tile', 'leaf']
    .map((name, i) => [name, texture(name === 'leaf' ? 128 : 256, name, 8421 + i * 83)]));
  const uniforms = {time: {value: 0}, wet: {value: 0}, wind: {value: .2}, snow: {value: 0}};
  const materials = [];
  function material(options, {wind = 0, wet = .5, snow = 0, viewClearance = false} = {}) {
    const mat = new THREE.MeshStandardMaterial(options);
    mat.alphaToCoverage=viewClearance;
    const settings = {value: new THREE.Vector3(wind, wet, snow)};
    mat.onBeforeCompile = shader => {
      shader.uniforms.gardenTime = uniforms.time; shader.uniforms.gardenWet = uniforms.wet;
      shader.uniforms.gardenWind = uniforms.wind; shader.uniforms.gardenSnow = uniforms.snow;
      shader.uniforms.gardenResponse = settings;
      shader.vertexShader = 'uniform float gardenTime; uniform float gardenWind; uniform vec3 gardenResponse;\nvarying vec3 gardenNormal; varying vec3 gardenPosition;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
        #include <begin_vertex>
        vec3 gardenWorld = transformed;
        #ifdef USE_INSTANCING
          gardenWorld = (instanceMatrix * vec4(gardenWorld, 1.)).xyz;
        #endif
        gardenWorld = (modelMatrix * vec4(gardenWorld, 1.)).xyz;
        gardenPosition = gardenWorld;
        float gardenBreeze = sin(gardenWorld.x * .72 + gardenWorld.z * .39 + gardenTime * 1.16)
          + .38 * sin(gardenWorld.x * 2.9 - gardenTime * 1.81);
        transformed.x += gardenBreeze * gardenResponse.x * gardenWind;
        transformed.z += cos(gardenTime * .83 + gardenWorld.z * .7) * gardenResponse.x * gardenWind * .5;
        gardenNormal = normalize(mat3(modelMatrix) * objectNormal);
      `);
      shader.fragmentShader = 'uniform float gardenWet; uniform float gardenSnow; uniform vec3 gardenResponse;\nvarying vec3 gardenNormal; varying vec3 gardenPosition;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
        #include <color_fragment>
        diffuseColor.rgb *= 1. - gardenWet * gardenResponse.y * .31;
        float gardenDust = smoothstep(.28, .90, gardenNormal.y) * gardenSnow * gardenResponse.z;
        diffuseColor.rgb = mix(diffuseColor.rgb, vec3(.79, .82, .80), gardenDust);
      `).replace('#include <roughnessmap_fragment>', `
        #include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, max(.27, roughnessFactor * .65), gardenWet * gardenResponse.y);
      `);
      if(viewClearance)shader.fragmentShader=shader.fragmentShader.replace('#include <alphatest_fragment>',`
        #include <alphatest_fragment>
        // Only the fixed exterior trees between the lens and the courtyard
        // make room for the view. The opposite forest remains fully opaque.
        vec3 gardenView=normalize(-vec3(viewMatrix[0][2],viewMatrix[1][2],viewMatrix[2][2]));
        float gardenViewLength=max(.1,(cameraPosition.y+.10)/max(.1,-gardenView.y));
        vec3 gardenRay=gardenPosition-cameraPosition;
        float gardenDepth=dot(gardenRay,gardenView);
        float gardenOffset=length(gardenRay-gardenView*gardenDepth);
        float gardenProgress=clamp(gardenDepth/gardenViewLength,0.,1.);
        float gardenRadius=mix(1.15,6.8,gardenProgress);
        float gardenCorridor=1.-smoothstep(gardenRadius,gardenRadius+1.5,gardenOffset);
        float gardenFront=step(0.,gardenDepth)*(1.-smoothstep(gardenViewLength-2.8,gardenViewLength-.5,gardenDepth));
        float gardenExterior=max(smoothstep(7.3,8.,abs(gardenPosition.x)),max(smoothstep(6.2,7.,gardenPosition.z),smoothstep(8.6,9.3,-gardenPosition.z)));
        float gardenVisibility=1.-gardenCorridor*gardenFront*gardenExterior;
        diffuseColor.a*=gardenVisibility;
        if(gardenVisibility<.00001)discard;
      `);
    };
    // All material types use one shader structure; response is a uniform.
    mat.customProgramCacheKey = () => 'karesansui-material-weather-v4'+(viewClearance?'-forest-view':'');
    materials.push(mat); return mat;
  }
  const m = {
    timber: material({color: '#ddd1b8', map: maps.wood, bumpMap: maps.wood, bumpScale: .024, roughness: .96}, {wet: .7, snow: .4}),
    darkWood: material({color: '#aca79b', map: maps.wood, bumpMap: maps.wood, bumpScale: .021, roughness: .94}, {wet: .7, snow: .4}),
    newWood: material({color: '#eee0c4', map: maps.wood, bumpMap: maps.wood, bumpScale: .009, roughness: .86}, {wet: .55}),
    plaster: material({map: maps.plaster, color: '#e2dbc5', bumpMap: maps.plaster, bumpScale: .018, roughness: 1}, {wet: .22}),
    paper: material({map: maps.plaster, color: '#ebddb9', roughness: 1, emissive: '#d19d54', emissiveIntensity: .03, side: THREE.DoubleSide}, {wet: 0}),
    tile: material({map: maps.tile, color: '#d0d3cf', bumpMap: maps.tile, bumpScale: .014, roughness: .88}, {wet: .75, snow: .7}),
    stone: material({map: maps.stone, bumpMap: maps.stone, bumpScale: .047, color: '#c9cbc4', roughness: .97, vertexColors: true}, {wet: .8, snow: .7}),
    granite: material({map: maps.stone, bumpMap: maps.stone, bumpScale: .022, color: '#dedbd0', roughness: .98}, {wet: .8, snow: .65}),
    moss: material({map: maps.moss, bumpMap: maps.moss, bumpScale: .024, color: '#a5a487', roughness: 1, vertexColors: true}, {wet: .3, snow: .7}),
    earth: material({map: maps.earth, bumpMap: maps.earth, bumpScale: .053, color: '#9b9c80', roughness: 1}, {wet: .8, snow: .8}),
    bark: material({map: maps.bark, bumpMap: maps.bark, bumpScale: .043, color: '#c3bbad', roughness: .97}, {wet: .8, snow: .2, viewClearance:true}),
    foliage: material({map: maps.leaf, color: '#69754d', roughness: .87, vertexColors: true, side: THREE.DoubleSide}, {wind: .035, wet: .4, snow: .8, viewClearance:true}),
    evergreen: material({map: maps.leaf, color: '#52664b', roughness: .92, vertexColors: true, side: THREE.DoubleSide}, {wind: .023, wet: .4, snow: .9, viewClearance:true}),
    shrubs: material({map: maps.leaf, color: '#59673a', roughness: .9, vertexColors: true, side: THREE.DoubleSide}, {wind: .013, wet: .4, snow: .7, viewClearance:true}),
    bamboo: material({map: maps.wood, bumpMap: maps.wood, bumpScale: .013, color: '#9e9970', roughness: .84}, {wet: .6, snow: .4}),
    iron: material({color: '#403b32', roughness: .78, metalness: .25}, {wet: .7}),
    ceramic: material({color: '#5d766b', roughness: .47, metalness: .02}, {wet: .35}),
    water: material({color: '#4c6966', roughness: .16, metalness: .28, transparent: true, opacity: .82}, {wet: 0}),
    glow: material({color: '#ffdfa4', emissive: '#ffc77f', emissiveIntensity: .6, roughness: 1}, {wet: 0}),
    interior: material({color: '#5b5242', map: maps.wood, roughness: .99}, {wet: 0}),
  };
  maps.earth.repeat.set(48, 48);
  const green = new THREE.Color('#69754d'), spring = new THREE.Color('#879052'), autumn = new THREE.Color('#987047'), winter = new THREE.Color('#717263');
  let lastSeason = -1;
  return {m, maps, materials, update(env, time) {
    uniforms.time.value = Number.isFinite(time) ? time : 0;
    const finite = (value, fallback = 0) => Number.isFinite(value) ? value : fallback;
    const wind = Number.isFinite(env.wind) ? env.wind : Math.hypot(finite(env.wind?.x), finite(env.wind?.y), finite(env.wind?.z));
    uniforms.wet.value = clamp(finite(env.wetness)); uniforms.wind.value = .25 + clamp(wind) * .85;
    // snow is settled cover. Snowfall alone should not instantly whiten every
    // surface, and an absent / malformed quantity must never poison vertices.
    uniforms.snow.value = clamp(finite(env.snow));
    const season = finite(env.season, 1);
    if (Array.isArray(env.seasonBlend) && env.seasonBlend.length === 4 && env.seasonBlend.every(Number.isFinite)) {
      const palette = [spring, green, autumn, winter], sum = env.seasonBlend.reduce((a,b) => a + Math.max(0,b), 0);
      if (sum > 0) {
        m.foliage.color.setRGB(0,0,0);
        for (let i = 0; i < 4; i++) {
          const weight = Math.max(0, env.seasonBlend[i]) / sum;
          m.foliage.color.r += palette[i].r * weight; m.foliage.color.g += palette[i].g * weight; m.foliage.color.b += palette[i].b * weight;
        }
      }
      lastSeason = -1;
    } else if (season !== lastSeason) {
      m.foliage.color.copy([spring, green, autumn, winter][Math.round(season) % 4] || green);
      lastSeason = season;
    }
    m.paper.emissiveIntensity = .04 + clamp(finite(env.night, 1-finite(env.daylight,1))) * .45;
  }, dispose() {materials.forEach(mat => mat.dispose()); Object.values(maps).forEach(map => map.dispose());}};
}
