import * as THREE from 'three';
import {createPbrTextures} from './habitat_textures.js';
import {createForestBackdrop} from './forest.js';

/* A persistent, entirely geometric forest floor. No photographic backplate and
   no state-dependent regeneration: weather changes materials, not the world. */
const TAU = Math.PI * 2;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mix = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const UP = new THREE.Vector3(0, 1, 0);
function randomSource(seed) {
  let s = seed >>> 0;
  return () => { s += 0x6D2B79F5; let t = Math.imul(s ^ s >>> 15, 1 | s); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
function noiseSource(seed) {
  const hash = (x, y) => {
    let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ seed;
    h = Math.imul(h ^ h >>> 13, 1274126177);
    return ((h ^ h >>> 16) >>> 0) / 4294967295;
  };
  return (x, y) => {
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
    return mix(mix(hash(ix, iy), hash(ix + 1, iy), u), mix(hash(ix, iy + 1), hash(ix + 1, iy + 1), u), v);
  };
}
function canvasTexture(width, height, draw, renderer, colour = true) {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const context = canvas.getContext('2d'); draw(context, width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = colour ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = Math.min(8, renderer?.capabilities?.getMaxAnisotropy?.() || 1);
  return texture;
}
function createTextures(renderer, seed) {
  const rnd = randomSource(seed + 9182), noise = noiseSource(seed + 771);
  const pbr = createPbrTextures(renderer);
  const soilBump = canvasTexture(512, 512, (ctx, w, h) => {
    const image = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = Math.floor(105 + noise(x / 3, y / 3) * 94 + (rnd() - .5) * 40), i = (y * w + x) * 4;
      image.data[i] = image.data[i + 1] = image.data[i + 2] = v; image.data[i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, renderer, false);
  soilBump.repeat.set(19, 15);
  const rings = canvasTexture(256, 256, (ctx, w, h) => {
    const image = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dx = x - w * .47, dy = y - h * .53, r = Math.hypot(dx, dy), n = noise(x / 19, y / 19);
      const ring = Math.pow(.5 + .5 * Math.sin(r * .65 + n * 2.4), 10), v = .88 - ring * .24 + (rnd() - .5) * .1;
      const i = (y * w + x) * 4;
      image.data[i] = 164 * v; image.data[i + 1] = 133 * v; image.data[i + 2] = 88 * v; image.data[i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    ctx.strokeStyle = 'rgba(48,30,16,.4)'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const a = rnd() * TAU; ctx.beginPath(); ctx.moveTo(w / 2 + Math.cos(a) * 23, h / 2 + Math.sin(a) * 23);
      ctx.lineTo(w / 2 + Math.cos(a + .03) * 80, h / 2 + Math.sin(a + .03) * 80); ctx.lineTo(w / 2 + Math.cos(a) * 126, h / 2 + Math.sin(a) * 126); ctx.stroke();
    }
  }, renderer);
  const leaf = canvasTexture(128, 256, (ctx, w, h) => {
    ctx.fillStyle = '#c5d292'; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 4500; i++) {
      ctx.fillStyle = `rgba(${rnd() < .5 ? '64,93,44' : '225,229,183'},.09)`; ctx.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 3, 1 + rnd() * 3);
    }
    ctx.strokeStyle = 'rgba(92,119,52,.44)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(w / 2, h); ctx.lineTo(w / 2, 0); ctx.stroke();
    ctx.lineWidth = 1;
    for (let y = 22; y < h; y += 23) for (const side of [-1, 1]) {
      ctx.beginPath(); ctx.moveTo(w / 2, y + 14); ctx.quadraticCurveTo(w / 2 + side * 18, y + 2, w / 2 + side * 57, y - 25); ctx.stroke();
    }
  }, renderer);
  const textures = {...pbr, soilBump, rings, leaf};
  Object.defineProperty(textures, 'ready', {value: pbr.ready, enumerable: false});
  return textures;
}

// A leaf has a raised midrib and curved margin, not a camera-facing billboard.
function leafGeometry(length = 1, width = .42, curl = .12, segments = 8) {
  const positions = [], uvs = [], indices = [];
  for (let j = 0; j <= segments; j++) {
    const t = j / segments, shape = Math.pow(Math.sin(Math.PI * t), .78), half = width * shape / 2;
    for (let i = 0; i < 3; i++) {
      const side = i - 1;
      positions.push(side * half, curl * t * t + (1 - Math.abs(side)) * .065 * shape, t * length);
      uvs.push(i / 2, 1 - t);
    }
  }
  for (let j = 0; j < segments; j++) for (let i = 0; i < 2; i++) {
    const a = j * 3 + i, b = a + 1, c = a + 3, d = c + 1;
    indices.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2)); g.setIndex(indices); g.computeVertexNormals(); return g;
}
function appendGeometry(destination, geometry, matrix, tint = null) {
  const pos = geometry.attributes.position, uv = geometry.attributes.uv, index = geometry.index;
  const start = destination.positions.length / 3, v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).applyMatrix4(matrix); destination.positions.push(v.x, v.y, v.z);
    destination.uvs.push(uv ? uv.getX(i) : 0, uv ? uv.getY(i) : 0);
    if (destination.colours) destination.colours.push(tint?.r ?? 1, tint?.g ?? 1, tint?.b ?? 1);
  }
  if (index) for (let i = 0; i < index.count; i++) destination.indices.push(start + index.getX(i));
  else for (let i = 0; i < pos.count; i++) destination.indices.push(start + i);
}
function geometryFromData(data) {
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(data.uvs, 2)); g.setIndex(data.indices);
  if (data.colours) g.setAttribute('color', new THREE.Float32BufferAttribute(data.colours, 3));
  g.computeVertexNormals(); g.computeBoundingSphere(); return g;
}
function fernGeometry() {
  const data = {positions: [], uvs: [], indices: [], colours: []}, m = new THREE.Matrix4(), o = new THREE.Object3D();
  const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, .48, .18), new THREE.Vector3(0, .85, .72), new THREE.Vector3(0, .7, 1.18)]);
  appendGeometry(data, new THREE.TubeGeometry(curve, 14, .013, 4, false), m, new THREE.Color('#b7ca70'));
  for (let j = 1; j <= 12; j++) {
    const t = j / 13, p = curve.getPoint(t), length = .48 * Math.pow(Math.sin(Math.PI * t), .75) + .03;
    for (const side of [-1, 1]) {
      o.position.copy(p); o.rotation.set(-.06 - t * .13, side * (.99 + t * .18), side * -.08); o.scale.set(1, 1, 1); o.updateMatrix();
      const leaflet = leafGeometry(length, length * .32, .027, 4);
      appendGeometry(data, leaflet, o.matrix, new THREE.Color().setRGB(.82 + t * .12, .91 + t * .08, .67 + t * .17)); leaflet.dispose();
    }
  }
  return geometryFromData(data);
}
function grassGeometry(seed) {
  const data = {positions: [], uvs: [], indices: []}, rnd = randomSource(seed), m = new THREE.Object3D();
  for (let i = 0; i < 5; i++) {
    const blade = leafGeometry(.3 + rnd() * .45, .024 + rnd() * .017, .035, 4);
    m.position.set((rnd() - .5) * .12, 0, (rnd() - .5) * .12); m.rotation.set(-1.15 + rnd() * .32, rnd() * TAU, (rnd() - .5) * .28); m.updateMatrix();
    appendGeometry(data, blade, m.matrix); blade.dispose();
  }
  return geometryFromData(data);
}

export function createHabitat(scene, renderer, {seed = 41871, quality = 'high', startView} = {}) {
  const group = new THREE.Group(); group.name = 'Persistent woodland habitat'; scene.add(group);
  const rnd = randomSource(seed), noise = noiseSource(seed), textures = createTextures(renderer, seed);
  const low = quality === 'low', density = low ? .53 : 1;
  const bounds = {minX: -18, maxX: 18, minZ: -14, maxZ: 14};
  const nx = low ? 120 : 180, nz = low ? 94 : 140, dx = 36 / nx, dz = 28 / nz;
  const positions = new Float32Array((nx + 1) * (nz + 1) * 3), colours = new Float32Array(positions.length), uvs = new Float32Array((nx + 1) * (nz + 1) * 2);
  const indices = new Uint32Array(nx * nz * 6);
  const heightField = new Float32Array((nx + 1) * (nz + 1));
  const dirtColour = new THREE.Color('#e4d7b8'), mossColour = new THREE.Color('#62853a'), shadeColour = new THREE.Color('#b9b49a'), c = new THREE.Color();
  const gaussian = (x, z, cx, cz, radius) => Math.exp(-((x - cx) ** 2 + (z - cz) ** 2) / radius ** 2);
  function analyticHeight(x, z) {
    const central = 1 - smooth(.87, 1.2, Math.max(Math.abs(x) / 9.3, Math.abs(z + 1.4) / 3.2));
    const broad = .16 * Math.sin(x * .27 + .6) * Math.cos(z * .3) + (noise(x * .34 + 8, z * .34 + 12) - .5) * .28;
    const rim = smooth(8.8, 18, Math.max(Math.abs(x), Math.abs(z) * 1.28)) * .42;
    const mounds = .75 * gaussian(x, z, -12, -5, 3.7) + .62 * gaussian(x, z, 12, -8, 3.6) + .4 * gaussian(x, z, -11, 7.5, 3.2);
    const fine = (noise(x * 1.5 + 27, z * 1.5 + 17) - .5) * .045;
    const nestRadius = Math.hypot(x - 9, z - 4), nest = .24 * Math.exp(-(((nestRadius - .46) / .22) ** 2)) - .19 * Math.exp(-((nestRadius / .17) ** 2));
    return -.15 + broad * (1 - .88 * central) + (rim + mounds) * (1 - central) + fine + nest;
  }
  function protectedGround(x, z, margin = 0) {
    const digitArea = Math.abs(x) < 8.95 + margin && z > -4.15 - margin && z < .95 + margin;
    const nestApproach = x > 6.4 - margin && x < 10.8 + margin && z > -.15 - margin && z < 5.35 + margin;
    return digitArea || nestApproach;
  }
  function mossAt(x, z) {
    const edge = smooth(.65, 1.23, Math.max(Math.abs(x) / 10.8, Math.abs(z + .6) / 6.2));
    const patch = smooth(.34, .74, noise(x * .46 + 31, z * .46 + 71));
    return clamp(edge * patch + gaussian(x, z, -11, 3, 2.2) * .6 + gaussian(x, z, 10.7, -7.5, 2.1) * .45) * (protectedGround(x, z) ? .08 : 1);
  }
  for (let j = 0; j <= nz; j++) for (let i = 0; i <= nx; i++) {
    const id = j * (nx + 1) + i, x = bounds.minX + i * dx, z = bounds.minZ + j * dz, y = analyticHeight(x, z);
    heightField[id] = y; positions[id * 3] = x; positions[id * 3 + 1] = y; positions[id * 3 + 2] = z;
    uvs[id * 2] = i / nx; uvs[id * 2 + 1] = j / nz;
    c.copy(dirtColour).lerp(shadeColour, noise(x * .21 + 1, z * .21 + 3) * .5).lerp(mossColour, mossAt(x, z));
    c.multiplyScalar(.75 + .25 * noise(x * .65 + 41, z * .65 + 59));
    const nestDark = 1 - .66 * gaussian(x, z, 9, 4, .2); c.multiplyScalar(nestDark);
    colours[id * 3] = c.r; colours[id * 3 + 1] = c.g; colours[id * 3 + 2] = c.b;
  }
  for (let j = 0, k = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i, b = a + 1, cc = a + nx + 1, d = cc + 1;
    indices[k++] = a; indices[k++] = cc; indices[k++] = b; indices[k++] = b; indices[k++] = cc; indices[k++] = d;
  }
  function cellAt(x, z) {
    const gx = clamp((x - bounds.minX) / dx, 0, nx - 1e-7), gz = clamp((z - bounds.minZ) / dz, 0, nz - 1e-7);
    const ix = Math.floor(gx), iz = Math.floor(gz); return {a: iz * (nx + 1) + ix, u: gx - ix, v: gz - iz};
  }
  // Barycentric interpolation uses precisely the same b-c diagonal as the mesh.
  function heightAt(x, z) {
    const {a, u, v} = cellAt(x, z), b = a + 1, cc = a + nx + 1, d = cc + 1;
    return u + v <= 1 ? heightField[a] * (1 - u - v) + heightField[b] * u + heightField[cc] * v
      : heightField[d] * (u + v - 1) + heightField[cc] * (1 - u) + heightField[b] * (1 - v);
  }
  function normalAt(x, z) {
    const {a, u, v} = cellAt(x, z), b = a + 1, cc = a + nx + 1, d = cc + 1;
    const sx = u + v <= 1 ? (heightField[b] - heightField[a]) / dx : (heightField[d] - heightField[cc]) / dx;
    const sz = u + v <= 1 ? (heightField[cc] - heightField[a]) / dz : (heightField[d] - heightField[b]) / dz;
    return new THREE.Vector3(-sx, 1, -sz).normalize();
  }
  const uniforms = {uHabitatWet: {value: 0}, uHabitatSnow: {value: 0}, uHabitatWind: {value: .18}, uHabitatTime: {value: 0}};
  const materials = [];
  const shaderCommon = `
    varying vec3 vHabitatPosition;
    varying vec3 vHabitatNormal;
    uniform float uHabitatTime;
    uniform float uHabitatWind;
  `;
  function windCode() {
    return `
      float habitatPhase = 0.;
      #ifdef USE_INSTANCING
        habitatPhase = instanceMatrix[3].x * 1.37 + instanceMatrix[3].z * .83;
      #endif
      float habitatBend = pow(max(0., position.y), 1.3) * uHabitatBend;
      transformed.x += sin(uHabitatTime * 1.9 + habitatPhase + position.y * 1.2) * habitatBend * uHabitatWind;
      transformed.z += cos(uHabitatTime * 1.35 + habitatPhase * .81) * habitatBend * uHabitatWind * .47;
    `;
  }
  function naturalMaterial(parameters, {wind = 0, snow = 1, wet = 1} = {}) {
    const material = new THREE.MeshStandardMaterial(parameters);
    material.userData.habitat = {wind, snow, wet};
    // Share shader programs across surface coefficients while keeping every
    // material's values separate. Match the former GLSL literal precision.
    const surface = {value: new THREE.Vector3(Number(snow.toFixed(3)), Number(wet.toFixed(3)), Number((.28 * wet).toFixed(3)))};
    const hasWind = wind !== 0, bend = {value: Number(wind.toFixed(4))};
    material.onBeforeCompile = shader => {
      Object.assign(shader.uniforms, uniforms);
      shader.uniforms.uHabitatSurface = surface;
      if (hasWind) shader.uniforms.uHabitatBend = bend;
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\n' + shaderCommon + (hasWind ? '\nuniform float uHabitatBend;\n' : ''))
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + (hasWind ? windCode() : '') + `
          vec4 habitatP = vec4(transformed, 1.);
          vec3 habitatN = objectNormal;
          #ifdef USE_INSTANCING
            habitatP = instanceMatrix * habitatP;
            habitatN = mat3(instanceMatrix) * habitatN;
          #endif
          vHabitatPosition = (modelMatrix * habitatP).xyz;
          vHabitatNormal = normalize(mat3(modelMatrix) * habitatN);
        `);
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', `#include <common>
          varying vec3 vHabitatPosition;
          varying vec3 vHabitatNormal;
          uniform float uHabitatWet;
          uniform float uHabitatSnow;
          uniform vec3 uHabitatSurface;
          float habitatHash(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * .1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
          }
          float habitatNoiseAt(vec2 p) {
            vec2 i = floor(p), f = fract(p);
            f = f * f * (3. - 2. * f);
            return mix(mix(habitatHash(i), habitatHash(i + vec2(1., 0.)), f.x),
                       mix(habitatHash(i + vec2(0., 1.)), habitatHash(i + vec2(1.)), f.x), f.y);
          }
        `).replace('#include <color_fragment>', `#include <color_fragment>
          vec2 habitatXZ = vHabitatPosition.xz;
          float habitatNoise = .56 * habitatNoiseAt(habitatXZ * .85)
            + .28 * habitatNoiseAt(habitatXZ * 2.4 + 7.3)
            + .16 * habitatNoiseAt(habitatXZ * 6.7 + 19.1);
          float habitatExposure = smoothstep(.12, .8, normalize(vHabitatNormal).y);
          float habitatSnow = smoothstep(habitatNoise * .58 + .10, habitatNoise * .58 + .33, uHabitatSnow) * habitatExposure * uHabitatSurface.x;
          diffuseColor.rgb *= 1. - uHabitatWet * uHabitatSurface.z;
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(.79, .84, .85), habitatSnow);
        `).replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
          roughnessFactor = mix(roughnessFactor, max(.26, roughnessFactor * .57), uHabitatWet * uHabitatSurface.y);
        `);
    };
    material.customProgramCacheKey = () => hasWind ? 'habitat-v4-wind' : 'habitat-v4-still';
    materials.push(material); return material;
  }
  function attachWindShadow(mesh, strength) {
    const depth = new THREE.MeshDepthMaterial({depthPacking: THREE.RGBADepthPacking, side: THREE.DoubleSide});
    const bend = {value: Number(strength.toFixed(4))};
    depth.onBeforeCompile = shader => {
      shader.uniforms.uHabitatTime = uniforms.uHabitatTime; shader.uniforms.uHabitatWind = uniforms.uHabitatWind;
      shader.uniforms.uHabitatBend = bend;
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nuniform float uHabitatTime; uniform float uHabitatWind; uniform float uHabitatBend;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + windCode());
    };
    depth.customProgramCacheKey = () => 'habitat-wind-depth-v4'; mesh.customDepthMaterial = depth;
  }
  const groundMaterial = naturalMaterial({map: textures.soil, normalMap: textures.soilNormal, normalScale: new THREE.Vector2(.44, .44), roughnessMap: textures.soilARM,
    aoMap: textures.soilARM, aoMapIntensity: .58, vertexColors: true, roughness: .98, metalness: 0});
  const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3)); geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2)); geometry.setIndex(new THREE.BufferAttribute(indices, 1)); geometry.computeVertexNormals();
  const ground = new THREE.Mesh(geometry, groundMaterial); ground.name = 'Collidable living soil'; ground.receiveShadow = true; ground.castShadow = false; group.add(ground);
  ground.userData.isGround = true;

  // A deep soil skirt allows low orbit angles without exposing a paper-thin plane.
  const skirtData = {positions: [], uvs: [], indices: [], colours: []}, boundary = [];
  for (let i = 0; i <= nx; i++) boundary.push([bounds.minX + i * dx, bounds.minZ]);
  for (let j = 1; j <= nz; j++) boundary.push([bounds.maxX, bounds.minZ + j * dz]);
  for (let i = nx - 1; i >= 0; i--) boundary.push([bounds.minX + i * dx, bounds.maxZ]);
  for (let j = nz - 1; j > 0; j--) boundary.push([bounds.minX, bounds.minZ + j * dz]);
  for (let i = 0; i <= boundary.length; i++) {
    const [x, z] = boundary[i % boundary.length], top = heightAt(x, z);
    for (let level = 0; level < 4; level++) {
      const yy = level === 0 ? top : -.8 - (level - 1) * 1.1;
      skirtData.positions.push(x, yy, z); skirtData.uvs.push(i / 45, level * .55);
      const tint = [1, .68, .48, .25][level]; skirtData.colours.push(tint, tint * .91, tint * .8);
    }
  }
  for (let i = 0; i < boundary.length; i++) for (let j = 0; j < 3; j++) {
    const a = i * 4 + j; skirtData.indices.push(a, a + 1, a + 4, a + 4, a + 1, a + 5);
  }
  const skirtMaterial = naturalMaterial({map: textures.soil, bumpMap: textures.soilBump, bumpScale: .08, vertexColors: true, roughness: 1, side: THREE.DoubleSide}, {snow: 0});
  const skirt = new THREE.Mesh(geometryFromData(skirtData), skirtMaterial); skirt.receiveShadow = true; group.add(skirt);

  // Low-density surrounding woodland carries the horizon beyond the interactive
  // clearing. Its first ring shares the exact boundary vertices, leaving no seam.
  function sceneryHeight(x, z) {
    const scale = Math.max(Math.abs(x) / 18, Math.abs(z) / 14);
    if (scale <= 1) return heightAt(x, z);
    const edgeY = heightAt(x / scale, z / scale);
    const distantY = .05 + noise(x * .071 + 41, z * .071 + 36) * 1.5 + .34 * Math.sin(x * .07) * Math.cos(z * .08);
    return mix(edgeY, distantY, smooth(1, 2.35, scale));
  }
  const ringData = {positions: [], uvs: [], indices: [], colours: []}, scales = [1, 1.2, 1.6, 2.3, 3.2, 4.5, 6.4, 8];
  for (let ring = 0; ring < scales.length; ring++) for (const [edgeX, edgeZ] of boundary) {
    const x = edgeX * scales[ring], z = edgeZ * scales[ring];
    ringData.positions.push(x, sceneryHeight(x, z), z); ringData.uvs.push((x + 18) / 36, (z + 14) / 28);
    if (ring === 0) {
      const id = Math.round((edgeZ + 14) / dz) * (nx + 1) + Math.round((edgeX + 18) / dx);
      c.setRGB(colours[id * 3], colours[id * 3 + 1], colours[id * 3 + 2]);
    } else {
      // Broad canopy occlusion belongs only to the surrounding soil. The
      // original clearing colors and all collider vertices stay untouched.
      const canopyShade = 1 - smooth(1.14, 2.85, scales[ring]) * (.27 + .12 * noise(x * .055 + 8, z * .055 + 63));
      c.copy(dirtColour).lerp(mossColour, mossAt(x, z) * .82)
        .multiplyScalar((.7 + .25 * noise(x * .12 + 21, z * .12 + 48)) * canopyShade);
    }
    ringData.colours.push(c.r, c.g, c.b);
  }
  for (let ring = 0; ring < scales.length - 1; ring++) for (let i = 0; i < boundary.length; i++) {
    const a = ring * boundary.length + i, b = ring * boundary.length + (i + 1) % boundary.length, cc = a + boundary.length, d = b + boundary.length;
    ringData.indices.push(a, b, cc, b, d, cc);
  }
  const surroundings = new THREE.Mesh(geometryFromData(ringData), groundMaterial); surroundings.name = 'Continuous distant forest floor'; surroundings.receiveShadow = true; group.add(surroundings);
  const sceneryRay = new THREE.Raycaster(), sceneryOrigin = new THREE.Vector3(), down = new THREE.Vector3(0, -1, 0);
  function scenerySurfaceAt(x, z) {
    sceneryOrigin.set(x, 40, z); sceneryRay.set(sceneryOrigin, down);
    const hit = sceneryRay.intersectObject(surroundings, false)[0];
    return hit ? hit.point.y : sceneryHeight(x, z);
  }
  // The camera can travel past the edge of the playable floor, where the surrounding scenery is the ground.
  function cameraHeightAt(x, z) {
    return x >= bounds.minX && x <= bounds.maxX && z >= bounds.minZ && z <= bounds.maxZ ? heightAt(x, z) : scenerySurfaceAt(x, z);
  }

  const barkMaterial = naturalMaterial({map: textures.bark, normalMap: textures.barkNormal, normalScale: new THREE.Vector2(.8, .8), roughnessMap: textures.barkARM,
    aoMap: textures.barkARM, aoMapIntensity: .65, roughness: .94, color: '#d8d4c5'});
  const rootMaterial = naturalMaterial({map: textures.root, normalMap: textures.rootNormal, normalScale: new THREE.Vector2(.55, .55), roughnessMap: textures.rootARM,
    aoMap: textures.rootARM, aoMapIntensity: .55, roughness: .96, color: '#b6af99'});
  const cutMaterial = naturalMaterial({map: textures.rings, roughness: .96, side: THREE.DoubleSide, color: '#bdb291'});
  const obstacles = [], collisionMeshes = [];
  const trunkSpecs = [
    {x: -14.5, z: -8.1, r: .92, h: 10.8, leanX: .42, leanZ: -.7},
    {x: 13.8, z: -9.6, r: 1.05, h: 12.2, leanX: -.5, leanZ: -.1},
    {x: -16.2, z: 5.7, r: .76, h: 9.3, leanX: .7, leanZ: .3},
    {x: 16.3, z: 7.2, r: .82, h: 10.4, leanX: -.5, leanZ: -.4},
    {x: -1.3, z: -12.5, r: .66, h: 11.8, leanX: -.45, leanZ: .3}
  ];
  const canopyLocations = [];
  function woodMesh(curve, radius, segments = 24, radial = 10, material = rootMaterial, solid = true) {
    if (material === barkMaterial) material = rootMaterial;
    const g = new THREE.TubeGeometry(curve, segments, radius, radial, false), pp = g.attributes.position;
    // Taper each tube toward its tip, retaining the curve's true 3D centerline.
    for (let i = 0; i <= segments; i++) {
      const point = curve.getPointAt(i / segments), taper = .98 - .76 * (i / segments) ** 1.25;
      for (let j = 0; j <= radial; j++) {
        const idx = i * (radial + 1) + j;
        pp.setXYZ(idx, point.x + (pp.getX(idx) - point.x) * taper, point.y + (pp.getY(idx) - point.y) * taper, point.z + (pp.getZ(idx) - point.z) * taper);
      }
    }
    g.computeVertexNormals(); const mesh = new THREE.Mesh(g, material); mesh.castShadow = true; mesh.receiveShadow = true; group.add(mesh);
    if (solid) collisionMeshes.push(mesh); return mesh;
  }
  for (const spec of trunkSpecs) {
    const y = heightAt(spec.x, spec.z), trunkGeometry = new THREE.CylinderGeometry(spec.r * .47, spec.r, spec.h, low ? 16 : 24, 10, false);
    const p = trunkGeometry.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const t = (p.getY(i) + spec.h / 2) / spec.h, xx = p.getX(i), zz = p.getZ(i), a = Math.atan2(zz, xx);
      const ridge = 1 + Math.sin(a * 7 + t * 2.1) * .025 + Math.cos(a * 13 - t * 1.6) * .018;
      p.setXYZ(i, xx * ridge + spec.leanX * t * t, p.getY(i), zz * ridge + spec.leanZ * t * t);
    }
    trunkGeometry.computeVertexNormals();
    const trunk = new THREE.Mesh(trunkGeometry, barkMaterial); trunk.position.set(spec.x, y + spec.h / 2 - .13, spec.z); trunk.castShadow = trunk.receiveShadow = true; group.add(trunk); collisionMeshes.push(trunk);
    obstacles.push({x: spec.x, z: spec.z, r: spec.r + .15, height: spec.h, type: 'tree'});
    for (let j = 0; j < 6; j++) {
      const a = j / 6 * TAU + rnd() * .36, length = 2.1 + rnd() * 2.1, endX = clamp(spec.x + Math.cos(a) * length, -17.8, 17.8), endZ = clamp(spec.z + Math.sin(a) * length, -13.8, 13.8);
      if (protectedGround(endX, endZ, .1)) continue;
      const points = [new THREE.Vector3(spec.x, y + spec.r * .29, spec.z)];
      for (let k = 1; k <= 4; k++) {
        const t = k / 4, x = mix(spec.x, endX, t) + Math.sin(t * Math.PI) * (rnd() - .5) * .38, z = mix(spec.z, endZ, t) + Math.sin(t * Math.PI) * (rnd() - .5) * .35;
        points.push(new THREE.Vector3(x, heightAt(x, z) + spec.r * .18 * (1 - t), z));
      }
      const curve = new THREE.CatmullRomCurve3(points), rootRadius = spec.r * (.26 + rnd() * .08);
      woodMesh(curve, rootRadius, low ? 12 : 20, 8);
      for (let k = 1; k <= 4; k++) {
        const q = curve.getPoint(k / 5); obstacles.push({x: q.x, z: q.z, r: rootRadius * (1 - k / 7) + .08, height: rootRadius * 1.6, type: 'root'});
      }
    }
    for (let j = 0; j < 4; j++) {
      const a = rnd() * TAU, branchY = y + spec.h * (.61 + j * .074), extent = 2.7 + rnd() * 1.8;
      const start = new THREE.Vector3(spec.x + spec.leanX * .5, branchY, spec.z + spec.leanZ * .5), end = new THREE.Vector3(start.x + Math.cos(a) * extent, branchY + 1.2 + rnd(), start.z + Math.sin(a) * extent);
      const mid = start.clone().lerp(end, .55); mid.y -= .15;
      woodMesh(new THREE.CatmullRomCurve3([start, mid, end]), spec.r * .22, 12, 6, barkMaterial, false);
      canopyLocations.push({x: end.x, y: end.y, z: end.z, r: 1.1 + rnd() * .65});
    }
    canopyLocations.push({x: spec.x + spec.leanX, y: y + spec.h - .2, z: spec.z + spec.leanZ, r: 1.7});
  }
  // A fallen log sits left of the clock. Cross-sections expose growth rings.
  {
    const points = [new THREE.Vector3(-11.9, heightAt(-11.9, 1.1) + .36, 1.1), new THREE.Vector3(-11.3, heightAt(-11.3, 3.5) + .41, 3.5), new THREE.Vector3(-10.5, heightAt(-10.5, 6.1) + .31, 6.1)];
    const curve = new THREE.CatmullRomCurve3(points), log = woodMesh(curve, .49, 24, 16, barkMaterial); log.name = 'Weathered fallen branch';
    for (const t of [0, 1]) {
      const p = curve.getPoint(t), tangent = curve.getTangent(t).normalize(), cap = new THREE.Mesh(new THREE.CircleGeometry(t === 0 ? .475 : .102, 32), cutMaterial);
      cap.position.copy(p).addScaledVector(tangent, t === 0 ? -.008 : .008); cap.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent); cap.receiveShadow = true; group.add(cap);
    }
    for (let i = 0; i < 8; i++) { const p = curve.getPoint(i / 7); obstacles.push({x: p.x, z: p.z, r: .51 * (1 - i / 11), height: .75, type: 'log'}); }
  }

  const fernMaterial = naturalMaterial({map: textures.leaf, color: '#647d35', vertexColors: true, roughness: .76, side: THREE.DoubleSide}, {wind: .075, snow: .8, wet: .75});
  const grassMaterial = naturalMaterial({map: textures.leaf, color: '#738044', roughness: .83, side: THREE.DoubleSide}, {wind: .095, snow: .7, wet: .7});
  const mossMaterial = naturalMaterial({color: '#748b3a', roughness: .95, side: THREE.DoubleSide}, {wind: .012, snow: .95});
  const litterMaterial = naturalMaterial({map: textures.leaf, color: '#a48246', roughness: .96, side: THREE.DoubleSide}, {snow: .95});
  const canopyMaterial = naturalMaterial({map: textures.leaf, color: '#6d8842', roughness: .81, side: THREE.DoubleSide}, {wind: .035, snow: .45, wet: .65});
  const instanceObject = new THREE.Object3D(), rotationY = new THREE.Quaternion();
  function instances(geometry, material, entries, name, shadow = true) {
    const mesh = new THREE.InstancedMesh(geometry, material, entries.length); mesh.name = name; mesh.castShadow = shadow; mesh.receiveShadow = true;
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i]; instanceObject.position.set(e.x, e.y ?? heightAt(e.x, e.z) + .018, e.z);
      if (e.normal) {
        instanceObject.quaternion.setFromUnitVectors(UP, e.normal); rotationY.setFromAxisAngle(UP, e.angle ?? 0); instanceObject.quaternion.multiply(rotationY);
      } else instanceObject.rotation.set(e.rx || 0, e.angle || 0, e.rz || 0);
      const s = e.scale ?? 1; instanceObject.scale.set(e.sx ?? s, e.sy ?? s, e.sz ?? s); instanceObject.updateMatrix(); mesh.setMatrixAt(i, instanceObject.matrix);
      if (e.colour) mesh.setColorAt(i, e.colour);
    }
    mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere(); group.add(mesh);
    if (material.userData.habitat?.wind) attachWindShadow(mesh, material.userData.habitat.wind);
    return mesh;
  }
  function freeGroundPoint(margin = .6) {
    for (let tries = 0; tries < 120; tries++) {
      const x = -17 + rnd() * 34, z = -13 + rnd() * 26;
      if (!protectedGround(x, z, margin) && !obstacles.some(o => Math.hypot(x - o.x, z - o.z) < o.r + .22)) return {x, z};
    }
    return {x: -9.6, z: 7.8};
  }
  const fernEntries = [], fernAnchors = [
    [-9.8, -5.9], [-12.4, -2.9], [-14.5, 9], [-7.7, 7.5], [11.2, -5.4], [13.6, 1.1], [12.7, 8.1], [5.1, -8.3], [-5.1, -9.4], [5.5, 7.7], [-2, 10.2], [10, 10.5]
  ];
  for (let i = 0; i < Math.floor(15 * density); i++) { const p = freeGroundPoint(1.15); fernAnchors.push([p.x, p.z]); }
  for (const [x, z] of fernAnchors) {
    const size = .57 + rnd() * .63, fronds = 5 + Math.floor(rnd() * 3);
    for (let j = 0; j < fronds; j++) fernEntries.push({x: x + (rnd() - .5) * .18, z: z + (rnd() - .5) * .18, angle: j / fronds * TAU + rnd() * .3, scale: size * (.8 + rnd() * .4), colour: new THREE.Color().setRGB(.74 + rnd() * .26, .78 + rnd() * .22, .64 + rnd() * .3)});
  }
  const ferns = instances(fernGeometry(), fernMaterial, fernEntries, 'Articulated geometric fern fronds');
  const grasses = [];
  for (let i = 0; i < Math.floor(570 * density); i++) {
    const p = freeGroundPoint(.3); if (rnd() > .4 + mossAt(p.x, p.z) * .65) continue;
    grasses.push({...p, angle: rnd() * TAU, scale: .55 + rnd() * .7, normal: normalAt(p.x, p.z), colour: new THREE.Color().setRGB(.65 + rnd() * .35, .72 + rnd() * .28, .6 + rnd() * .35)});
  }
  for (let i = 0; i < Math.floor(490 * density); i++) {
    const x = (rnd() - .5) * 73, z = (rnd() - .5) * 59;
    if (Math.abs(x) < 18 && Math.abs(z) < 14) continue;
    grasses.push({x, z, y: scenerySurfaceAt(x, z), angle: rnd() * TAU, scale: .8 + rnd() * .95, colour: new THREE.Color().setRGB(.68 + rnd() * .27, .72 + rnd() * .28, .55 + rnd() * .36)});
  }
  instances(grassGeometry(seed + 99), grassMaterial, grasses, 'Sedges and woodland grasses');
  const mossEntries = [];
  for (let i = 0; i < Math.floor(6500 * density); i++) {
    const x = -17.5 + rnd() * 35, z = -13.5 + rnd() * 27, patch = mossAt(x, z);
    if (patch < .23 || rnd() > patch || protectedGround(x, z, .1)) continue;
    mossEntries.push({x, z, y: heightAt(x, z) + .019, angle: rnd() * TAU, rx: -1.05 + rnd() * .55, scale: .045 + rnd() * .065, colour: new THREE.Color().setRGB(.54 + rnd() * .44, .6 + rnd() * .4, .45 + rnd() * .42)});
  }
  instances(leafGeometry(1, .33, .08, 3), mossMaterial, mossEntries, 'Individual moss shoots', false);
  const litter = [];
  for (let i = 0; i < Math.floor(470 * density); i++) {
    const p = freeGroundPoint(.14), size = .19 + rnd() * .52;
    litter.push({...p, y: heightAt(p.x, p.z) + .023, angle: rnd() * TAU, normal: normalAt(p.x, p.z), sx: size * (.8 + rnd() * .5), sy: size * (.65 + rnd()), sz: size,
      colour: new THREE.Color().setRGB(.59 + rnd() * .4, .53 + rnd() * .37, .38 + rnd() * .42)});
  }
  instances(leafGeometry(1, .58, .14, 6), litterMaterial, litter, 'Curled fallen leaves');
  const canopy = [], twigData = {positions: [], uvs: [], indices: []}, identity = new THREE.Matrix4();
  for (const anchor of canopyLocations) for (let twig = 0; twig < (low ? 3 : 4); twig++) {
    const a = rnd() * TAU, length = anchor.r * (.65 + rnd() * .52);
    const start = new THREE.Vector3(anchor.x, anchor.y, anchor.z);
    const end = new THREE.Vector3(anchor.x + Math.sin(a) * length, anchor.y + (.05 + rnd() * .34) * length, anchor.z + Math.cos(a) * length);
    const mid = start.clone().lerp(end, .5); mid.y += length * .12;
    const curve = new THREE.CatmullRomCurve3([start, mid, end]), twigGeometry = new THREE.TubeGeometry(curve, 7, .014, 4, false);
    appendGeometry(twigData, twigGeometry, identity); twigGeometry.dispose();
    for (let j = 0; j < 6; j++) for (const side of [-1, 1]) {
      const p = curve.getPoint((j + 1) / 7), size = (.26 + rnd() * .17) * (1 - j * .045);
      canopy.push({x: p.x, y: p.y + .012, z: p.z, angle: a + side * (1.04 + rnd() * .17), rx: -.12 + rnd() * .3, rz: side * -.1,
        scale: size, colour: new THREE.Color().setRGB(.62 + rnd() * .38, .75 + rnd() * .25, .49 + rnd() * .38)});
    }
  }
  const twigs = new THREE.Mesh(geometryFromData(twigData), rootMaterial); twigs.name = 'Leaf-bearing twigs'; twigs.castShadow = true; twigs.receiveShadow = true; group.add(twigs);
  instances(leafGeometry(1, .47, .09, 5), canopyMaterial, canopy, 'Light-filtering woodland canopy');

  // Beyond the clearing, instanced trunks and leaf clusters form a 360-degree
  // forest backdrop. They are visual context, outside the stone simulation bounds.
  const distantTrunks = [], distantCrowns = [];
  for (let i = 0; i < Math.floor(76 * density); i++) {
    const a = i * 2.399963 + rnd() * .4, r = 2.65 + rnd() * 1.45, x = Math.sin(a) * 18 * r, z = Math.cos(a) * 14 * r;
    const h = 8.5 + rnd() * 7.5, radius = .25 + rnd() * .4, y = scenerySurfaceAt(x, z);
    distantTrunks.push({x, z, y: y + h / 2 - .22, angle: rnd() * TAU, sx: radius, sy: h, sz: radius, colour: new THREE.Color().setRGB(.7 + rnd() * .22, .7 + rnd() * .22, .65 + rnd() * .23)});
    distantCrowns.push({x, z, y: y + h * .82, angle: rnd() * TAU, sx: 2.1 + rnd() * 1.3, sy: 1.5 + rnd(), sz: 2.1 + rnd() * 1.3, colour: new THREE.Color().setRGB(.62 + rnd() * .24, .71 + rnd() * .23, .52 + rnd() * .22)});
  }
  instances(new THREE.CylinderGeometry(.52, 1, 1, 7, 3), barkMaterial, distantTrunks, 'Distant forest trunks', false);
  const crownData = {positions: [], uvs: [], indices: []}, crownObject = new THREE.Object3D(), crownLeaf = leafGeometry(.68, .35, .045, 3);
  for (let i = 0; i < 44; i++) {
    const a = rnd() * TAU, r = Math.sqrt(rnd()); crownObject.position.set(Math.sin(a) * r, (rnd() - .5) * 1.7, Math.cos(a) * r);
    crownObject.rotation.set((rnd() - .5) * .8, rnd() * TAU, (rnd() - .5) * .5); crownObject.updateMatrix(); appendGeometry(crownData, crownLeaf, crownObject.matrix);
  }
  crownLeaf.dispose(); instances(geometryFromData(crownData), canopyMaterial, distantCrowns, 'Distant leafy crowns', false);

  // These tiny grains are substrate detail, far smaller than the clock's stones.
  const grainMaterial = naturalMaterial({color: '#c6b69a', roughness: 1});
  const grains = [];
  for (let i = 0; i < Math.floor(1500 * density); i++) {
    const x = -16.8 + rnd() * 33.6, z = -12.8 + rnd() * 25.6;
    if (protectedGround(x, z) && rnd() < .8) continue;
    const radius = .009 + rnd() * .023;
    grains.push({x, z, y: heightAt(x, z) + radius * .45, angle: rnd() * TAU, sx: radius, sy: radius * .64, sz: radius * (.8 + rnd() * .25), colour: new THREE.Color().setRGB(.5 + rnd() * .5, .5 + rnd() * .5, .48 + rnd() * .45)});
  }
  instances(new THREE.IcosahedronGeometry(1, 0), grainMaterial, grains, 'Fixed soil grains, below 0.035 radius', false);

  // A shallow opening is embedded in the same heightfield, with visible earthen walls.
  const nestInteriorMaterial = new THREE.MeshStandardMaterial({color: '#17130c', roughness: 1, side: THREE.DoubleSide}); materials.push(nestInteriorMaterial);
  const nest = new THREE.Mesh(new THREE.CylinderGeometry(.165, .10, .17, 24, 1, true), nestInteriorMaterial);
  nest.position.set(9, heightAt(9, 4) + .025, 4); nest.receiveShadow = true; group.add(nest);
  const opening = new THREE.Mesh(new THREE.CircleGeometry(.145, 24), nestInteriorMaterial); opening.rotation.x = -Math.PI / 2; opening.position.set(9, heightAt(9, 4) - .06, 4); group.add(opening);

  const stemMaterial = naturalMaterial({color: '#b5aa7f', roughness: .83});
  const mushroomMaterial = naturalMaterial({color: '#a0ad82', emissive: '#528375', emissiveIntensity: 0, roughness: .68}, {snow: .9, wet: .65});
  const caps = [], stems = [], dewEntries = [];
  for (const anchor of [[-10.8, -1.1], [10.9, -7.1], [-8.1, 8.5]]) for (let i = 0; i < 8; i++) {
    const x = anchor[0] + (rnd() - .5) * 1.1, z = anchor[1] + (rnd() - .5) * 1.1, size = .085 + rnd() * .11, h = size * (1.6 + rnd() * .8);
    stems.push({x, z, y: heightAt(x, z) + h / 2, sx: size * .19, sy: h, sz: size * .19, angle: rnd() * TAU});
    caps.push({x, z, y: heightAt(x, z) + h, sx: size, sy: size * .55, sz: size, angle: rnd() * TAU, colour: new THREE.Color().setRGB(.82 + rnd() * .18, .85 + rnd() * .15, .77 + rnd() * .21)});
  }
  instances(new THREE.CylinderGeometry(1, 1.2, 1, 6), stemMaterial, stems, 'Mushroom stems');
  const capGeometry = new THREE.LatheGeometry([new THREE.Vector2(0, .45), new THREE.Vector2(.3, .42), new THREE.Vector2(.7, .18), new THREE.Vector2(1, -.07), new THREE.Vector2(.8, -.19), new THREE.Vector2(.22, -.08), new THREE.Vector2(0, -.07)].reverse(), low ? 10 : 16);
  instances(capGeometry, mushroomMaterial, caps, 'Woodland mushroom caps');
  // Dew beads occupy permanent physical leaf locations. Their visibility varies smoothly.
  for (let i = 0; i < fernEntries.length; i += 2) {
    const e = fernEntries[i], s = e.scale, a = e.angle;
    dewEntries.push({x: e.x + Math.sin(a) * .61 * s, z: e.z + Math.cos(a) * .61 * s, y: heightAt(e.x, e.z) + .81 * s, scale: .016 + rnd() * .007});
  }
  const dewMaterial = new THREE.MeshPhysicalMaterial({color: '#d3e7df', roughness: .04, metalness: .03, transparent: true, opacity: 0, transmission: 0, clearcoat: 1, clearcoatRoughness: .04, depthWrite: false});
  materials.push(dewMaterial); const dew = instances(new THREE.SphereGeometry(1, 6, 4), dewMaterial, dewEntries, 'Morning dew', false);
  dew.renderOrder = 3;
  // An independent random stream preserves every existing terrain vertex,
  // obstacle, plant and grain while extending the woodland in all directions.
  const forest = createForestBackdrop(group, {seed: seed ^ 0x73e91a, quality,
    heightAt: scenerySurfaceAt, makeMaterial: naturalMaterial, textures, startView});
  const fernSeason = [new THREE.Color('#77934c'), new THREE.Color('#627c36'), new THREE.Color('#a18846'), new THREE.Color('#847447')];
  const canopySeason = [new THREE.Color('#8da952'), new THREE.Color('#67843e'), new THREE.Color('#ba7d33'), new THREE.Color('#807148')];
  const grassSeason = [new THREE.Color('#829654'), new THREE.Color('#738044'), new THREE.Color('#a5945e'), new THREE.Color('#9a9277')];
  const seasonalColor = (palette, season, material) => {
    const s = ((season % 4) + 4) % 4, i = Math.floor(s); material.color.copy(palette[i]).lerp(palette[(i + 1) % 4], s - i);
  };
  let elapsed = 0, wet = 0, snow = 0, dewAmount = 0, wind = .18, season = 1;
  function update(dt, env = {}) {
    const delta = clamp(Number.isFinite(dt) ? dt : 0, 0, .12); elapsed += delta;
    const follow = 1 - Math.exp(-delta * .7), quick = 1 - Math.exp(-delta * 2);
    wet += (clamp(Number(env.wetness) || 0) - wet) * follow;
    snow += (clamp(Number(env.snow) || 0) - snow) * follow;
    dewAmount += (clamp(Number(env.dew) || 0) - dewAmount) * quick;
    const desiredWind = typeof env.wind === 'number' ? env.wind : (env.wind?.isVector3 ? env.wind.length() * .4 : (env.wind?.strength ?? .18));
    wind += (clamp(desiredWind, 0, 2) - wind) * quick;
    if (Number.isFinite(env.season)) {
      let difference = (((env.season - season) % 4) + 6) % 4 - 2; season += difference * follow;
    }
    uniforms.uHabitatWet.value = wet; uniforms.uHabitatSnow.value = snow; uniforms.uHabitatTime.value = elapsed; uniforms.uHabitatWind.value = wind;
    seasonalColor(fernSeason, season, fernMaterial); seasonalColor(canopySeason, season, canopyMaterial); seasonalColor(grassSeason, season, grassMaterial);
    forest.update(season);
    const daylight = clamp(Number.isFinite(env.daylight) ? env.daylight : 1);
    mushroomMaterial.emissiveIntensity = (1 - daylight) ** 3 * .4;
    dewMaterial.opacity = dewAmount * .73; dew.visible = dewAmount > .001;
    return {wetness: wet, snow, season};
  }
  group.updateMatrixWorld(true);
  return {
    group, ground, heightAt, cameraHeightAt, normalAt, bounds, colliderData: {vertices: positions, indices}, collisionMeshes, obstacles, update, materials, textures, ready: textures.ready, forest,
    nest: {x: 9, z: 4, y: heightAt(9, 4)},
    statistics: {terrainVertices: heightField.length, terrainTriangles: indices.length / 3, fernFronds: fernEntries.length, mossShoots: mossEntries.length, canopyLeaves: canopy.length, litterLeaves: litter.length, fixedGrains: grains.length, forest: forest.statistics},
    dispose() {
      group.traverse(object => { if (object.geometry) object.geometry.dispose(); if (object.customDepthMaterial) object.customDepthMaterial.dispose(); });
      materials.forEach(material => material.dispose()); Object.values(textures).forEach(texture => texture.dispose()); scene.remove(group);
    }
  };
}
