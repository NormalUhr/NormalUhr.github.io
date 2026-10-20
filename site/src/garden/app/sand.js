import * as THREE from 'three';
import { createSandField, SAND_BOUNDS, SAND_LIMITS } from './sand_field.js';

const clamp01 = x => Math.min(1, Math.max(0, x));
const smooth01 = x => { x = clamp01(x); return x * x * (3 - 2 * x); };
const DRY_SAND = new THREE.Color('#d8ceb8');
const WET_SAND = new THREE.Color('#a9a18b');

/** A tiny repeatable mineral-grain texture; there are no external requests. */
function makeGrainTexture() {
  const size = 256, data = new Uint8Array(size * size * 4);
  let seed = 0x6cbf29a;
  const random = () => {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const r = random(), mineral = random();
      const wave = Math.sin(x / size * Math.PI * 8) * Math.cos(y / size * Math.PI * 6) * 3.5;
      let value = 228 + (r - 0.5) * 26 + wave;
      if (mineral > 0.985) value -= 35;
      else if (mineral < 0.035) value += 16;
      data[i] = Math.min(255, value + 4);
      data[i + 1] = Math.min(255, value + 2);
      data[i + 2] = Math.min(255, value - 3);
      data[i + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.repeat.set(11.2 / 0.46, 8.2 / 0.46);
  texture.needsUpdate = true;
  texture.name = '沙粒与石英';
  return texture;
}

const FIELD_VERTEX = `
uniform sampler2D uSandField;
uniform vec2 uSandTexel;
varying vec2 vSandUv;
vec2 sandTextureUv(vec2 worldUv) {
  return worldUv * (vec2(1.0) - uSandTexel) + uSandTexel * 0.5;
}
`;

/** Persistent deforming ground, shared by the clock, visitors and all workers. */
export function createSand(scene, { quality = 'high' } = {}) {
  // Low quality reduces geometry; keeping the same field resolution means fine
  // teeth, state persistence and interaction never depend on the viewport size.
  const field = createSandField({ width: 1024, height: 768 });
  const { width, height, bounds, stepX, stepZ, heights, base } = field;
  const pixels = new Uint16Array(width * height * 4);
  // The mesh has fewer samples than the material. Filtering vertex displacement
  // to its spatial scale prevents a regular zigzag at narrow tooth boundaries.
  // The full-resolution field still owns contact heights and visible normals.
  const geometryRows = new Float32Array(width * height);
  const texture = new THREE.DataTexture(pixels, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.name = '统一沙面高度与坡度';
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = true;
  texture.anisotropy = quality === 'low' || quality === 'mobile' ? 2 : 4;
  texture.flipY = false;
  texture.unpackAlignment = 1;
  const uniforms = {
    uSandField: { value: texture },
    uSandTexel: { value: new THREE.Vector2(1 / width, 1 / height) },
    uSandNight: { value: 0 },
  };
  const grain = makeGrainTexture();
  const material = new THREE.MeshStandardMaterial({
    color: DRY_SAND, map: grain, bumpMap: grain, bumpScale: 0.0009,
    roughness: 0.94, metalness: 0, envMapIntensity: 0.72,
  });
  material.name = '有浅槽与砂脊的庭院砂';

  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = FIELD_VERTEX + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <beginnormal_vertex>', `
      #include <beginnormal_vertex>
      vec4 sandVertexSample = texture2D(uSandField, sandTextureUv(uv));
      objectNormal = normalize(vec3(-sandVertexSample.g, 1.0, -sandVertexSample.b));
    `);
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vSandUv = sandTextureUv(uv);
      transformed.y += sandVertexSample.r;
    `);
    shader.fragmentShader = `
      uniform sampler2D uSandField;
      uniform mat3 normalMatrix;
      uniform float uSandNight;
      varying vec2 vSandUv;
    ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>', `
      #include <map_fragment>
      vec4 sandSurface = texture2D(uSandField, vSandUv);
      // The small teeth are below the shadow map's resolution. The alpha
      // channel carries local cavity/ridge response derived from actual height.
      // Preparing that response before mip filtering lets a distant pixel see
      // the average fine shadow, avoiding black moire rings as the view turns.
      // All physical marks share it; there is no separate painted clock layer.
      float sandOcclusion = mix(0.34, 0.62, uSandNight);
      diffuseColor.rgb *= 1.0 + sandSurface.a * sandOcclusion;
    `);
    shader.fragmentShader = shader.fragmentShader.replace('#include <normal_fragment_begin>', `
      #include <normal_fragment_begin>
      normal = normalize(normalMatrix * vec3(-sandSurface.g, 1.0, -sandSurface.b));
      nonPerturbedNormal = normal;
    `);
  };
  material.customProgramCacheKey = () => 'karesansui-persistent-sand-v2';

  const low = quality === 'low' || quality === 'mobile';
  const geometry = new THREE.PlaneGeometry(11.2, 8.2, low ? 224 : 384, low ? 164 : 280);
  geometry.rotateX(-Math.PI / 2);
  const uv = geometry.attributes.uv;
  // Field rows increase with world z, unlike PlaneGeometry's default v axis.
  for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
  geometry.boundingBox = new THREE.Box3(new THREE.Vector3(bounds.minX, SAND_LIMITS.minHeight, bounds.minZ),
    new THREE.Vector3(bounds.maxX, SAND_LIMITS.maxHeight, bounds.maxZ));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Math.hypot(5.6, 4.1, 0.05));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = '可以留下痕迹的沙面';
  mesh.receiveShadow = true;
  mesh.castShadow = true;
  mesh.userData.sandSurface = true;

  // Shadow passes use exactly the same vertical deformation as the visible mesh.
  // MeshDepthMaterial is used by directional/spot lights; distance by point lamps.
  const depthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  const distanceMaterial = new THREE.MeshDistanceMaterial();
  for (const shadowMaterial of [depthMaterial, distanceMaterial]) {
    shadowMaterial.onBeforeCompile = shader => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = FIELD_VERTEX + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
        #include <begin_vertex>
        transformed.y += texture2D(uSandField, sandTextureUv(uv)).r;
      `);
    };
    shadowMaterial.customProgramCacheKey = () => 'karesansui-sand-shadow-v1';
  }
  mesh.customDepthMaterial = depthMaterial;
  mesh.customDistanceMaterial = distanceMaterial;
  scene.add(mesh);

  let gpuMipSamples = 0;
  for (let w = width, h = height; w > 1 || h > 1;) {
    w = Math.max(1, Math.floor(w / 2)); h = Math.max(1, Math.floor(h / 2)); gpuMipSamples += w * h;
  }
  const gpuMipBufferBytes = gpuMipSamples * 4 * Uint16Array.BYTES_PER_ELEMENT;
  let gpuCommits = 0, lastUpdatedTexels = 0, wetness = 0, nightAdaptation = 0, lightInitialized = false, disposed = false;
  const toHalf = THREE.DataUtils.toHalfFloat;
  function flush() {
    const dirty = field.consumeDirty();
    if (!dirty) return false;
    // Extend for the derivative and the five-tap geometric reconstruction.
    const x0 = Math.max(0, dirty.x0 - 2), x1 = Math.min(width - 1, dirty.x1 + 2);
    const z0 = Math.max(0, dirty.z0 - 2), z1 = Math.min(height - 1, dirty.z1 + 2);
    for (let z = Math.max(0, z0 - 2); z <= Math.min(height - 1, z1 + 2); z++) {
      const row = z * width;
      for (let x = x0; x <= x1; x++) {
        geometryRows[row + x] = (heights[row + Math.max(0, x - 2)] +
          4 * heights[row + Math.max(0, x - 1)] + 6 * heights[row + x] +
          4 * heights[row + Math.min(width - 1, x + 1)] + heights[row + Math.min(width - 1, x + 2)]) / 16;
      }
    }
    for (let z = z0; z <= z1; z++) {
      const zPrevious = Math.max(z - 1, 0), zNext = Math.min(z + 1, height - 1);
      const rowFarTop = Math.max(0, z - 2) * width, rowFarBottom = Math.min(height - 1, z + 2) * width;
      const invZ = 1 / ((zNext - zPrevious) * stepZ * 4);
      for (let x = x0; x <= x1; x++) {
        const xPrevious = Math.max(x - 1, 0), xNext = Math.min(x + 1, width - 1);
        const i = z * width + x, k = i * 4;
        // A Sobel stencil keeps the normal of a narrow, diagonal tooth smooth.
        // The height itself remains untouched, including overlap and crossings.
        const topLeft = heights[zPrevious * width + xPrevious], topRight = heights[zPrevious * width + xNext];
        const bottomLeft = heights[zNext * width + xPrevious], bottomRight = heights[zNext * width + xNext];
        const slopeX = (topRight + 2 * heights[z * width + xNext] + bottomRight -
          topLeft - 2 * heights[z * width + xPrevious] - bottomLeft) / ((xNext - xPrevious) * stepX * 4);
        const slopeZ = (bottomLeft + 2 * heights[zNext * width + x] + bottomRight -
          topLeft - 2 * heights[zPrevious * width + x] - topRight) * invZ;
        const geometricHeight = (geometryRows[rowFarTop + x] + 4 * geometryRows[zPrevious * width + x] +
          6 * geometryRows[i] + 4 * geometryRows[zNext * width + x] + geometryRows[rowFarBottom + x]) / 16;
        pixels[k] = toHalf(geometricHeight);
        pixels[k + 1] = toHalf(slopeX);
        pixels[k + 2] = toHalf(slopeZ);
        // Sub-millimetre settled grain is left alone. Deeper cavities receive
        // local occlusion; raised grains expose a little more reflected light.
        // Store this signed response before mip generation: averaging raw
        // heights and then taking a nonlinear shadow would erase distant teeth.
        const relief = heights[i] - base[i];
        const cavity = smooth01((-relief - 0.00045) / (0.0065 - 0.00045));
        const ridge = smooth01((relief - 0.00045) / (0.0032 - 0.00045));
        pixels[k + 3] = toHalf(-cavity + ridge * 0.12);
      }
    }
    lastUpdatedTexels = (x1 - x0 + 1) * (z1 - z0 + 1);
    // Three r180 supports RGBA DataTexture partial updates. Short local strokes
    // send only touched row spans. Large initialization/import edits upload once.
    // Do not clear pending ranges: more than one update can precede a render.
    const fullUpload = lastUpdatedTexels > width * height * 0.2 || gpuCommits === 0;
    if (fullUpload) {
      texture.clearUpdateRanges();
      texture.userData.fullUploadPending = true;
    } else if (!texture.userData.fullUploadPending) {
      for (let z = z0; z <= z1; z++) texture.addUpdateRange((z * width + x0) * 4, (x1 - x0 + 1) * 4);
    }
    texture.needsUpdate = true;
    gpuCommits++;
    return true;
  }
  // The initial upload has no ranges. onUpdate runs only after the renderer has
  // consumed the pending texture data; subsequent strokes can then be partial.
  texture.onUpdate = () => { texture.userData.fullUploadPending = false; };
  flush();

  function update(dt = 0, env = {}) {
    if (disposed) return;
    flush();
    const targetWet = clamp01(Number.isFinite(env.wetness) ? env.wetness : Number.isFinite(env.wet) ? env.wet : 0);
    const elapsed = Number.isFinite(dt) ? Math.max(0, Math.min(dt, 0.2)) : 0;
    const targetNight = Number.isFinite(env.daylight) ? 1 - clamp01(env.daylight)
      : Number.isFinite(env.night) ? clamp01(env.night) : 0;
    // Start at the actual local light level, then follow it continuously. Time
    // previews and sunrise/sunset cannot switch groove colours in one frame.
    if (!lightInitialized) { nightAdaptation = targetNight; lightInitialized = true; }
    else nightAdaptation += (targetNight - nightAdaptation) * (1 - Math.exp(-elapsed * 1.8));
    uniforms.uSandNight.value = nightAdaptation;
    wetness += (targetWet - wetness) * (1 - Math.exp(-elapsed * 0.65));
    material.color.copy(DRY_SAND).lerp(WET_SAND, wetness);
    material.roughness = 0.94 - wetness * 0.21;
    material.bumpScale = 0.0009 - wetness * 0.00026;
  }

  return {
    mesh, bounds: SAND_BOUNDS, texture, field,
    stroke: field.stroke, update,
    heightAt: field.heightAt, contains: field.contains,
    snapshot() { return { ...field.snapshot(), gpuCommits, lastUpdatedTexels, wetness, nightAdaptation,
      gpuBufferBytes: pixels.byteLength + gpuMipBufferBytes, gpuMipBufferBytes,
      reconstructionBufferBytes: geometryRows.byteLength, geometryVertices: geometry.attributes.position.count,
      geometryTriangles: geometry.index.count / 3, pendingUploadRanges: texture.updateRanges.length }; },
    exportState: field.exportState,
    importState: field.importState,
    dispose() {
      if (disposed) return;
      disposed = true;
      scene.remove(mesh);
      geometry.dispose(); material.dispose(); depthMaterial.dispose(); distanceMaterial.dispose();
      texture.dispose(); grain.dispose();
    },
  };
}
