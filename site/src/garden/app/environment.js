import * as THREE from 'three';
import {environmentGzip, environmentMetadata} from './environment_data.js';
import {createLiveEnvironment} from './environment_source.js';

let preparation;
let preparedData = null;

/**
 * Start at the beginning of boot, alongside physics and material preparation.
 * This performs CPU decompression only; no renderer or WebGL context is needed.
 * null deliberately selects the original PMREM path on older browsers.
 */
export function prepareEnvironment() {
  if (!preparation) preparation = (async () => {
    if (typeof DecompressionStream !== 'function' || !environmentGzip) return null;
    const packed = atob(environmentGzip);
    const compressed = new Uint8Array(packed.length);
    for (let i = 0; i < packed.length; i++) compressed[i] = packed.charCodeAt(i);
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
    const buffer = await new Response(stream).arrayBuffer();
    const {width, height, rawBytes, formatVersion} = environmentMetadata;
    if (formatVersion !== 2 || buffer.byteLength !== rawBytes || rawBytes !== width * height * 8) return null;
    // Undo the baker's byte planes and integer gradient predictor. Uint16Array
    // wrapping reconstructs the exact original RGBA16F bit pattern; there is no
    // float arithmetic, image codec, precision reduction, or gamma conversion.
    const bytes = new Uint8Array(buffer), pixels = width * height, stride = width * 4;
    const data = new Uint16Array(pixels * 4);
    for (let channel = 0; channel < 4; channel++) {
      const lo = channel * 2 * pixels, hi = lo + pixels;
      for (let p = 0; p < pixels; p++) {
        const i = p * 4 + channel, x = p % width, y = Math.floor(p / width);
        const residual = bytes[lo + p] | bytes[hi + p] << 8;
        data[i] = residual + (x ? data[i - 4] : 0) + (y ? data[i - stride] : 0)
          - (x && y ? data[i - stride - 4] : 0);
      }
    }
    preparedData = {data, width, height};
    return preparedData;
  })().catch(() => null);
  return preparation;
}

/** CubeUV data is already prefiltered; Three must not generate another PMREM. */
export function createEnvironment(renderer, prepared = preparedData) {
  if (!prepared || !(prepared.data instanceof Uint16Array)
      || prepared.width !== 384 || prepared.height !== 512
      || prepared.data.length !== prepared.width * prepared.height * 4) {
    return createLiveEnvironment(renderer);
  }
  const texture = new THREE.DataTexture(prepared.data, prepared.width, prepared.height,
    THREE.RGBAFormat, THREE.HalfFloatType, THREE.CubeUVReflectionMapping,
    THREE.ClampToEdgeWrapping, THREE.ClampToEdgeWrapping,
    THREE.LinearFilter, THREE.LinearFilter, 1, THREE.LinearSRGBColorSpace);
  texture.name = 'PMREM.cubeUv';
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.unpackAlignment = 4;
  texture.needsUpdate = true;
  return {texture, dispose: () => texture.dispose()};
}
