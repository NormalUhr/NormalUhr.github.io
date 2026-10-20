/**
 * One persistent, bounded sand surface. Coordinates and heights are metres.
 * All tools edit this field; no tool owns a separate layer or drawing canvas.
 * This module deliberately has no renderer, clock, window or DOM dependency.
 */
export const SAND_BOUNDS = Object.freeze({ minX: -5.6, maxX: 5.6, minZ: -4.1, maxZ: 4.1 });
export const SAND_LIMITS = Object.freeze({ minHeight: -0.042, maxHeight: 0.032 });
const TAU = Math.PI * 2;
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const smooth01 = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };

function encodeBytes(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 24576) {
    binary += String.fromCharCode(...bytes.subarray(i, Math.min(i + 24576, bytes.length)));
  }
  return btoa(binary);
}

function decodeBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < bytes.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** A fixed allocation height field. Brushes only visit their local rectangle. */
export function createSandField({ width = 1024, height = 768 } = {}) {
  width = Number.isFinite(width) ? clamp(Math.round(width), 32, 1536) : 1024;
  height = Number.isFinite(height) ? clamp(Math.round(height), 32, 1536) : 768;
  const bounds = SAND_BOUNDS;
  const spanX = bounds.maxX - bounds.minX, spanZ = bounds.maxZ - bounds.minZ;
  const stepX = spanX / (width - 1), stepZ = spanZ / (height - 1);
  const heights = new Float32Array(width * height);
  const base = new Float32Array(width * height);
  const xs = new Float32Array(width), zs = new Float32Array(height);
  const bx = new Float32Array(width), bz = new Float32Array(height);
  let revision = 0, strokeCount = 0, cachedRevision = -1, cachedStats;
  let dirty = { x0: 0, z0: 0, x1: width - 1, z1: height - 1 };

  for (let x = 0; x < width; x++) {
    xs[x] = bounds.minX + x * stepX;
    bx[x] = Math.sin(xs[x] * 0.79) * 0.00045 + Math.cos(xs[x] * 2.1) * 0.00018;
  }
  for (let z = 0; z < height; z++) {
    zs[z] = bounds.minZ + z * stepZ;
    bz[z] = Math.cos(zs[z] * 0.91) * 0.00034 + Math.sin(zs[z] * 1.53) * 0.00013;
    for (let x = 0; x < width; x++) {
      const i = z * width + x;
      base[i] = bx[x] + bz[z];
      // A very shallow, settled preparation pattern, independent of the clock.
      // Smoothing removes it too. It is never regenerated during update/resize.
      heights[i] = base[i] + Math.cos((zs[z] + 0.022 * Math.sin(xs[x] * 0.56)) * TAU / 0.092) * 0.00038;
    }
  }

  function contains(x, z) {
    return Number.isFinite(x) && Number.isFinite(z) && x >= bounds.minX && x <= bounds.maxX && z >= bounds.minZ && z <= bounds.maxZ;
  }

  function sample(array, x, z) {
    if (!Number.isFinite(x) || !Number.isFinite(z)) return 0;
    const fx = clamp((x - bounds.minX) / stepX, 0, width - 1);
    const fz = clamp((z - bounds.minZ) / stepZ, 0, height - 1);
    const ix = Math.min(width - 2, Math.floor(fx)), iz = Math.min(height - 2, Math.floor(fz));
    const tx = fx - ix, tz = fz - iz, k = iz * width + ix;
    return (array[k] * (1 - tx) + array[k + 1] * tx) * (1 - tz) +
      (array[k + width] * (1 - tx) + array[k + width + 1] * tx) * tz;
  }

  function markDirty(x0, z0, x1, z1) {
    if (!dirty) dirty = { x0, z0, x1, z1 };
    else {
      dirty.x0 = Math.min(dirty.x0, x0); dirty.x1 = Math.max(dirty.x1, x1);
      dirty.z0 = Math.min(dirty.z0, z0); dirty.z1 = Math.max(dirty.z1, z1);
    }
  }

  function stroke(ax, az, bx, bz, options = {}) {
    // Coordinates far beyond the garden lose their local precision when
    // projected onto a short brush. Treat them as invalid input, not a wipe.
    if (![ax, az, bx, bz].every(v => Number.isFinite(v) && Math.abs(v) <= 10000)) return { changed: false, revision };
    const kind = options.kind || 'rake';
    if (!['rake', 'visitor', 'smooth'].includes(kind)) return { changed: false, revision };
    const defaultWidth = kind === 'rake' ? 0.16 : kind === 'visitor' ? 0.07 : 0.38;
    const defaultDepth = kind === 'rake' ? 0.009 : 0.012;
    const brushWidth = clamp(Number.isFinite(options.width) ? options.width : defaultWidth, 0.018, 1.6);
    const depth = clamp(Number.isFinite(options.depth) ? Math.abs(options.depth) : defaultDepth, 0, 0.024);
    const strength = clamp(Number.isFinite(options.strength) ? options.strength : 1, 0, 1);
    if (strength === 0) return { changed: false, revision };
    const dx = bx - ax, dz = bz - az, length = Math.hypot(dx, dz);
    if (!Number.isFinite(length)) return { changed: false, revision };
    // A stationary contact has a deterministic direction and makes a small dab.
    const ux = length > 1e-7 ? dx / length : 1, uz = length > 1e-7 ? dz / length : 0;
    const nx = -uz, nz = ux;
    const sigma = kind === 'rake' ? Math.max(0.0067, brushWidth * 0.046) : brushWidth * 0.155;
    // Integrating a tooth over a pixel footprint prevents sub-pixel tooth flicker.
    const pixelAcross = Math.hypot(nx * stepX, nz * stepZ);
    const filteredSigma = Math.sqrt(sigma * sigma + pixelAcross * pixelAcross / 12);
    const peakCorrection = sigma / filteredSigma;
    const outer = kind === 'smooth' ? brushWidth * 0.5 : kind === 'rake' ? brushWidth * 0.5 + filteredSigma * 3.0 : brushWidth * 0.71;
    const inner = kind === 'smooth' ? outer * 0.68 : kind === 'rake' ? brushWidth * 0.5 + filteredSigma * 0.9 : brushWidth * 0.42;
    const cap = kind === 'smooth' ? 0.032 : Math.max(filteredSigma * 1.25, pixelAcross * 0.7);
    const padding = outer + cap;
    const minX = Math.min(ax, bx) - padding, maxX = Math.max(ax, bx) + padding;
    const minZ = Math.min(az, bz) - padding, maxZ = Math.max(az, bz) + padding;
    if (maxX < bounds.minX || minX > bounds.maxX || maxZ < bounds.minZ || minZ > bounds.maxZ) return { changed: false, revision };
    const x0 = clamp(Math.floor((minX - bounds.minX) / stepX), 0, width - 1);
    const x1 = clamp(Math.ceil((maxX - bounds.minX) / stepX), 0, width - 1);
    const z0 = clamp(Math.floor((minZ - bounds.minZ) / stepZ), 0, height - 1);
    const z1 = clamp(Math.ceil((maxZ - bounds.minZ) / stepZ), 0, height - 1);
    const toothSpacing = brushWidth / 4;
    const invSigmaSq = 1 / (filteredSigma * filteredSigma);
    let changed = false;

    for (let z = z0; z <= z1; z++) {
      const pz = zs[z] - az, row = z * width;
      for (let x = x0; x <= x1; x++) {
        const px = xs[x] - ax;
        const along = px * ux + pz * uz;
        const beyond = Math.max(-along, along - length, 0);
        if (beyond >= cap) continue;
        const across = px * nx + pz * nz, absAcross = Math.abs(across);
        if (absAcross >= outer) continue;
        const edgeWeight = absAcross <= inner ? 1 : 1 - smooth01((absAcross - inner) / (outer - inner));
        const endWeight = beyond <= 0 ? 1 : 1 - smooth01(beyond / cap);
        const weight = strength * edgeWeight * endWeight;
        if (weight < 0.0005) continue;

        let relief = 0;
        if (kind !== 'smooth') {
          const toothCount = kind === 'rake' ? 5 : 1;
          for (let tooth = 0; tooth < toothCount; tooth++) {
            const d = across - (kind === 'rake' ? (tooth - 2) * toothSpacing : 0);
            if (Math.abs(d) > filteredSigma * 4.7) continue;
            const valley = Math.exp(-0.5 * d * d * invSigmaSq);
            const sideA = d - filteredSigma * 2.05, sideB = d + filteredSigma * 2.05;
            const ridges = Math.exp(-0.5 * sideA * sideA * invSigmaSq * 1.7) +
              Math.exp(-0.5 * sideB * sideB * invSigmaSq * 1.7);
            relief += depth * peakCorrection * (-valley + ridges * 0.24);
          }
          // Slightly changing tool pressure, with stable world coordinates.
          // There is no frame-dependent random noise or accumulated digging.
          relief *= 0.97 + 0.03 * Math.sin(xs[x] * 29.1 + zs[z] * 18.7);
        }
        const i = row + x, target = base[i] + relief;
        const value = clamp(heights[i] + (target - heights[i]) * weight, SAND_LIMITS.minHeight, SAND_LIMITS.maxHeight);
        if (Math.abs(value - heights[i]) > 1e-8) {
          heights[i] = value;
          changed = true;
        }
      }
    }
    strokeCount++;
    if (changed) {
      revision++;
      markDirty(x0, z0, x1, z1);
    }
    return { changed, revision };
  }

  function snapshot() {
    if (cachedRevision !== revision) {
      let minHeight = Infinity, maxHeight = -Infinity;
      let sum = 0, energy = 0, invalid = 0;
      for (let i = 0; i < heights.length; i++) {
        const h = heights[i];
        if (!Number.isFinite(h)) { invalid++; continue; }
        minHeight = Math.min(minHeight, h); maxHeight = Math.max(maxHeight, h);
        sum += h; energy += Math.abs(h - base[i]);
      }
      cachedStats = { minHeight, maxHeight, meanHeight: sum / heights.length, meanRelief: energy / heights.length, invalid };
      cachedRevision = revision;
    }
    return {
      revision, strokeCount, width, height, samples: heights.length,
      bufferBytes: heights.byteLength + base.byteLength + xs.byteLength + zs.byteLength + bx.byteLength + bz.byteLength,
      dirty: dirty ? { ...dirty } : null, ...cachedStats,
    };
  }

  function exportState() {
    const quantum = 0.00001;
    const bytes = new Uint8Array(heights.length * 2), view = new DataView(bytes.buffer);
    for (let i = 0; i < heights.length; i++) view.setInt16(i * 2, Math.round(heights[i] / quantum), true);
    return { type: 'karesansui-sand', version: 1, width, height, bounds: { ...bounds },
      encoding: 's16le-base64', quantum, heights: encodeBytes(bytes), revision, strokeCount };
  }

  function importState(data) {
    if (!data || data.type !== 'karesansui-sand' || data.version !== 1 || data.encoding !== 's16le-base64') return false;
    const sw = data.width, sh = data.height;
    if (!Number.isInteger(sw) || !Number.isInteger(sh) || sw < 32 || sh < 32 || sw > 1536 || sh > 1536) return false;
    if (typeof data.heights !== 'string' || data.heights.length > Math.ceil(sw * sh * 2 / 3) * 4 + 8) return false;
    if (data.quantum !== 0.00001 || !data.bounds || Object.keys(bounds).some(k => data.bounds[k] !== bounds[k])) return false;
    let bytes;
    try { bytes = decodeBytes(data.heights); } catch { return false; }
    if (bytes.byteLength !== sw * sh * 2) return false;
    const view = new DataView(bytes.buffer), source = new Float32Array(sw * sh);
    for (let i = 0; i < source.length; i++) {
      const h = view.getInt16(i * 2, true) * data.quantum;
      if (h < SAND_LIMITS.minHeight - 0.00001 || h > SAND_LIMITS.maxHeight + 0.00001) return false;
      source[i] = h;
    }
    if (sw === width && sh === height) heights.set(source);
    else {
      // Restoring on another screen quality keeps the same world-space surface.
      for (let z = 0; z < height; z++) {
        const fz = z / (height - 1) * (sh - 1), iz = Math.min(sh - 2, Math.floor(fz)), tz = fz - iz;
        for (let x = 0; x < width; x++) {
          const fx = x / (width - 1) * (sw - 1), ix = Math.min(sw - 2, Math.floor(fx)), tx = fx - ix, k = iz * sw + ix;
          heights[z * width + x] = (source[k] * (1 - tx) + source[k + 1] * tx) * (1 - tz) +
            (source[k + sw] * (1 - tx) + source[k + sw + 1] * tx) * tz;
        }
      }
    }
    // A restore is a deliberate state operation, never an incidental resize.
    revision = Math.max(revision + 1, Number.isSafeInteger(data.revision) && data.revision >= 0 ? data.revision : 0);
    strokeCount = Number.isSafeInteger(data.strokeCount) && data.strokeCount >= 0 ? data.strokeCount : 0;
    cachedRevision = -1;
    markDirty(0, 0, width - 1, height - 1);
    return true;
  }

  return {
    get revision() { return revision; },
    width, height, bounds, stepX, stepZ, heights, base,
    contains, heightAt: (x, z) => sample(heights, x, z), baseAt: (x, z) => sample(base, x, z),
    stroke, snapshot, exportState, importState,
    consumeDirty() { const result = dirty; dirty = null; return result; },
  };
}
