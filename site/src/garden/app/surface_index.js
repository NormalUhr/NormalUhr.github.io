/**
 * A conservative broad phase over the persistent sand samples. A true result
 * proves the whole foot box is clear. footClear additionally checks a
 * continuous foot disc against all vertices which could affect its bilinear
 * samples. It returns null when the field or boundary needs a caller fallback.
 * No samples, clock coverage or routes are changed by this index.
 *
 * Call invalidate(a,b,width,kind) immediately AFTER each known sand.stroke.
 * Unreported revision changes (including importState) force a complete rebuild.
 */
const TILE = 8;
const DANGER = -.00140;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class SurfaceIndex {
  constructor(sand) {
    this.sand = sand;
    this.field = null;
    this.seenRevision = null;
    this.fullDirty = true;
    this.prefixDirty = true;
    this.dirtyList = [];
    this.dirtyRows = [];
    this.stats = { queries: 0, clear: 0, footQueries: 0, discQueries: 0, discRows: 0, fullBuilds: 0,
      localBlocks: 0, rowBuilds: 0, localRows: 0, prefixBuilds: 0, sampledPixels: 0, externalChanges: 0 };
    this.attach();
  }
  attach() {
    const f = this.sand?.field || this.sand;
    if (!f || !Number.isInteger(f.width) || !Number.isInteger(f.height) || f.width < 2 || f.height < 2 || f.width > 65535 ||
      !f.heights || !f.base || f.heights.length !== f.width * f.height || f.base.length !== f.heights.length ||
      !f.bounds || !['minX', 'maxX', 'minZ', 'maxZ'].every(k => Number.isFinite(f.bounds[k])) ||
      f.bounds.maxX <= f.bounds.minX || f.bounds.maxZ <= f.bounds.minZ) {
      this.field = null; return false;
    }
    this.field = f; this.width = f.width; this.height = f.height;
    this.heights = f.heights; this.base = f.base; this.bounds = f.bounds;
    this.stepX = (f.bounds.maxX - f.bounds.minX) / (f.width - 1);
    this.stepZ = (f.bounds.maxZ - f.bounds.minZ) / (f.height - 1);
    this.pixelDiagonal = Math.hypot(this.stepX, this.stepZ);
    this.cols = Math.ceil(f.width / TILE); this.rows = Math.ceil(f.height / TILE);
    this.hazard = new Uint8Array(this.cols * this.rows);
    this.dirty = new Uint8Array(this.hazard.length);
    this.prefix = new Uint32Array((this.cols + 1) * (this.rows + 1));
    // The height field caps each dimension at 1536; a row count fits in Uint16.
    this.rowPrefix = new Uint16Array((this.width + 1) * this.height);
    this.rowDirty = new Uint8Array(this.height);
    this.dirtyList.length = 0; this.fullDirty = this.prefixDirty = true;
    this.dirtyRows.length = 0;
    this.seenRevision = this.readRevision();
    return true;
  }
  readRevision() {
    const value = this.field?.revision ?? this.sand?.revision;
    return Number.isSafeInteger(value) && value >= 0 ? value : null;
  }
  current() {
    const f = this.sand?.field || this.sand;
    if (!this.field || f !== this.field || f.width !== this.width || f.height !== this.height ||
      f.heights !== this.heights || f.base !== this.base) if (!this.attach()) return null;
    return this.readRevision();
  }
  allDirty(revision) {
    this.fullDirty = true; this.prefixDirty = true;
    this.dirtyList.length = 0; if (this.dirty) this.dirty.fill(0);
    this.dirtyRows.length = 0; if (this.rowDirty) this.rowDirty.fill(0);
    this.seenRevision = revision;
  }
  invalidate(a, b, width, kind = 'rake') {
    const revision = this.current();
    if (revision === null) { this.allDirty(null); return; }
    if (!a || !b || ![a.x, a.z, b.x, b.z].every(Number.isFinite) || !['rake', 'visitor', 'smooth'].includes(kind)) {
      this.allDirty(revision); return;
    }
    if (this.fullDirty) { this.seenRevision = revision; return; }
    // A no-op stroke does not increment the height-field revision. More than
    // one unseen change cannot safely be attributed to this brush rectangle.
    if (revision === this.seenRevision) return;
    if (revision !== this.seenRevision + 1) {
      this.stats.externalChanges++; this.allDirty(revision); return;
    }
    this.seenRevision = revision;
    const w = clamp(Number.isFinite(width) ? width : kind === 'rake' ? .16 : kind === 'visitor' ? .07 : .38, .018, 1.6);
    // Bound the exact brush support for every stroke orientation, including
    // anti-aliased tooth width on low-resolution fields and the rounded ends.
    const pixel = Math.max(this.stepX, this.stepZ);
    const sigma = kind === 'rake' ? Math.max(.0067, w * .046) : w * .155;
    const filtered = Math.sqrt(sigma * sigma + pixel * pixel / 12);
    const outer = kind === 'smooth' ? w * .5 : kind === 'rake' ? w * .5 + filtered * 3 : w * .71;
    const cap = kind === 'smooth' ? .032 : Math.max(filtered * 1.25, pixel * .7);
    const padding = outer + cap + pixel;
    const { minX, maxX, minZ, maxZ } = this.bounds;
    const x0 = Math.min(a.x, b.x) - padding, x1 = Math.max(a.x, b.x) + padding;
    const z0 = Math.min(a.z, b.z) - padding, z1 = Math.max(a.z, b.z) + padding;
    if (x1 < minX || x0 > maxX || z1 < minZ || z0 > maxZ) return;
    const left = Math.floor(clamp(Math.floor((x0 - minX) / this.stepX), 0, this.width - 1) / TILE);
    const right = Math.floor(clamp(Math.ceil((x1 - minX) / this.stepX), 0, this.width - 1) / TILE);
    const top = Math.floor(clamp(Math.floor((z0 - minZ) / this.stepZ), 0, this.height - 1) / TILE);
    const bottom = Math.floor(clamp(Math.ceil((z1 - minZ) / this.stepZ), 0, this.height - 1) / TILE);
    for (let row = top; row <= bottom; row++) for (let col = left; col <= right; col++) {
      const i = row * this.cols + col;
      if (!this.dirty[i]) { this.dirty[i] = 1; this.dirtyList.push(i); }
    }
    for (let row = top * TILE; row < Math.min(this.height, (bottom + 1) * TILE); row++) {
      if (!this.rowDirty[row]) { this.rowDirty[row] = 1; this.dirtyRows.push(row); }
    }
  }
  row(index) {
    const source = index * this.width, dest = index * (this.width + 1);
    let total = 0;
    for (let x = 0; x < this.width; x++) {
      const relief = this.heights[source + x] - this.base[source + x];
      if (!Number.isFinite(relief) || relief <= DANGER + 1e-9) total++;
      this.rowPrefix[dest + x + 1] = total;
    }
    this.stats.sampledPixels += this.width; this.stats.rowBuilds++;
  }
  block(index) {
    const x0 = index % this.cols * TILE, z0 = Math.floor(index / this.cols) * TILE;
    const x1 = Math.min(this.width, x0 + TILE), z1 = Math.min(this.height, z0 + TILE);
    let hazard = 0;
    for (let z = z0; z < z1; z++) {
      const start = z * (this.width + 1);
      if (this.rowPrefix[start + x1] !== this.rowPrefix[start + x0]) { hazard = 1; break; }
    }
    if (hazard !== this.hazard[index]) { this.hazard[index] = hazard; this.prefixDirty = true; }
  }
  flush(revision) {
    if (revision !== this.seenRevision) {
      this.stats.externalChanges++; this.allDirty(revision);
    }
    if (this.fullDirty) {
      for (let row = 0; row < this.height; row++) this.row(row);
      for (let i = 0; i < this.hazard.length; i++) this.block(i);
      this.fullDirty = false; this.prefixDirty = true; this.stats.fullBuilds++;
      this.dirty.fill(0); this.dirtyList.length = 0;
      this.rowDirty.fill(0); this.dirtyRows.length = 0;
    } else if (this.dirtyList.length) {
      for (const row of this.dirtyRows) { this.row(row); this.rowDirty[row] = 0; }
      this.stats.localRows += this.dirtyRows.length; this.dirtyRows.length = 0;
      for (const i of this.dirtyList) { this.block(i); this.dirty[i] = 0; }
      this.stats.localBlocks += this.dirtyList.length; this.dirtyList.length = 0;
    }
    if (this.prefixDirty) {
      const stride = this.cols + 1;
      for (let row = 0; row < this.rows; row++) {
        let total = 0;
        const source = row * this.cols, above = row * stride, dest = above + stride;
        for (let col = 0; col < this.cols; col++) {
          total += this.hazard[source + col];
          this.prefix[dest + col + 1] = this.prefix[above + col + 1] + total;
        }
      }
      this.prefixDirty = false; this.stats.prefixBuilds++;
    }
  }
  clearBox(x, z, pad = .135) {
    this.stats.queries++;
    if (![x, z, pad].every(Number.isFinite) || pad < 0) return false;
    const revision = this.current(); if (revision === null) return false;
    // The small expansion also contains the original exact test's 1.05-radius
    // sampling tolerance. Pixel neighbours below cover bilinear interpolation.
    const radius = pad * 1.03;
    const { minX, maxX, minZ, maxZ } = this.bounds;
    if (x - radius < minX || x + radius > maxX || z - radius < minZ || z + radius > maxZ) {
      // heightAt clamps at an edge, whereas the existing analytical base does
      // not. Leave these uncommon boxes to the exact world test.
      return false;
    }
    this.flush(revision);
    const left = Math.floor(clamp(Math.floor((x - radius - minX) / this.stepX) - 1, 0, this.width - 1) / TILE);
    const right = Math.floor(clamp(Math.floor((x + radius - minX) / this.stepX) + 1, 0, this.width - 1) / TILE);
    const top = Math.floor(clamp(Math.floor((z - radius - minZ) / this.stepZ) - 1, 0, this.height - 1) / TILE);
    const bottom = Math.floor(clamp(Math.floor((z + radius - minZ) / this.stepZ) + 1, 0, this.height - 1) / TILE);
    const stride = this.cols + 1, a = top * stride + left, b = top * stride + right + 1;
    const c = (bottom + 1) * stride + left, d = (bottom + 1) * stride + right + 1;
    const clear = this.prefix[d] - this.prefix[b] - this.prefix[c] + this.prefix[a] === 0;
    if (clear) this.stats.clear++;
    return clear;
  }
  footClear(x, z, pad = .135) {
    this.stats.footQueries++;
    if (this.clearBox(x, z, pad)) return true;
    if (![x, z, pad].every(Number.isFinite) || pad < 0 || !this.field || this.readRevision() === null) return null;
    const radius = pad * 1.03;
    const { minX, maxX, minZ, maxZ } = this.bounds;
    if (x - radius < minX || x + radius > maxX || z - radius < minZ || z + radius > maxZ) return null;
    // clearBox has just flushed the same revision. A bilinear value is a convex
    // combination of four vertices, each at most one pixel diagonal away. If
    // no dangerous vertex lies in this enlarged disc, every point of the
    // entire foot disc is clear, including points between old sample offsets.
    const reach = radius + this.pixelDiagonal + 1e-9, reachSq = reach * reach;
    const first = Math.max(0, Math.ceil((z - reach - minZ) / this.stepZ));
    const last = Math.min(this.height - 1, Math.floor((z + reach - minZ) / this.stepZ));
    const stride = this.width + 1;
    this.stats.discQueries++;
    for (let row = first; row <= last; row++) {
      const dz = minZ + row * this.stepZ - z;
      const half = Math.sqrt(Math.max(0, reachSq - dz * dz));
      const left = Math.max(0, Math.ceil((x - half - minX) / this.stepX));
      const right = Math.min(this.width - 1, Math.floor((x + half - minX) / this.stepX));
      this.stats.discRows++;
      if (left <= right && this.rowPrefix[row * stride + right + 1] !== this.rowPrefix[row * stride + left]) return false;
    }
    return true;
  }
  snapshot() {
    return { available: Boolean(this.field && this.readRevision() !== null), tileSize: TILE, cols: this.cols || 0, rows: this.rows || 0,
      seenRevision: this.seenRevision, dirtyBlocks: this.dirtyList.length, dirtyRows: this.dirtyRows.length, fullDirty: this.fullDirty,
      bufferBytes: (this.hazard?.byteLength || 0) + (this.dirty?.byteLength || 0) + (this.prefix?.byteLength || 0) +
        (this.rowPrefix?.byteLength || 0) + (this.rowDirty?.byteLength || 0), ...this.stats };
  }
}
