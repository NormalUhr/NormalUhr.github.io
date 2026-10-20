import { createSandField } from './sand_field.js';

/**
 * A read-only witness for clock grooves, separate from the visible sand.
 * Broad brush bounds find possible damage; a height comparison decides whether
 * that damage really reached the teeth. Planning never writes to real sand.
 */
export class ClockSurface {
  constructor(sand, points, { pristine = false } = {}) {
    this.field = sand.field || sand;
    this.points = points;
    this.pristine = pristine;
    this.reference = null;
    this.checkedRevision = new Float64Array(points.length - 1).fill(-1);
    this.checkedTolerance = new Float64Array(points.length - 1);
    this.intact = new Uint8Array(points.length - 1);
  }

  reset() {
    this.reference = null;
    this.pristine = false;
    this.checkedRevision.fill(-1);
  }

  ensure(recordedMinute) {
    if (this.reference) return this.reference;
    const field = this.field;
    if (!field.heights || !field.width || !field.height || !field.heightAt) return null;
    const reference = createSandField({ width: field.width, height: field.height });
    if (this.pristine) reference.heights.set(field.heights);
    else {
      // Old saves have only the real surface. Reconstruct a comparison surface,
      // never their visible sand, from the recorded clock curve at this quality.
      const step = this.points[1].m - this.points[0].m;
      for (let i = 0; i + 1 < this.points.length && i * step < recordedMinute; i++) {
        const a = this.points[i], next = this.points[i + 1];
        const t = Math.min(1, (recordedMinute - a.m) / step);
        reference.stroke(a.x, a.z, a.x + (next.x - a.x) * t, a.z + (next.z - a.z) * t,
          { kind: 'rake', width: .16, strength: 1 });
      }
    }
    this.pristine = false;
    this.reference = reference;
    return reference;
  }

  beforeDisturb(recordedMinute) { this.ensure(recordedMinute); }

  paintClock(a, b) {
    // Until the first disturbance the visible field itself is the pristine
    // witness. Afterwards mirror only real clock painting, never smoothing,
    // visitor strokes or footprints, so small errors cannot accumulate unseen.
    this.reference?.stroke(a.x, a.z, b.x, b.z, { kind: 'rake', width: .16, strength: 1 });
  }

  isIntact(index, recordedMinute, tolerance = .0002) {
    const reference = this.ensure(recordedMinute);
    if (!reference || index < 0 || index + 1 >= this.points.length) return false;
    const revision = this.field.revision;
    if (Number.isFinite(revision) && this.checkedRevision[index] === revision && this.checkedTolerance[index] === tolerance) return this.intact[index] === 1;
    const a = this.points[index], b = this.points[index + 1];
    const length = Math.hypot(b.x - a.x, b.z - a.z);
    if (!length) return false;
    const nx = -(b.z - a.z) / length, nz = (b.x - a.x) / length;
    let intact = true;
    // Check all five valleys and the ridges between them, at three positions
    // along the edge. A deep visitor furrow cannot impersonate a complete rake.
    for (const t of [.2, .5, .8]) {
      const x = a.x + (b.x - a.x) * t, z = a.z + (b.z - a.z) * t;
      for (const offset of [-.08, -.06, -.04, -.02, 0, .02, .04, .06, .08]) {
        const px = x + nx * offset, pz = z + nz * offset;
        if (Math.abs(this.field.heightAt(px, pz) - reference.heightAt(px, pz)) > tolerance) {
          intact = false;
          break;
        }
      }
      if (!intact) break;
    }
    if (Number.isFinite(revision)) this.checkedRevision[index] = revision;
    this.checkedTolerance[index] = tolerance;
    this.intact[index] = Number(intact);
    return intact;
  }
}
