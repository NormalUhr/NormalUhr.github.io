import { words } from './words.js';
export const TAU = Math.PI * 2;
export const FLOOR_HEIGHT = 6;
export const CORE_RADIUS = 4.4;
export const COLLAR_INNER = 4.6;
export const COLLAR_OUTER = 5.4;
export const BRIDGES = {
  hour: { id: 'hour', name: words.bridges.hour, radius: 22, width: 3.2, levels: [1, 2, 3], startLevel: 2 },
  minute: { id: 'minute', name: words.bridges.minute, radius: 27, width: 2.0, levels: [4, 5, 6], startLevel: 5 },
};
export const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
export const smooth = t => { t = clamp(t); return t * t * (3 - 2 * t); };
export const lerp = (a, b, t) => a + (b - a) * t;
export const wrap = a => ((a % TAU) + TAU) % TAU;
export const shortest = (a, b) => ((b - a + Math.PI) % TAU + TAU) % TAU - Math.PI;
export const polar = (r, angle, y = 0) => ({ x: Math.sin(angle) * r, y, z: -Math.cos(angle) * r });

export function clockAngles(minutes) {
  return { hour: wrap(minutes / 720 * TAU), minute: wrap(minutes / 60 * TAU) };
}

export function stairProfile(config) {
  const start = 6.2, end = config.radius - 1.7, middle = (start + end) / 2;
  const landing = 1.2, firstEnd = middle - landing / 2, secondStart = middle + landing / 2;
  const steps = 18, rise = 3 / steps, run = (firstEnd - start) / steps;
  const height = r => {
    if (r <= start) return 0;
    if (r < firstEnd) return Math.min(3, Math.ceil((r - start) / run - 1e-9) * rise);
    if (r <= secondStart) return 3;
    if (r < end) return 3 + Math.min(3, Math.ceil((r - secondStart) / run - 1e-9) * rise);
    return 6;
  };
  return { start, end, middle, landing, firstEnd, secondStart, steps, rise, run, height };
}

/** A geometry study: real-time azimuth is separate from the vertical docking cycle. */
export class BridgeMechanism {
  constructor(config) {
    this.config = config; this.level = config.startLevel; this.y = (this.level - 1) * FLOOR_HEIGHT;
    this.targetLevel = this.level; this.phase = 'open'; this.phaseTime = 0; this.open = 1;
    this.pendingLevel = null; this.visits = 0;
  }
  request(level) {
    if (!this.config.levels.includes(level)) throw new RangeError('楼层不属于此连廊的服务范围');
    if (this.phase !== 'open') { this.pendingLevel = level; return; }
    if (level === this.level) return;
    this.targetLevel = level; this.fromY = this.y; this.phase = 'closing'; this.phaseTime = 0;
  }
  update(dt) {
    this.phaseTime += dt;
    if (this.phase === 'closing') {
      this.open = 1 - smooth(this.phaseTime / 1.8);
      if (this.phaseTime >= 1.8) { this.open = 0; this.phase = 'moving'; this.phaseTime = 0; }
    } else if (this.phase === 'moving') {
      const toY = (this.targetLevel - 1) * FLOOR_HEIGHT;
      const seconds = 5.2 * Math.abs(toY - this.fromY) / FLOOR_HEIGHT;
      this.y = lerp(this.fromY, toY, smooth(this.phaseTime / seconds));
      if (this.phaseTime >= seconds) { this.y = toY; this.level = this.targetLevel; this.phase = 'opening'; this.phaseTime = 0; }
    } else if (this.phase === 'opening') {
      this.open = smooth(this.phaseTime / 1.8);
      if (this.phaseTime >= 1.8) { this.open = 1; this.phase = 'open'; this.phaseTime = 0; this.visits++; }
    } else if (this.pendingLevel !== null) {
      const next = this.pendingLevel; this.pendingLevel = null; this.request(next);
    }
  }
  get coreLevel() { return this.level - 1; }
  snapshot() { return { id: this.config.id, level: this.level, targetLevel: this.targetLevel, y: this.y, phase: this.phase, open: this.open, visits: this.visits }; }
}

export function geometricEvidence() {
  const h = stairProfile(BRIDGES.hour), m = stairProfile(BRIDGES.minute);
  let min = Infinity, at = null;
  for (let r = 5.2; r <= 22; r += .002) {
    const gap = 18 + m.height(r) - (12 + h.height(r));
    if (gap < min) { min = gap; at = r; }
  }
  return { minimumDeckGap: min, atRadius: at, assumedBeamDepth: .8, overheadClearance: min - .8,
    adultEnvelope: 2.0, clearanceAboveAdult: min - .8 - 2.0,
    hourRetractedCornerRadius: Math.hypot(22 - .9, 3.2 / 2),
    minuteRetractedCornerRadius: Math.hypot(27 - .9, 2.0 / 2),
    centralRadialGap: COLLAR_INNER - CORE_RADIUS };
}
