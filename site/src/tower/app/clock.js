import { TAU, clockAngles, shortest, smooth, lerp, clamp } from './mechanism.js';

export const localMinutes = (date = new Date()) => date.getHours() * 60 + date.getMinutes() + date.getSeconds() / 60 + date.getMilliseconds() / 60000;
const withinDay = minutes => ((minutes % 1440) + 1440) % 1440;
const dayDelta = (from, to) => ((to - from + 720) % 1440 + 1440) % 1440 - 720;
const rates = { hour: TAU / 720, minute: TAU / 60 };

/** Clock azimuth is independent of the bridge's vertical service schedule.
 * All accelerated changes wait for people to clear the seams, then close the
 * bridges before moving. The controller never edits a person's world state.
 */
export class TowerClock {
  constructor(date = new Date()) {
    this.mode = 'live'; this.phase = 'idle'; this.elapsed = 0;
    this.minutes = localMinutes(date); this.angles = clockAngles(this.minutes);
    this.gateFactor = 1; this.previewMinutes = this.minutes;
    this.operation = null; this.queued = null; this.lastDate = date.getTime();this.lastLocalMinutes=this.minutes;
  }
  get busy() { return this.phase !== 'idle'; }
  get allowBridgeEntry() { return this.phase === 'idle' && this.gateFactor > .999; }
  requestLive(date = new Date()) { this._request({ mode: 'live', minutes: localMinutes(date) }); }
  requestPreview(minutes) {
    if (!Number.isFinite(minutes)) return false;
    this._request({ mode: 'preview', minutes: withinDay(minutes) }); return true;
  }
  startTour() {
    if (this.busy) return false;
    this._request({ mode: 'tour', minutes: this.minutes }); return true;
  }
  _request(request) {
    if (this.busy) { this.queued = request; return; }
    this.operation = { ...request, from: { ...this.angles }, fromMinutes: this.minutes };
    this.phase = 'waiting'; this.elapsed = 0;
  }
  _track(minutes) {
    const target = clockAngles(minutes);
    for (const id of ['hour', 'minute']) this.angles[id] += shortest(this.angles[id], target[id]);
  }
  update(dt, { seamsClear = true, date = new Date() } = {}) {
    dt = clamp(dt, 0, .25);
    const realMinutes = localMinutes(date);
    const wallJump = Math.abs((date.getTime() - this.lastDate) / 1000 - dt) > .30 || Math.abs(dayDelta(this.lastLocalMinutes,realMinutes)*60-dt)>.30;
    if (wallJump && (this.mode === 'live' || this.operation?.mode === 'live')) {
      if (this.phase === 'turning') {
        // Retarget from the currently visible pose, never from the old start.
        const op=this.operation,target=clockAngles(realMinutes);
        op.from={...this.angles};op.fromMinutes=this.minutes;op.to={};
        for(const id of ['hour','minute'])op.to[id]=op.from[id]+shortest(op.from[id],target[id]);
        op.toMinutes=op.fromMinutes+dayDelta(op.fromMinutes,realMinutes);this.elapsed=0;
      } else if (this.phase === 'opening') {
        // Even a nearly open plate must clear its passengers and retract again.
        this.operation={mode:'live',minutes:realMinutes,from:{...this.angles},fromMinutes:this.minutes};
        this.phase='waiting';this.elapsed=0;
      } else if (this.phase === 'idle') this._request({mode:'live',minutes:realMinutes});
    }
    if (this.mode === 'preview' && this.phase !== 'turning') this.previewMinutes += dt / 60;
    if (this.phase === 'idle') {
      const next = this.mode === 'live' ? realMinutes : this.previewMinutes;
      const target = clockAngles(next);
      // OS time changes and tab suspensions are handled as a visible, gated turn.
      if (Math.abs(shortest(this.angles.minute, target.minute)) > .035 || Math.abs(shortest(this.angles.hour, target.hour)) > .006 || Math.abs(dayDelta(this.minutes,next)) > Math.max(.005,dt/60*2)) this._request({ mode: this.mode, minutes: next });
      else { this.minutes = next; this._track(next); }
    }
    if (this.phase === 'waiting') {
      // Keep ordinary time flowing while an already committed passenger steps off.
      const next = this.mode === 'live' ? this.minutes + dt / 60 : this.previewMinutes;
      this.minutes = next; this._track(next);
      if (seamsClear) {
        this.phase = 'closing'; this.elapsed = 0;
        this.operation.from = { ...this.angles }; this.operation.fromMinutes = this.minutes;
        this.operation.fromOpen=this.gateFactor;
      }
    } else if (this.phase === 'closing') {
      this.elapsed += dt; this.gateFactor = (this.operation.fromOpen??1)*(1 - smooth(this.elapsed / 1.6));
      // Azimuth remains fixed after the first plate begins retracting.
      if (this.elapsed >= 1.6) {
        const op = this.operation; this.gateFactor = 0; this.phase = 'turning'; this.elapsed = 0;
        op.from = { ...this.angles }; op.fromMinutes = this.minutes;
        const destination = op.mode === 'live' ? realMinutes : op.minutes;
        op.toMinutes = op.mode === 'tour' ? op.fromMinutes + 60 : op.fromMinutes + dayDelta(op.fromMinutes, destination);
        op.to = {}; const target = clockAngles(destination);
        for (const id of ['hour', 'minute']) op.to[id] = op.from[id] + (op.mode === 'tour' ? rates[id] * 60 : shortest(op.from[id], target[id]));
        this.mode = op.mode;
      }
    } else if (this.phase === 'turning') {
      this.elapsed += dt; const op = this.operation;
      const duration = op.mode === 'tour' ? 26 : 6;
      const u = smooth(this.elapsed / duration);
      if (op.mode === 'live') {
        const target = clockAngles(realMinutes);
        for (const id of ['hour', 'minute']) op.to[id] += shortest(op.to[id], target[id]);
        op.toMinutes += dayDelta(op.toMinutes, realMinutes);
      }
      for (const id of ['hour', 'minute']) this.angles[id] = lerp(op.from[id], op.to[id], u);
      // Daylight takes the shortest path around a full day, including AM / PM.
      this.minutes = lerp(op.fromMinutes, op.toMinutes, u);
      if (this.elapsed >= duration) {
        this.phase = 'opening'; this.elapsed = 0;
        if (op.mode === 'live') this.minutes = realMinutes;
        else {
          this.mode = 'preview';
          this.previewMinutes = op.mode === 'tour' ? op.fromMinutes + 60 : op.minutes;
          this.minutes = this.previewMinutes;
        }
      }
    } else if (this.phase === 'opening') {
      this.elapsed += dt; this.gateFactor = smooth(this.elapsed / 1.6);
      this.minutes = this.mode === 'live' ? realMinutes : this.previewMinutes;
      this._track(this.minutes);
      if (this.elapsed >= 1.6) {
        this.phase = 'idle'; this.elapsed = 0; this.gateFactor = 1; this.operation = null;
        if (this.queued) { const next = this.queued; this.queued = null; this._request(next); }
      }
    }
    this.lastDate = date.getTime();this.lastLocalMinutes=realMinutes;
    return this;
  }
  snapshot() {
    return { version: 1, mode: this.mode, phase: this.phase, elapsed: this.elapsed,
      minutes: this.minutes, angles: { ...this.angles }, gateFactor: this.gateFactor,
      previewMinutes: this.previewMinutes, lastDate:this.lastDate, lastLocalMinutes:this.lastLocalMinutes, operation: this.operation && JSON.parse(JSON.stringify(this.operation)), queued: this.queued };
  }
  restore(data, date = new Date()) {
    if (!data || data.version !== 1 || !Number.isFinite(data.angles?.hour) || !Number.isFinite(data.angles?.minute) || !Number.isFinite(data.minutes)) return false;
    this.angles = { ...data.angles }; this.minutes = data.minutes;
    this.mode = ['live', 'preview', 'tour'].includes(data.mode) ? data.mode : 'live';
    this.previewMinutes = Number.isFinite(data.previewMinutes) ? data.previewMinutes : this.minutes;
    this.lastDate=Number.isFinite(data.lastDate)?data.lastDate:0;
    this.lastLocalMinutes=Number.isFinite(data.lastLocalMinutes)?data.lastLocalMinutes:localMinutes(date);
    if (['waiting', 'closing', 'turning', 'opening'].includes(data.phase) && data.operation) {
      this.phase = data.phase; this.operation = JSON.parse(JSON.stringify(data.operation));
      this.elapsed = Number(data.elapsed) || 0; this.gateFactor = clamp(Number(data.gateFactor) || 0);
      this.queued = data.queued || null;
    } else {
      this.phase = 'idle'; this.gateFactor = 1; this.operation = null;
      if (this.mode === 'tour') this.mode = 'preview';
      if (this.mode === 'live') this.requestLive(date);
    }
    return true;
  }
  get displayTime() {
    const minutes = Math.floor(withinDay(this.minutes));
    return `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
  }
}
