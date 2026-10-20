/**
 * Retain visible-page time until the simulation can process it. Rendering and
 * expensive route planning may delay a frame; neither is permission to erase
 * that time. Visibility, pause, and OS-suspend handling explicitly call reset.
 * This module has no wall clock, DOM, renderer, or hidden catch-up limit.
 */
export class FrameTime {
  constructor({ maxStep = .05 } = {}) {
    if (!Number.isFinite(maxStep) || maxStep <= 0 || maxStep > .05) throw new RangeError('FrameTime maxStep must be between 0 and .05 seconds.');
    this.maxStep = maxStep;
    this._pending = 0;
    this._received = 0;
    this._advanced = 0;
    this._discarded = 0;
    this._resets = 0;
  }

  add(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) throw new RangeError('FrameTime accepts finite, nonnegative elapsed seconds.');
    this._pending += seconds;
    this._received += seconds;
    return this._pending;
  }

  /** Run synchronous small steps; a spent CPU budget leaves the rest queued. */
  drain(step, shouldYield = () => false) {
    let steps = 0, advancedSeconds = 0;
    while (this._pending > 0) {
      // Make at least one real step per visible frame. An already spent frame
      // budget must not turn pending work into a permanent no-progress loop.
      if (steps > 0 && shouldYield()) break;
      const dt = Math.min(this.maxStep, this._pending);
      step(dt);
      this._pending = Math.max(0, this._pending - dt);
      this._advanced += dt;
      advancedSeconds += dt;
      steps++;
    }
    return { steps, advancedSeconds, pendingSeconds: this._pending };
  }

  /** Explicitly discard a paused/background backlog, never elapsed silently. */
  reset() {
    const discardedSeconds = this._pending;
    this._discarded += discardedSeconds;
    this._pending = 0;
    this._resets++;
    return discardedSeconds;
  }

  get pendingSeconds() { return this._pending; }

  snapshot() {
    return { maxStep: this.maxStep, receivedSeconds: this._received, advancedSeconds: this._advanced,
      pendingSeconds: this._pending, discardedSeconds: this._discarded, resets: this._resets };
  }
}
