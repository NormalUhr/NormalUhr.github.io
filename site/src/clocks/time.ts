/**
 * The one source of time that the postman clock and his town read.
 *
 * Clocks show wall-clock time in a time zone, read through Intl rather than by dividing
 * the milliseconds since midnight, because a daylight-saving change skips or repeats a
 * civil hour and only the zone's own rules say which hour it is.
 */

/** A moment, with the civil fields the clock reads. */
export interface Snapshot {
  /** Milliseconds since the epoch. */
  ts: number
  /** IANA time zone the civil fields are in. */
  zone: string
  /** Civil date in that zone, as YYYY-MM-DD. */
  date: string
  /** 0 to 23. */
  h: number
  /** 0 to 59. */
  m: number
  /** 0 to 59. */
  s: number
  /** Fraction of the current second, from 0 up to but not including 1. */
  u: number
  /** 60 * h + m. */
  minuteIndex: number
  /** (s + u) / 60. */
  minuteProgress: number
  /** (m + minuteProgress) / 60. */
  hourProgress: number
  /** (h + hourProgress) / 24. */
  dayProgress: number
}

export const pad = (n: number) => String(n).padStart(2, '0')

export const localZone = () => Intl.DateTimeFormat().resolvedOptions().timeZone

const formats = new Map<string, Intl.DateTimeFormat>()

function format(zone: string) {
  let f = formats.get(zone)
  if (!f) {
    // h23, not hour12: false, which some engines still print as 24 at midnight.
    f = new Intl.DateTimeFormat('en-US', {
      timeZone: zone,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    formats.set(zone, f)
  }
  return f
}

interface Civil {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  second: number
}

function civil(ts: number, zone: string): Civil {
  const parts = format(zone).formatToParts(ts)
  // The format always has all six parts, so each lookup finds one.
  const get = (type: Intl.DateTimeFormatPartTypes) => Number(parts.find((p) => p.type === type)!.value)
  return { year: get('year'), month: get('month'), day: get('day'), hour: get('hour'), minute: get('minute'), second: get('second') }
}

/** Minutes the zone is ahead of UTC at `ts`. */
export function offsetMinutes(ts: number, zone: string) {
  const c = civil(ts, zone)
  const asUtc = Date.UTC(c.year, c.month - 1, c.day, c.hour, c.minute, c.second)
  return Math.round((asUtc - Math.floor(ts / 1000) * 1000) / 60000)
}

export function snapshot(ts: number, zone: string): Snapshot {
  const c = civil(ts, zone)
  const u = (((ts % 1000) + 1000) % 1000) / 1000
  const minuteProgress = (c.second + u) / 60
  const hourProgress = (c.minute + minuteProgress) / 60
  return {
    ts,
    zone,
    date: `${c.year}-${pad(c.month)}-${pad(c.day)}`,
    h: c.hour,
    m: c.minute,
    s: c.second,
    u,
    minuteIndex: 60 * c.hour + c.minute,
    minuteProgress,
    hourProgress,
    dayProgress: (c.hour + hourProgress) / 24,
  }
}

/**
 * The moment a zone's clocks read `date` at h:m:s. Two passes, because the zone's offset
 * can differ between the first guess and the answer when a daylight-saving change lies
 * between them.
 */
export function civilToTs(date: string, h: number, m: number, s: number, zone: string) {
  const [year, month, day] = date.split('-').map(Number)
  const guess = Date.UTC(year, month - 1, day, h, m, s)
  const first = guess - offsetMinutes(guess, zone) * 60000
  return guess - offsetMinutes(first, zone) * 60000
}

/**
 * The time the clock reads. Normally the browser's own time in the reader's zone. In
 * preview it can be moved, paused or put in another zone, so the clock can be checked at
 * any moment without waiting for it.
 */
export class TimeSource {
  private place = localZone()
  private offset = 0
  private pinned: number | null = null
  private listeners = new Set<() => void>()

  get zone() {
    return this.place
  }

  now() {
    return this.pinned ?? Date.now() + this.offset
  }

  snapshot() {
    return snapshot(this.now(), this.zone)
  }

  /** How fast this time runs against real time: 1, or 0 while paused. */
  get rate() {
    return this.pinned === null ? 1 : 0
  }

  get paused() {
    return this.pinned !== null
  }

  /** Makes the clock read `ts` from now on. */
  seek(ts: number) {
    if (this.pinned === null) this.offset = ts - Date.now()
    else this.pinned = ts
    this.changed()
  }

  pause(paused: boolean) {
    if (paused === this.paused) return
    const now = this.now()
    this.pinned = paused ? now : null
    if (!paused) this.offset = now - Date.now()
    this.changed()
  }

  /** Makes the clock read the time in another zone. */
  setZone(zone: string) {
    if (zone === this.place) return
    this.place = zone
    this.changed()
  }

  /** Called whenever the time jumps for a reason other than its passing. */
  onChange(listener: () => void) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private changed() {
    for (const listener of this.listeners) listener()
  }
}
