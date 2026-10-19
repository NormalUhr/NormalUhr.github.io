/**
 * The calendar behind the postman's town. Each day is a house and each month is a street,
 * and the street of one month meets the next at a crossing, so all of time lies along one
 * row of plots. A plot is numbered from January 1970, and the number is all the town needs
 * to know what stands on it.
 */
import { seeded } from '../scene.ts'
import { pad } from '../time.ts'

/** A day, as YYYY-MM-DD. */
export type Day = string

/** Plots between the last house of a street and the first house of the next: a garden, the crossing, a garden. */
export const GAP = 3

const MS = 864e5
/** The average month, in plots, for a first guess at which month a plot falls in. */
const MONTH = 365.2425 / 12 + GAP

export function parts(day: Day) {
  return { y: Number(day.slice(0, 4)), m: Number(day.slice(5, 7)), d: Number(day.slice(8, 10)) }
}

export const dayOf = (y: number, m: number, d: number): Day => `${y}-${pad(m)}-${pad(d)}`

export const monthLength = (y: number, m: number) => new Date(Date.UTC(y, m, 0)).getUTCDate()

export const isDay = (s: string) => /^\d{4}-\d{2}-\d{2}$/.test(s) && shift(s, 0) === s

/** The day `n` days after `day`. */
export function shift(day: Day, n: number): Day {
  const { y, m, d } = parts(day)
  return new Date(Date.UTC(y, m - 1, d + n)).toISOString().slice(0, 10)
}

/** Days from `a` to `b`. */
export function between(a: Day, b: Day) {
  const p = parts(a)
  const q = parts(b)
  return Math.round((Date.UTC(q.y, q.m - 1, q.d) - Date.UTC(p.y, p.m - 1, p.d)) / MS)
}

/** The first plot of the street for month `index`, counted in months from January 1970. */
function firstPlot(index: number) {
  return Date.UTC(1970, index, 1) / MS + GAP * index
}

export function plotOf(day: Day) {
  const { y, m, d } = parts(day)
  return firstPlot((y - 1970) * 12 + m - 1) + d - 1
}

/**
 * What stands on a plot: the house of a day, or one of the plots between two streets.
 * `gap` 0 is the garden after the month's last house, 1 the crossing and 2 the garden
 * before the next month's first house.
 */
export type Spot = { y: number; m: number; d: number; day: Day; gap?: undefined } | { y: number; m: number; gap: 0 | 1 | 2 }

export function spotAt(plot: number): Spot {
  let index = Math.floor(plot / MONTH)
  while (firstPlot(index) > plot) index--
  while (firstPlot(index + 1) <= plot) index++
  const y = 1970 + Math.floor(index / 12)
  const m = (((index % 12) + 12) % 12) + 1
  const k = plot - firstPlot(index)
  const length = monthLength(y, m)
  if (k < length) return { y, m, d: k + 1, day: dayOf(y, m, k + 1) }
  return { y, m, gap: (k - length) as 0 | 1 | 2 }
}

/**
 * The day of the house nearest to a plot. The crossing between two streets is as far from
 * the last house of one month as from the first house of the next, so there it is the one
 * in direction `dir`.
 */
export function nearestDay(plot: number, dir: number): Day {
  const spot = spotAt(Math.round(plot))
  if (spot.gap === undefined) return spot.day
  const end = dayOf(spot.y, spot.m, monthLength(spot.y, spot.m))
  if (spot.gap === 0 || (spot.gap === 1 && dir < 0)) return end
  return shift(end, 1)
}

export type Season = 'spring' | 'summer' | 'autumn' | 'winter'

export function season(day: Day): Season {
  const { m } = parts(day)
  return m === 12 || m <= 2 ? 'winter' : m <= 5 ? 'spring' : m <= 8 ? 'summer' : 'autumn'
}

export type Weather = 'clear' | 'rain' | 'snow'

/** The same day always has the same weather, because the date is the seed. */
export function weather(day: Day): Weather {
  const r = seeded(`sky-${day}`)()
  if (season(day) === 'winter') return r < 0.35 ? 'snow' : 'clear'
  return r < 0.2 ? 'rain' : 'clear'
}

/** About one house in ten has a cat asleep on its roof, and always the same houses, because the date is the seed. */
export const hasCat = (day: Day) => seeded(`cat-${day}`)() < 0.1

export type Festival = 'new-year-eve' | 'halloween' | 'christmas' | 'spring-eve' | 'spring-day' | 'spring' | 'mid-autumn'

let lunar: Intl.DateTimeFormat | null | undefined

/** Month and day in the Chinese calendar. A leap month never matches, because its number carries a suffix. */
function lunarDate(day: Day) {
  if (lunar === undefined) {
    try {
      lunar = new Intl.DateTimeFormat('en-u-ca-chinese', { timeZone: 'UTC', month: 'numeric', day: 'numeric' })
    } catch {
      lunar = null
    }
  }
  if (!lunar) return null
  const { y, m, d } = parts(day)
  const p = lunar.formatToParts(Date.UTC(y, m - 1, d))
  return { m: p.find((x) => x.type === 'month')?.value, d: Number(p.find((x) => x.type === 'day')?.value) }
}

export function festival(day: Day): Festival | null {
  const { m, d } = parts(day)
  if (m === 12 && d === 31) return 'new-year-eve'
  if (m === 10 && d === 31) return 'halloween'
  if (m === 12 && (d === 24 || d === 25)) return 'christmas'
  const l = lunarDate(day)
  if (!l) return null
  if (l.m === '1' && l.d <= 15) return l.d === 1 ? 'spring-day' : 'spring'
  if (l.m === '8' && l.d === 15) return 'mid-autumn'
  const next = lunarDate(shift(day, 1))
  if (next?.m === '1' && next.d === 1) return 'spring-eve'
  return null
}
