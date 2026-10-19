/**
 * The date in the corner of the town, printed the way an old film camera printed it on a
 * photo: an apostrophe and the last two figures of the year, then the month and the day,
 * each figure lit from seven bars.
 */
import { parts, type Day } from './days'

/** The bars each figure lights: a top, b top right, c bottom right, d bottom, e bottom left, f top left, g middle. */
const LIT = ['abcdef', 'bc', 'abdeg', 'abcdg', 'bcfg', 'acdfg', 'acdefg', 'abc', 'abcdefg', 'abcdfg']
/** A figure's width and height, and the thickness of its bars. */
const W = 5.4
const H = 10
const T = 1.2
/** From one figure to the next, and the space between the year, the month and the day. */
const STEP = 7
const SPACE = 4.4
/** The whole date, from the apostrophe to the last figure. */
export const WIDTH = 2 * T + 3 * (STEP + W) + 2 * SPACE

const bar = (x: number, y: number, w: number, h: number) => `M${x} ${y}h${w}v${h}h${-w}Z`

function figure(n: number, x: number) {
  const across = W - 1.2 * T
  const down = H / 2 - 0.9 * T
  const bars: Record<string, string> = {
    a: bar(x + 0.6 * T, 0, across, T),
    b: bar(x + W - T, 0.6 * T, T, down),
    c: bar(x + W - T, H / 2 + 0.3 * T, T, down),
    d: bar(x + 0.6 * T, H - T, across, T),
    e: bar(x, H / 2 + 0.3 * T, T, down),
    f: bar(x, 0.6 * T, T, down),
    g: bar(x + 0.6 * T, (H - T) / 2, across, T),
  }
  return [...LIT[n]].map((k) => bars[k]).join('')
}

/** The bars that print `day`, as one path to fill, from x = 0 to WIDTH and y = 0 to the figures' height. */
export function datestamp(day: Day) {
  const { y, m, d } = parts(day)
  let path = `M${T * 0.5} 0h${T}l${-T * 0.6} ${H * 0.3}h${-T}Z`
  let x = 2 * T
  for (const n of [y % 100, m, d]) {
    path += figure(Math.floor(n / 10), x) + figure(n % 10, x + STEP)
    x += STEP + W + SPACE
  }
  return path
}
