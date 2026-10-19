/**
 * What the postman can bring back from a day: the articles and short posts published on it.
 * The site writes this for each language at build time, as /<lang>/postman.json.
 */
import type { Day } from './days'

export interface Article {
  kind: 'article'
  title: string
  /** The subtitle, when the article has one in the reader's language. */
  deck?: string
  category: string
  /** The category's name in the reader's language. */
  topic: string
  minutes: number
  href: string
  /** The opening lines, when the article is written in the reader's language. */
  lede?: string
  /** Which language the article is in, when that is not the reader's. */
  only?: string
  /** The notice for a lede whose text was translated by AI. */
  translated?: string
}

export interface Note {
  kind: 'note'
  html: string
  href: string
  /** The notice for a post that was translated by AI. */
  translated?: string
}

export type Story = Article | Note

/** Every day that has something, keyed by its date. */
export type Mail = Record<Day, Story[]>

/** The days that have mail, with the nearest one before or after any day. */
export function sorter(mail: Mail) {
  const days = Object.keys(mail).sort()
  /** The index of the first day with mail that is not before `day`. */
  const from = (day: Day) => {
    let lo = 0
    let hi = days.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (days[mid] < day) lo = mid + 1
      else hi = mid
    }
    return lo
  }
  return {
    days,
    get: (day: Day): Story[] => mail[day] ?? [],
    before: (day: Day): Day | undefined => days[from(day) - 1],
    after: (day: Day): Day | undefined => {
      const i = from(day)
      return days[i] === day ? days[i + 1] : days[i]
    },
  }
}
