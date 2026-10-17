import { site, type Lang } from '../site.config'
import points from '../generated/country-points.json'

export interface Country {
  /** ISO 3166-1 alpha-2 code. */
  code: string
  /** English name, as GoatCounter gives it. */
  name: string
  visits: number
  /** Visits in the 24 hours before the build. */
  recent: number
  /** Latitude and longitude of the country's dot on the globe. */
  point: [number, number]
}

interface LocationsPage {
  stats: { id: string; name: string; count: number }[]
  more: boolean
}

const POINTS = points as Record<string, [number, number]>
const zhNames = new Intl.DisplayNames(['zh-Hans'], { type: 'region', style: 'short' })

/** The country's name in the page's language. */
export const placeName = (c: Country, lang: Lang) => (lang === 'zh' ? zhNames.of(c.code) ?? c.name : c.name)

let loading: Promise<Country[] | null> | undefined

/**
 * Visits per country over the whole history of the site and over the last 24 hours, read
 * from GoatCounter's API once per build. The key comes from the GOATCOUNTER_TOKEN environment
 * variable and never reaches the built pages. Without a key, or without any located visit,
 * this is null and the globe is left out. With a key, a failed request fails the build, so a
 * broken key cannot quietly remove the globe from the live site.
 */
export function countryVisits(): Promise<Country[] | null> {
  loading ??= load()
  return loading
}

async function load(): Promise<Country[] | null> {
  const token = process.env.GOATCOUNTER_TOKEN?.trim()
  if (!site.goatcounter || !token) return null

  // GoatCounter wants the start rounded to the hour.
  const dayAgo = new Date(Date.now() - 24 * 3600 * 1000)
  dayAgo.setUTCMinutes(0, 0, 0)
  const [all, day] = await Promise.all([
    locations(token, `${site.since}-01-01T00:00:00Z`),
    locations(token, dayAgo.toISOString().replace('.000Z', 'Z')),
  ])

  const countries: Country[] = []
  for (const [code, { name, count }] of all) {
    // Visits whose country GoatCounter could not tell have no place on the globe.
    const point = POINTS[code]
    if (point) countries.push({ code, name, visits: count, recent: day.get(code)?.count ?? 0, point })
  }
  return countries.length ? countries.sort((a, b) => b.visits - a.visits) : null
}

async function locations(token: string, start: string) {
  const found = new Map<string, { name: string; count: number }>()
  for (let offset = 0; ; offset += 100) {
    const url = new URL(`https://${site.goatcounter}.goatcounter.com/api/v0/stats/locations`)
    // Without a start, GoatCounter answers for recent days only.
    url.searchParams.set('start', start)
    url.searchParams.set('limit', '100')
    url.searchParams.set('offset', String(offset))
    const res = await fetch(url, {
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    })
    if (!res.ok) throw new Error(`GoatCounter locations: HTTP ${res.status} ${await res.text()}`)
    const page = (await res.json()) as LocationsPage
    for (const s of page.stats) found.set(s.id, { name: s.name, count: s.count })
    if (!page.more) break
  }
  return found
}
