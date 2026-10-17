// Writes src/generated/country-points.json: one point per country for the visitor globe,
// keyed by ISO 3166-1 alpha-2 code. The points are the label points of Natural Earth's
// 1:50m countries, which are public domain and sit inside each country's main landmass.
//
//   node scripts/country-points.mjs
import fs from 'node:fs'
import path from 'node:path'

const SOURCE =
  'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson'
const SITE = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const OUT = path.join(SITE, 'src', 'generated', 'country-points.json')

const res = await fetch(SOURCE)
if (!res.ok) throw new Error(`Natural Earth download failed: HTTP ${res.status}`)
const { features } = await res.json()

const round = (x) => Math.round(x * 100) / 100
const points = {}
// ISO_A2 first. ISO_A2_EH only fills the codes that ISO_A2 leaves out (France, Norway,
// Taiwan, Kosovo), because it also gives two Australian territories the code AU, and
// either of them would otherwise move Australia's point into the Indian Ocean.
for (const field of ['ISO_A2', 'ISO_A2_EH']) {
  for (const { properties: p } of features) {
    const code = p[field]
    if (/^[A-Z]{2}$/.test(code ?? '') && !(code in points)) points[code] = [round(p.LABEL_Y), round(p.LABEL_X)]
  }
}

const lines = Object.keys(points).sort().map((code) => `  "${code}": [${points[code].join(', ')}]`)
fs.writeFileSync(OUT, `{\n${lines.join(',\n')}\n}\n`)
console.log(`${lines.length} countries -> ${path.relative(process.cwd(), OUT)}`)
