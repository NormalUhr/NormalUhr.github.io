// Serves dist/ and checks every built page: JS errors, horizontal overflow, missing
// titles, and whether every internal link points at something that actually exists.
//
//   node scripts/audit.mjs [--shots]
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import puppeteer from 'puppeteer-core'

const SITE = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const DIST = path.join(SITE, 'dist')
const SHOTS = path.join(SITE, 'shots')
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const PORT = 4399
const WANT_SHOTS = process.argv.includes('--shots')

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript',
  '.json': 'application/json', '.xml': 'application/xml', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf',
  '.mp4': 'video/mp4', '.mp3': 'audio/mpeg', '.ico': 'image/x-icon',
}

function resolveFile(pathname) {
  const clean = decodeURIComponent(pathname.split('?')[0].split('#')[0])
  const candidates = [
    path.join(DIST, clean),
    path.join(DIST, clean, 'index.html'),
    path.join(DIST, `${clean}.html`),
  ]
  for (const c of candidates) {
    if (!path.resolve(c).startsWith(DIST)) continue
    if (fs.existsSync(c) && fs.statSync(c).isFile()) return c
  }
  return null
}

const server = http.createServer((req, res) => {
  const file = resolveFile(new URL(req.url, 'http://x').pathname)
  if (!file) {
    res.writeHead(404, { 'content-type': 'text/plain' })
    return res.end('404')
  }
  const type = TYPES[path.extname(file)] ?? 'application/octet-stream'
  const size = fs.statSync(file).size
  // <video> and <audio> ask for byte ranges, so a plain 200 is not enough.
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '')
  if (range) {
    const start = range[1] ? Number(range[1]) : 0
    const end = range[2] ? Number(range[2]) : size - 1
    res.writeHead(206, {
      'content-type': type,
      'accept-ranges': 'bytes',
      'content-range': `bytes ${start}-${end}/${size}`,
      'content-length': end - start + 1,
    })
    return fs.createReadStream(file, { start, end }).pipe(res)
  }
  res.writeHead(200, { 'content-type': type, 'accept-ranges': 'bytes', 'content-length': size })
  fs.createReadStream(file).pipe(res)
})
await new Promise((r) => server.listen(PORT, r))

/** Every built page, as a site-root path. */
function pages(dir = DIST, prefix = '') {
  const out = []
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name)
    // Media lives behind a symlink; do not walk into it.
    if (fs.lstatSync(full).isSymbolicLink()) continue
    if (fs.statSync(full).isDirectory()) out.push(...pages(full, `${prefix}/${name}`))
    else if (name === 'index.html') out.push(prefix === '' ? '/' : prefix)
    else if (name.endsWith('.html')) out.push(`${prefix}/${name.replace(/\.html$/, '')}`)
  }
  return out
}

const isRedirectStub = (route) => {
  const file = resolveFile(route)
  return file ? /http-equiv=["']?refresh/i.test(fs.readFileSync(file, 'utf8')) : false
}
const every = pages().sort()
const all = every.filter((r) => !isRedirectStub(r))
const stubs = every.length - all.length
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true })
const problems = []
const seenLinks = new Map()

const WIDTHS = [1440, 390]

for (const route of all) {
 for (const width of WIDTHS) {
  const page = await browser.newPage()
  await page.setViewport({ width, height: 900 })
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e.message).slice(0, 160)))
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 160)) })
  page.on('response', (r) => {
    // Third-party fonts and analytics are unreachable offline by design.
    if (/googleapis|gstatic|goatcounter|giscus|gc\.zgo\.at|\/v1\//.test(r.url())) return
    if (r.status() >= 400) errors.push(`HTTP ${r.status()} ${r.url().replace(`http://localhost:${PORT}`, '')}`)
  })

  const res = await page.goto(`http://localhost:${PORT}${route}`, { waitUntil: 'load' })
  await new Promise((r) => setTimeout(r, 120))

  const info = await page.evaluate(() => ({
    title: document.title,
    overflow: document.documentElement.scrollWidth > window.innerWidth + 2,
    links: [...document.querySelectorAll('a[href^="/"]')].map((a) => a.getAttribute('href')),
    text: (document.body?.innerText ?? '').slice(0, 4000),
  }))

  if (res.status() !== 200) problems.push(`${route}: HTTP ${res.status()}`)
  if (!info.title) problems.push(`${route}: empty <title>`)
  if (info.overflow) problems.push(`${route}: scrolls horizontally at ${width}px wide`)
  if (/undefined|\[object Object\]|NaN/.test(info.text)) problems.push(`${route}: placeholder text in body`)
  if (errors.length) problems.push(`${route}: ${[...new Set(errors)].join(' | ')}`)
  for (const href of info.links) {
    const key = href.split('#')[0]
    if (!seenLinks.has(key)) seenLinks.set(key, route)
  }
  await page.close()
 }
}

// Every media reference in every built page must resolve to a real file. This is
// checked against the filesystem rather than in the browser, because a browser
// cancels media preloads as soon as the page closes.
const mediaMissing = new Map()
const mediaExternal = new Map()
const walkHtml = (dir) => {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name)
    if (fs.lstatSync(full).isSymbolicLink()) continue
    if (fs.statSync(full).isDirectory()) walkHtml(full)
    else if (name.endsWith('.html')) {
      const html = fs.readFileSync(full, 'utf8')
      for (const m of html.matchAll(/<(?:img|video|audio|source)\b[^>]*\bsrc="([^"]+)"/g)) {
        const url = m[1]
        if (/^https?:\/\//.test(url)) { mediaExternal.set(url, path.relative(DIST, full)); continue }
        if (!url.startsWith('/')) { mediaMissing.set(url, `${path.relative(DIST, full)}: not an absolute path`); continue }
        if (!resolveFile(url)) mediaMissing.set(url, path.relative(DIST, full))
      }
    }
  }
}
walkHtml(DIST)
for (const [url, where] of mediaMissing) problems.push(`missing media ${url} (${where})`)
if (mediaExternal.size) {
  console.log(`\nnote: ${mediaExternal.size} media reference(s) point at another host, so they depend on it staying up:`)
  for (const [url, where] of mediaExternal) console.log(`  ${where}  ->  ${url.slice(0, 96)}`)
}

// A localhost service URL is fine while developing and wrong in a deployed build.
for (const f of ['src/site.config.ts']) {
  const cfg = fs.readFileSync(path.join(SITE, f), 'utf8')
  for (const m of cfg.matchAll(/^\s*(likes|goatcounter):\s*'(https?:\/\/(?:localhost|127\.0\.0\.1)[^']*)'/gm)) {
    problems.push(`site.config.ts: ${m[1]} points at ${m[2]} — fine for local testing, wrong to deploy`)
  }
}

// Internal links must resolve to something in dist.
const broken = [...seenLinks].filter(([href]) => !resolveFile(href))
for (const [href, from] of broken) problems.push(`dead link ${href} (first seen on ${from})`)

if (WANT_SHOTS) {
  fs.mkdirSync(SHOTS, { recursive: true })
  const wanted = [
    ['home', '/en/', 'dark', 1440, 980],
    ['home-zh', '/zh/', 'dark', 1440, 980],
    ['article', '/en/p/grpo', 'dark', 1440, 1000],
    ['article-light', '/en/p/grpo', 'light', 1440, 900],
    ['code', '/en/p/decorators', 'dark', 1440, 900],
    ['articles', '/en/articles', 'dark', 1440, 950],
    ['archive', '/zh/archive', 'dark', 1440, 900],
    ['music', '/zh/p/vogel-im-kafig', 'dark', 1440, 950],
    ['fallback', '/en/p/vogel-im-kafig', 'dark', 1440, 700],
    ['posts-empty', '/en/posts', 'dark', 1440, 560],
    ['about', '/en/about', 'dark', 1440, 700],
    ['search', '/en/search', 'dark', 1440, 560],
  ]
  for (const [name, route, scheme, w, h] of wanted) {
    const page = await browser.newPage()
    await page.setViewport({ width: w, height: h })
    await page.evaluateOnNewDocument((want) => {
      try { if (want === 'light') localStorage.setItem('theme', 'light'); else localStorage.removeItem('theme') } catch (e) {}
    }, scheme)
    await page.goto(`http://localhost:${PORT}${route}`, { waitUntil: 'load' })
    await page.evaluate(() => document.fonts.ready)
    await new Promise((r) => setTimeout(r, 600))
    await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
    await page.close()
  }
  console.log(`screenshots -> ${path.relative(process.cwd(), SHOTS)}`)
}

await browser.close()
server.close()

console.log(`\nchecked ${all.length} pages at ${WIDTHS.join('px and ')}px, ${seenLinks.size} internal links, all media references` +
  ` (${stubs} redirect stubs served but not measured)`)
const unique = [...new Set(problems)]
if (unique.length) {
  console.log(`\n${unique.length} problem(s):`)
  for (const p of unique) console.log('  ' + p)
  process.exitCode = 1
} else {
  console.log('clean: no JS errors, no dead links, no missing media, no horizontal overflow, every page titled')
}
