// Loads the built site in headless Chrome and checks the postman clock, the town it belongs to,
// and the Easter eggs that open the town. The clock, on the page that does not exist, is
// compared with src/clocks/encode.ts at the moments where clocks usually go wrong. The town is
// checked against the mail the site built into dist/<lang>/postman.json: where the postman
// starts, which day he walks to, and that the paper he hands over is that day's. Each egg must
// open the town on its day, and the town must fit a phone without scrolling sideways. The clock
// in the masthead must show the visitor's local time and open the woodland clock in the page's
// language, where the ants keep the same time, the opening view is open, and nothing in the forest
// fades with the camera. The years in the footer and the dish of raked sand beside them must open
// the karesansui garden, which keeps the same time and keeps what the visitor rakes. The clock
// beside the reading time of an article must open the cloister clock tower, which keeps the same
// time and keeps its settings.
//
//   node scripts/clocks.mjs [--shots]
import path from 'node:path'
import fs from 'node:fs'
import puppeteer from 'puppeteer-core'
import { serve } from './serve.mjs'
import * as E from '../src/clocks/encode.ts'
import * as D from '../src/clocks/museum/days.ts'
import { sorter } from '../src/clocks/museum/mail.ts'
import phrases from '../src/clocks/scenes/postman/phrases.ts'
import { WORDS as WOODLAND } from '../src/woodland/app/words.js'
import { WORDS as GARDEN } from '../src/garden/app/words.js'
import { WORDS as TOWER } from '../src/tower/app/words.js'

const SITE = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const DIST = path.join(SITE, 'dist')
const SHOTS = path.join(SITE, 'shots', 'clocks')
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const PORT = 4398
const THIRD_PARTY = /googleapis|gstatic|goatcounter|giscus|gc\.zgo\.at|workers\.dev/
const WANT_SHOTS = process.argv.includes('--shots')

const AT = ['00:00', '00:01', '12:00', '14:04', '14:05', '14:37', '14:59', '15:00', '23:05', '23:59']
const ACROSS = [
  ['14:04:58', 'a minute', '14:05'],
  ['14:59:58', 'an hour', '15:00'],
  ['23:59:58', 'midnight', '00:00'],
]
const MAIL = Object.fromEntries(['en', 'zh'].map((lang) => [lang, JSON.parse(fs.readFileSync(path.join(DIST, lang, 'postman.json'), 'utf8'))]))
const wait = (ms) => new Promise((r) => setTimeout(r, ms))

const server = await serve(DIST, PORT)
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true })
const problems = []
let checks = 0
const check = (ok, what) => {
  checks++
  if (!ok) problems.push(what)
}

async function open(query, { width = 1280, height = 900, theme = 'dark', lang = 'en', still = false, path = `/${lang}/`, before, counts = false } = {}) {
  const page = await browser.newPage()
  // GoatCounter's script is replaced by one that keeps the events the page asks it to count.
  if (counts) {
    await page.setRequestInterception(true)
    page.on('request', (r) => (/gc\.zgo\.at\/count\.js/.test(r.url()) ? r.respond({ status: 200, contentType: 'text/javascript', body: COUNTER }) : r.continue()))
  }
  await page.setViewport({ width, height, deviceScaleFactor: WANT_SHOTS ? 2 : 1 })
  if (still) await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e.message).slice(0, 200)))
  page.on('console', (m) => {
    // A request that a third party refuses, such as the like counter refusing localhost, is
    // reported at the page itself, with the third party's address only in the message.
    if (m.type() === 'error' && !THIRD_PARTY.test(m.location()?.url ?? '') && !THIRD_PARTY.test(m.text())) errors.push(m.text().slice(0, 200))
  })
  await page.evaluateOnNewDocument((t) => {
    try {
      if (t === 'light') localStorage.setItem('theme', 'light')
      else localStorage.removeItem('theme')
    } catch {}
  }, theme)
  if (before) await page.evaluateOnNewDocument(before)
  await page.goto(`http://localhost:${PORT}${path}?${query}`, { waitUntil: 'load' })
  return { page, errors }
}

/** Stands in for GoatCounter's script and keeps the name of every event it is asked to count. */
const COUNTER = 'window.goatcounter = { count: (v) => v.event && (window.counted = window.counted || []).push(v.path) }'
/** The events a page opened with `counts` has counted, in order. */
const counted = (page) => page.evaluate(() => window.counted ?? [])

const noErrors = (errors, where) => check(errors.length === 0, `${where}: ${[...new Set(errors)].join(' | ')}`)

// ---------- the postman clock, on the page that does not exist ----------

const LOST = '/404.html'

/** Runs in the page: the street on the sign and the house the postman is delivering to. */
function read(root) {
  const one = (sel, key) => root.querySelector(sel)?.getAttribute(key) ?? null
  return { street: one('[data-street]', 'data-street'), dest: one('[data-dest]', 'data-house') }
}

/** What the postman must show at HH:MM, in the shape `read` returns. */
function expected(hm) {
  const [h, m] = hm.split(':').map(Number)
  const { street, number } = E.postman({ h, m })
  return { street: String(street), dest: String(number) }
}

async function clock(page) {
  await page.waitForSelector('.lost[data-ready]:not([hidden])', { timeout: 10000 })
}

async function verifyClock(page, hm, what) {
  const got = await page.$eval('.lost', read)
  const want = expected(hm)
  check(got.street === want.street && got.dest === want.dest, `${what}: drew ${JSON.stringify(got)}, expected ${JSON.stringify(want)} for ${hm}`)
}

async function hide(page, hidden) {
  await page.evaluate((h) => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => (h ? 'hidden' : 'visible') })
    document.dispatchEvent(new Event('visibilitychange'))
  }, hidden)
}

for (const hm of AT) {
  const { page, errors } = await open(`clock-time=${hm}&clock-pause`, { path: LOST })
  await clock(page)
  await verifyClock(page, hm, `at ${hm}`)
  noErrors(errors, `on the clock at ${hm}`)
  await page.close()
}

// Boundaries crossed while running, checked once any flourish has settled. One page at a time,
// because Chrome hides every tab but the newest, and a hidden clock does not draw.
for (const [from, what, after] of ACROSS) {
  const { page, errors } = await open(`clock-time=${from}`, { path: LOST })
  await clock(page)
  await wait(6500)
  await verifyClock(page, after, `after ${what} passed`)
  noErrors(errors, `while ${what} passed`)
  await page.close()
}

// Time moves on while the tab is in the background; the clock must be right as soon as it is back.
{
  const { page, errors } = await open('clock-time=14:36:57', { path: LOST })
  await clock(page)
  await hide(page, true)
  await wait(5000)
  await hide(page, false)
  await wait(400)
  await verifyClock(page, '14:37', 'after the page came back')
  noErrors(errors, 'with the page in the background')
  await page.close()
}

// ---------- the town ----------

/** Runs in the page: where the postman stands, or null while he walks, the date printed in the corner, and what he holds. */
function town(svg) {
  const dest = svg.querySelector('[data-dest]')
  const shown = (el) => el && el.style.display !== 'none'
  return {
    day: dest?.getAttribute('data-day') ?? null,
    date: svg.querySelector('.mu-datestamp')?.dataset.date ?? null,
    holding: Boolean(shown(svg.querySelector('.pm-envelope')?.parentElement) || shown(svg.querySelector('.mu-held'))),
  }
}

const where = (page) => page.$eval('.cd-stage svg.mu', town)
const arrival = (page, day, timeout = 9000) =>
  page.waitForFunction((d) => document.querySelector('.cd-stage svg.mu [data-dest]')?.getAttribute('data-day') === d, { timeout }, day)

/** The hand each paper is written in, which leads its font stack. */
const HAND = { en: 'Grape Nuts', zh: 'Caveat Brush' }

/**
 * The paper's text must hold the day's stories: each article's title, and each short post's words.
 * It must be written in the hand for its language, which the town asked Google Fonts for.
 */
async function verifyPaper(page, lang, day, what) {
  await page.waitForSelector('dialog.np[open]', { timeout: 4000 })
  const hand = await page.$eval('dialog.np', (paper) => getComputedStyle(paper).fontFamily)
  check(hand.startsWith(`"${HAND[lang]}"`), `${what}: the paper is written in ${hand}`)
  const asked = await page.evaluate((f) => Boolean(document.querySelector(`link[href^="https://fonts.googleapis.com/"][href*="${f}"]`)), `family=${HAND[lang].replaceAll(' ', '+')}`)
  check(asked, `${what}: the town never asked for the paper's hands`)
  const ok = await page.$eval(
    'dialog.np',
    (paper, stories) => {
      const text = (html) => {
        const t = document.createElement('template')
        t.innerHTML = html
        return t.content.textContent.replace(/\s+/g, ' ').trim()
      }
      const page = paper.textContent.replace(/\s+/g, ' ')
      return stories.every((s) => page.includes(s.kind === 'article' ? s.title : text(s.html)))
    },
    MAIL[lang][day],
  )
  check(ok, `${what}: the paper does not carry the stories of ${day}`)
}

for (const lang of ['en', 'zh']) {
  const post = sorter(MAIL[lang])
  // A day a few days after the newest mail, so the postman starts with empty hands.
  const newest = post.days.at(-1)
  const today = D.shift(newest, 4)
  const { page, errors } = await open(`clock-open&clock-time=${today}T10:00&clock-pause`, { lang, counts: true })
  await arrival(page, today)
  check((await counted(page)).length === 0, `${lang}: opening the town with clock-open counted ${await counted(page)}`)
  const start = await where(page)
  check(start.holding === false, `${lang}: the postman starts at ${today} with something in his hands`)
  check(start.date === today, `${lang}: the town opened at ${today} prints the date ${start.date}`)

  // With empty hands he goes back to the newest mail and takes it out of the mailbox.
  await page.click('.mu-man')
  await arrival(page, newest).catch(() => {})
  const there = await where(page)
  check(there.day === newest && there.date === newest && there.holding, `${lang}: after one click the postman is at ${JSON.stringify(there)}, expected ${newest} with its mail`)

  // Handing it over opens that day's paper; Escape closes the paper and leaves the town open.
  await page.click('.mu-man')
  await verifyPaper(page, lang, newest, `${lang} at ${newest}`)
  await page.click('dialog.np time')
  check((await page.$eval('dialog.np', (d) => d.open)) && (await where(page)).day === newest, `${lang}: the date on the paper did something`)
  await page.keyboard.press('Escape')
  await page.waitForSelector('dialog.np', { hidden: true, timeout: 3000 }).catch(() => {})
  check(await page.$eval('dialog.clockdlg', (d) => d.open), `${lang}: Escape on the paper closed the town as well`)

  // The keys walk a day and come home.
  await page.keyboard.press('ArrowRight')
  await arrival(page, D.shift(newest, 1)).catch(() => check(false, `${lang}: ArrowRight did not walk to ${D.shift(newest, 1)}`))
  await page.keyboard.press('Home')
  await arrival(page, today).catch(() => check(false, `${lang}: Home did not walk back to ${today}`))
  noErrors(errors, `in the ${lang} town`)
  await page.close()
}

// The longest stretch without mail is crossed in one trip.
{
  const post = sorter(MAIL.en)
  let gap = { from: post.days[0], to: post.days[0] }
  for (let i = 1; i < post.days.length; i++) {
    if (D.between(post.days[i - 1], post.days[i]) > D.between(gap.from, gap.to)) gap = { from: post.days[i - 1], to: post.days[i] }
  }
  const { page, errors } = await open(`clock-open&clock-time=${gap.to}T10:00&clock-pause`)
  await arrival(page, gap.to)
  await page.click('.mu-man')
  await verifyPaper(page, 'en', gap.to, `at ${gap.to}`)
  await page.keyboard.press('Escape')
  await page.waitForSelector('dialog.np', { hidden: true, timeout: 3000 }).catch(() => {})
  await page.click('.mu-man')
  await arrival(page, gap.from).catch(() => check(false, `the trip from ${gap.to} back to ${gap.from} did not arrive`))
  const printed = (await where(page)).date
  check(printed === gap.from, `after the trip to ${gap.from} the town prints the date ${printed}`)
  noErrors(errors, 'on a long trip')
  await page.close()
}

// A cat sleeps on the roofs of the same houses wherever the postman walks.
{
  // A house with a cat two days before the visitor's day, in the same month, so it is built and in sight.
  let cat = D.shift(sorter(MAIL.en).days.at(-1), 2)
  while (!D.hasCat(cat) || D.parts(cat).m !== D.parts(D.shift(cat, 2)).m) cat = D.shift(cat, -1)
  const today = D.shift(cat, 2)
  const { page, errors } = await open(`clock-open&clock-time=${today}T10:00&clock-pause`)
  await arrival(page, today)
  const roofs = async (when) => {
    const seen = await page.$eval('.cd-stage svg.mu', (svg) =>
      [...svg.querySelectorAll('[data-day]:not(.mu-future)')].map((h) => [h.getAttribute('data-day'), Boolean(h.querySelector('.mu-cat'))]),
    )
    const wrong = seen.filter(([day, has]) => has !== D.hasCat(day)).map(([day]) => day)
    check(wrong.length === 0, `${when}: the cats on the roofs of ${wrong.join(', ')} are not the ones those houses have`)
    check(seen.some(([day, has]) => day === cat && has), `${when}: no cat on the roof of ${cat}`)
  }
  await roofs('at the start')
  await page.keyboard.press('ArrowLeft')
  await wait(400)
  await roofs('while the postman walks')
  await arrival(page, D.shift(today, -1)).catch(() => check(false, `ArrowLeft did not walk to ${D.shift(today, -1)}`))
  await roofs('after the walk')
  noErrors(errors, 'with the cats on the roofs')
  await page.close()
}

// The lampposts at a crossing name the month that ends there and the month that begins, and move with the street.
{
  const { y, m } = D.parts(sorter(MAIL.en).days.at(-1))
  const first = D.dayOf(y, m, 1)
  const before = D.shift(first, -1)
  const today = D.shift(first, 1)
  const name = (day) => `${phrases.en.months.split(' ')[D.parts(day).m - 1]} ${D.parts(day).y}`
  const { page, errors } = await open(`clock-open&clock-time=${today}T10:00&clock-pause`)
  await arrival(page, today)
  const signs = () =>
    page.$eval('.cd-stage svg.mu', (svg) => ({
      loose: svg.querySelectorAll('.pm-sign').length - svg.querySelectorAll('[data-plot] .pm-sign').length,
      at: Object.fromEntries([...svg.querySelectorAll('[data-plot] .pm-sign text')].map((t) => [t.textContent, Math.round(t.getBoundingClientRect().x)])),
    }))
  const was = await signs()
  check(was.loose === 0 && name(before) in was.at && name(first) in was.at, `the lampposts at the crossing name ${JSON.stringify(was)}, expected ${name(before)} and ${name(first)}`)
  await page.keyboard.press('ArrowRight')
  await arrival(page, D.shift(today, 1)).catch(() => {})
  const now = (await signs()).at[name(first)]
  check(now < was.at[name(first)] - 20, `the lamppost of ${name(first)} did not move with the street: x ${was.at[name(first)]} -> ${now}`)
  noErrors(errors, 'at a crossing')
  await page.close()
}

// With reduced motion the postman moves at once and nothing moves by itself.
{
  const post = sorter(MAIL.en)
  const today = D.shift(post.days.at(-1), 4)
  const { page, errors } = await open(`clock-open&clock-time=${today}T10:00&clock-pause`, { still: true })
  await arrival(page, today)
  await wait(300)
  const moving = await page.$eval('.cd-stage svg.mu', (svg) => svg.getAnimations({ subtree: true }).length)
  check(moving === 0, `with reduced motion ${moving} animations run in the town`)
  await page.click('.mu-man')
  await arrival(page, post.days.at(-1), 600).catch(() => check(false, 'with reduced motion the postman did not move at once'))
  noErrors(errors, 'with reduced motion')
  await page.close()
}

// ---------- the Easter eggs ----------

/** The visitor's day in the checks of the eggs, a few days after the newest mail. */
const DAY = D.shift(sorter(MAIL.en).days.at(-1), 4)

/** Waits for the town to open with the postman standing at a day, and returns that day. */
const opened = (page) =>
  page
    .waitForFunction(() => document.querySelector('dialog.clockdlg[open] svg.mu [data-dest]')?.getAttribute('data-day'), { timeout: 9000 })
    .then((h) => h.jsonValue(), () => null)

/** Runs in the page before it loads: the visitor's clock reads 14:59:58.5, so the hour turns 1.5 seconds in. */
function nearTheHour() {
  const Real = Date
  const now = new Real()
  const shift = new Real(now.getFullYear(), now.getMonth(), now.getDate(), 14, 59, 58, 500).getTime() - Real.now()
  globalThis.Date = class extends Real {
    constructor(...args) {
      super(...(args.length ? args : [Real.now() + shift]))
    }
    static now() {
      return Real.now() + shift
    }
  }
}

// By chance, the postman walks along the bottom of the window, and pressing him opens the town at today.
{
  const { page, errors } = await open(`eggs=now&clock-time=${DAY}T10:00&clock-pause`, { counts: true })
  const walking = await page
    .waitForSelector('.egg-walker', { timeout: 4000 })
    .then(() => page.$eval('.egg-street', (s) => s.getAnimations({ subtree: true }).length > 0), () => false)
  check(walking, 'with ?eggs=now the postman did not walk past')
  if (walking) {
    // He comes in from beyond the left edge of the window, and cannot be pressed before he is in it.
    await page.waitForFunction(() => document.querySelector('.egg-walker').getBoundingClientRect().left >= 0, { timeout: 4000 })
    await page.click('.egg-walker')
  }
  check((await opened(page)) === DAY, `the postman who walked past did not open the town at ${DAY}`)
  check(String(await counted(page)) === 'egg-postman-walk', `the postman who walked past counted ${await counted(page)}`)
  noErrors(errors, 'with the postman walking past')
  await page.close()
}

// He also walks past when the visitor's clock passes the hour, sooner than any walk by chance could start.
{
  const { page, errors } = await open('', { before: nearTheHour })
  const came = await page.waitForSelector('.egg-walker', { timeout: 2800 }).then(() => true, () => false)
  check(came, 'the postman did not walk past at the turn of the hour')
  noErrors(errors, 'at the turn of the hour')
  await page.close()
}

// A date on a page opens the town on that day.
{
  const { page, errors } = await open(`clock-time=${DAY}T10:00&clock-pause`, { path: '/en/articles/', counts: true })
  const day = await page.$eval('main time[datetime]', (t) => t.getAttribute('datetime'))
  await page.click('main time[datetime]')
  check((await opened(page)) === day, `the date ${day} did not open the town on that day`)
  check(String(await counted(page)) === 'egg-postman-date', `the date counted ${await counted(page)}`)
  noErrors(errors, 'with a date clicked')
  await page.close()
}

// The postman clock on the page that does not exist opens the town at today.
{
  const { page, errors } = await open(`clock-time=${DAY}T10:00&clock-pause`, { path: LOST, counts: true })
  await clock(page)
  await page.click('.lost')
  check((await opened(page)) === DAY, `the clock on the missing page did not open the town at ${DAY}`)
  check(String(await counted(page)) === 'egg-postman-lost', `the clock on the missing page counted ${await counted(page)}`)
  noErrors(errors, 'with the clock on the missing page pressed')
  await page.close()
}

// A reader who reaches the end of an article gets a letter, which opens the town on the article's day.
{
  const post = sorter(MAIL.en)
  const day = post.days.findLast((d) => MAIL.en[d].some((s) => s.kind === 'article'))
  const { href } = MAIL.en[day].find((s) => s.kind === 'article')
  const { page, errors } = await open(`eggs=now&clock-time=${DAY}T10:00&clock-pause`, { path: href, counts: true })
  await wait(1500)
  check(!(await page.$('.egg-letter')), 'the letter came before the reader reached the end of the article')
  // One jump to the bottom of the page passes the end of the article without stopping at it.
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))
  const came = await page.waitForSelector('.egg-letter', { timeout: 3000 }).then(() => true, () => false)
  check(came, 'no letter came at the end of the article')
  if (came) {
    await wait(1000)
    const top = await page.evaluate(() => scrollY)
    await page.click('.egg-letter')
    check((await opened(page)) === day, `the letter did not open the town on ${day}, the day of the article`)
    check(String(await counted(page)) === 'egg-postman-letter', `the letter counted ${await counted(page)}`)
    // The town opens over the place the reader has scrolled to.
    await wait(800)
    const moved = await page.evaluate((y) => Math.round(scrollY - y), top)
    check(moved === 0, `opening the town from the letter scrolled the page by ${moved}px`)
  }
  noErrors(errors, 'with the letter at the end of an article')
  await page.close()
}

// With reduced motion the postman who comes by stands still.
{
  const { page, errors } = await open(`eggs=now&clock-time=${DAY}T10:00&clock-pause`, { still: true })
  await page.waitForSelector('.egg-walker', { timeout: 4000 }).catch(() => {})
  const moving = await page.$eval('.egg-street', (s) => s.getAnimations({ subtree: true }).length).catch(() => -1)
  check(moving === 0, `with reduced motion the postman who comes by has ${moving} animations`)
  noErrors(errors, 'with reduced motion and the postman coming by')
  await page.close()
}

// ---------- the woodland clock ----------

/** Opens the woodland from the clock in the masthead, and returns its frame once it has drawn its first frame. */
async function woodland(page) {
  await page.click('.ltime')
  const frame = await page.waitForSelector('dialog.frdlg[open] iframe', { timeout: 5000 }).then((h) => h.contentFrame(), () => null)
  const ready = frame && (await frame.waitForFunction(() => window.__ANTKEEPERS__?.system?.ready, { timeout: 60000 }).then(() => true, () => false))
  return ready ? frame : null
}

// The clock in the masthead shows the visitor's local time. Pressing it opens the woodland over the
// page in the page's language, where the ants lay out the same time. Escape in the woodland and its
// close button both close it, and the page stays where the reader was.
for (const lang of ['en', 'zh']) {
  const { page, errors } = await open('', { lang, counts: true })
  const shown = await page.waitForFunction(() => document.querySelector('.ltime')?.textContent, { timeout: 5000 }).then((h) => h.jsonValue(), () => null)
  const local = await page.evaluate(() => {
    const d = new Date()
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  })
  check(shown === local, `${lang}: the clock in the masthead shows ${shown}, expected the local time ${local}`)
  await page.evaluate(() => window.scrollTo({ top: 600, behavior: 'instant' }))
  const top = await page.evaluate(() => scrollY)
  const frame = await woodland(page)
  check(frame, `${lang}: pressing the clock in the masthead did not open the woodland`)
  check(String(await counted(page)) === 'egg-woodland', `${lang}: opening the woodland counted ${await counted(page)}`)
  if (frame) {
    // The two clocks may read either side of the turn of a minute, so a difference is read again a second later.
    const differ = async () => {
      const pair = [await page.$eval('.ltime', (b) => b.textContent), await frame.$eval('#clockValue', (c) => c.textContent)]
      return pair[0] === pair[1] ? null : pair.join(' and ')
    }
    const twice = (await differ()) && (await wait(1200), await differ())
    check(!twice, `${lang}: the clock in the masthead and the ants show different times, ${twice}`)
    const lit = await frame.evaluate(() => window.__ANTKEEPERS__.system.moonlight?.debug.materialCount ?? 0)
    check(lit > 0, `${lang}: the woodland's moonlight reaches no material`)
    const label = await frame.$eval('#resetBtn', (b) => b.textContent)
    check(label === WOODLAND[lang].reset, `${lang}: the woodland's button reads "${label}", expected "${WOODLAND[lang].reset}"`)
    // Visitors come at their natural pace, with no test panel, no label over a visitor and no line of control hints.
    const test = await frame.evaluate(() => ({
      testing: window.__ANTKEEPERS__.snapshot().visitors?.testing.enabled,
      shown: ['visitorPanel', 'visitorMarker', 'cameraHint'].filter((id) => document.getElementById(id)),
    }))
    check(test.testing === false && !test.shown.length, `${lang}: the woodland runs its visitor test (testing ${test.testing}, shows ${test.shown.join(' ') || 'nothing'})`)
    // The camera walks. Holding W carries it forward over the clearing.
    const depth = () => frame.evaluate(() => window.__ANTKEEPERS__.system.camera.position.z)
    const start = await depth()
    await frame.focus('#scene')
    await page.keyboard.down('KeyW')
    await wait(600)
    await page.keyboard.up('KeyW')
    const walked = start - (await depth())
    check(walked > 0.5, `${lang}: holding W moved the woodland's camera forward by ${walked.toFixed(2)}`)
    await page.keyboard.press('Escape')
    const closed = await page
      .waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 3000 })
      .then(() => true, () => false)
    check(closed, `${lang}: Escape in the woodland did not close it`)
    const moved = await page.evaluate((y) => Math.round(scrollY - y), top)
    check(moved === 0, `${lang}: the woodland moved the page under it by ${moved}px`)
    const again = await woodland(page)
    check(again, `${lang}: the woodland did not open a second time`)
    if (again) {
      await page.click('.fr-close')
      const shut = await page.waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 3000 }).then(() => true, () => false)
      check(shut, `${lang}: the close button did not close the woodland`)
    }
  }
  noErrors(errors, `in the ${lang} woodland`)
  await page.close()
}

// The opening view looks over the clearing from among the trees at its edge. The forest plants no
// tree on that line of sight, so the opening view is open on a desktop and on a phone, and nothing in
// the forest fades with the camera. To find fading, each view is drawn twice while the woodland is
// paused, once as built and once with the fading of the original woodland taken out of the forest's
// shaders. To find how much of the opening view the forest covers, that view is also drawn without it.
for (const [size, opts] of [['desktop', {}], ['phone', { width: 390, height: 844 }]]) {
  const { page, errors } = await open('', { lang: 'en', ...opts })
  const frame = await woodland(page)
  check(frame, `${size}: the woodland did not open for the forest check`)
  if (frame) {
    await frame.evaluate(() => {
      window.__ANTKEEPERS__.setEnvironment({ hours: 12 })
      window.__ANTKEEPERS__.view.hours = 12
    })
    await wait(1000)
    const seen = await frame.evaluate(() => {
      const k = window.__ANTKEEPERS__
      const { renderer, scene, camera, controls, habitat } = k.system
      const gl = renderer.getContext()
      const opening = [camera.position.toArray(), controls.target.toArray()]
      const views = { opening, forest: [[0, 2.5, 8], [0, 2.5, 30]], trees: [[0, 5, 30], [0, 0, 0]] }
      const draw = ([position, target]) => {
        camera.position.set(...position)
        controls.target.set(...target)
        controls.update(0)
        renderer.render(scene, camera)
        const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4)
        gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
        return pixels
      }
      const share = (a, b) => {
        let changed = 0
        for (let i = 0; i < a.length; i += 4) if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 30) changed++
        return changed / (a.length / 4)
      }
      k.view.paused = true
      const built = Object.fromEntries(Object.entries(views).map(([name, view]) => [name, draw(view)]))
      habitat.forest.group.visible = false
      const bare = draw(opening)
      habitat.forest.group.visible = true
      for (const material of Object.values(habitat.forest.materials)) {
        const compile = material.onBeforeCompile
        const key = material.customProgramCacheKey
        material.onBeforeCompile = (shader) => {
          compile(shader)
          shader.fragmentShader = shader.fragmentShader.replace('diffuseColor.a *= forestVisibility;', '').replace('if (forestVisibility <= .00001) discard;', '')
        }
        material.customProgramCacheKey = () => key.call(material) + '-whole'
        material.needsUpdate = true
      }
      const faded = Object.fromEntries(Object.entries(views).map(([name, view]) => [name, share(built[name], draw(view))]))
      k.view.paused = false
      return { cover: share(built.opening, bare), faded }
    })
    check(seen.cover < (size === 'phone' ? 0.2 : 0.1), `${size}: the forest covers ${(100 * seen.cover).toFixed(1)}% of the opening view`)
    for (const [name, what] of [['opening', 'in the opening view'], ['forest', 'looking at the forest from the clearing'], ['trees', 'looking down at the clearing from among the trees']])
      check(seen.faded[name] < 0.005, `${size}: ${what}, ${(100 * seen.faded[name]).toFixed(1)}% of the frame is forest that fades with the camera`)
  }
  noErrors(errors, `in the ${size} woodland forest check`)
  await page.close()
}

// On the narrowest phones the clock is left out, so the masthead keeps its two rows. The years in
// the footer stay, and nothing scrolls sideways.
{
  const { page } = await open('', { width: 320, height: 640 })
  check((await page.$eval('.ltime', (b) => getComputedStyle(b).display)) === 'none', 'at 320px the clock in the masthead is shown')
  check((await page.$eval('.since', (b) => b.getBoundingClientRect().width)) > 0, 'at 320px the years in the footer are missing')
  check(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), 'at 320px the page scrolls sideways')
  await page.close()
}

// ---------- the karesansui garden ----------

/** Opens the garden from one of its doors in the footer, and returns its frame once it has drawn its first frame. */
async function garden(page, door = '.since') {
  await page.click(door)
  const frame = await page.waitForSelector('dialog.frdlg[open] iframe', { timeout: 5000 }).then((h) => h.contentFrame(), () => null)
  const ready = frame && (await frame.waitForFunction(() => window.__GARDEN__?.snapshot().ready, { timeout: 60000 }).then(() => true, () => false))
  return ready ? frame : null
}

/** A stroke across open sand, away from the stones and the spiral, one for each language. */
const STROKE = { en: -4.5, zh: 2.5 }
const stroke = (lang) => Array.from({ length: 12 }, (_, i) => ({ x: STROKE[lang] + i * 0.12, z: -3.3 }))
/** The height of the sand along a stroke, in millimetres. */
const sand = (frame, line) => frame.evaluate((pts) => pts.map((p) => __GARDEN__.sand.heightAt(p.x, p.z) * 1000), line)
const deepest = (a, b) => Math.max(...a.map((h, i) => Math.abs(h - b[i])))

// The years at the foot of every page open the garden in the page's language, and its own clock
// shows the same time as the one in the masthead. Escape first ends raking and only then closes
// the garden, and the page stays where the reader was. A stroke raked just before the garden
// closes is still in the sand when it opens again, because the garden saves before its frame goes.
for (const lang of ['en', 'zh']) {
  const { page, errors } = await open('', { lang, counts: true })
  await page.evaluate(() => window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' }))
  const top = await page.evaluate(() => scrollY)
  const frame = await garden(page)
  check(frame, `${lang}: pressing the years in the footer did not open the garden`)
  check(String(await counted(page)) === 'egg-garden-years', `${lang}: opening the garden counted ${await counted(page)}`)
  if (frame) {
    const label = await frame.$eval('#homeBtn span', (s) => s.textContent)
    check(label === GARDEN[lang].home, `${lang}: the garden's button reads "${label}", expected "${GARDEN[lang].home}"`)
    await frame.click('#clockBtn')
    // The two clocks may read either side of the turn of a minute, so a difference is read again a second later.
    const differ = async () => {
      const pair = [await page.$eval('.ltime', (b) => b.textContent), await frame.$eval('#timeLabel', (t) => t.textContent)]
      return pair[0] === pair[1] ? null : pair.join(' and ')
    }
    await wait(300)
    const twice = (await differ()) && (await wait(1200), await differ())
    check(!twice, `${lang}: the clock in the masthead and the garden's clock show different times, ${twice}`)

    const line = stroke(lang)
    check(await frame.evaluate((pts) => pts.every((p) => __GARDEN__.world.allowed(p.x, p.z)), line), `${lang}: the test stroke crosses ground the garden does not let a visitor rake`)
    const before = await sand(frame, line)
    await frame.evaluate((pts) => __GARDEN__.mark(pts), line)
    const raked = await sand(frame, line)
    check(deepest(raked, before) > 1, `${lang}: the test stroke left no mark in the sand`)

    await page.keyboard.press('d')
    const raking = await frame.$eval('#drawBtn', (b) => b.getAttribute('aria-pressed'))
    await page.keyboard.press('Escape')
    await wait(400)
    const after = await frame.$eval('#drawBtn', (b) => b.getAttribute('aria-pressed'))
    const still = await page.evaluate(() => !!document.querySelector('dialog.frdlg')?.open)
    check(raking === 'true' && after === 'false' && still, `${lang}: Escape while raking should end the raking and keep the garden open (raking ${raking}, then ${after}, open ${still})`)
    await page.keyboard.press('Escape')
    const closed = await page.waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 4000 }).then(() => true, () => false)
    check(closed, `${lang}: Escape in the garden did not close it`)
    const moved = await page.evaluate((y) => Math.round(scrollY - y), top)
    check(moved === 0, `${lang}: the garden moved the page under it by ${moved}px`)

    const again = await garden(page)
    check(again, `${lang}: the garden did not open a second time`)
    if (again) {
      const kept = await sand(again, line)
      check(deepest(kept, raked) < 0.5, `${lang}: the stroke raked before closing changed by ${deepest(kept, raked).toFixed(2)} mm after reopening`)
      await page.click('.fr-close')
      const shut = await page.waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 4000 }).then(() => true, () => false)
      check(shut, `${lang}: the close button did not close the garden`)
    }
  }
  noErrors(errors, `in the ${lang} garden`)
  await page.close()
}

// The dish of raked sand beside the years shows a stone for each hour of the twelve-hour clock,
// as the garden does, and opens the garden too.
for (const lang of ['en', 'zh']) {
  const { page, errors } = await open('', { lang, counts: true })
  const dish = await page
    .waitForFunction(() => document.querySelector('.dish')?.dataset.hours, { timeout: 5000 })
    .then(() => page.$eval('.dish', (d) => ({ stones: d.querySelectorAll('.dish-stone.on').length })), () => null)
  const hours = await page.evaluate(() => new Date().getHours() % 12 || 12)
  check(dish?.stones === hours, `${lang}: the sand dish shows ${dish?.stones} stones at ${hours} o'clock`)
  // The dish and the clock in the masthead may read either side of the turn of a minute, so a difference is read again a second later.
  const named = async () => {
    const [label, time] = [await page.$eval('.dish', (d) => d.getAttribute('aria-label')), await page.$eval('.ltime', (b) => b.textContent)]
    return label?.endsWith(time) ? null : `"${label}" and ${time}`
  }
  const misnamed = (await named()) && (await wait(1200), await named())
  check(!misnamed, `${lang}: the sand dish and the clock in the masthead disagree, ${misnamed}`)
  await page.evaluate(() => window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' }))
  const frame = await garden(page, '.dish')
  check(frame, `${lang}: pressing the sand dish did not open the garden`)
  check(String(await counted(page)) === 'egg-garden-dish', `${lang}: opening the garden from the sand dish counted ${await counted(page)}`)
  if (frame) {
    await page.click('.fr-close')
    await page.waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 4000 }).catch(() => {})
  }
  noErrors(errors, `with the ${lang} sand dish`)
  await page.close()
}

// ---------- the cloister clock tower ----------

/** The newest article in each language, where the clock beside the reading time opens the tower. */
const ARTICLE = Object.fromEntries(['en', 'zh'].map((lang) => [lang, sorter(MAIL[lang]).days.flatMap((d) => MAIL[lang][d]).findLast((s) => s.kind === 'article').href]))

/** Opens the tower from the clock beside the reading time, and returns its frame once it has drawn its first frame. */
async function tower(page) {
  await page.click('.minread')
  const frame = await page.waitForSelector('dialog.frdlg[open] iframe', { timeout: 5000 }).then((h) => h.contentFrame(), () => null)
  const ready = frame && (await frame.waitForFunction(() => window.__CLOCKSTAIRS__?.snapshot.ready, { timeout: 60000 }).then(() => true, () => false))
  return ready ? frame : null
}

// The reading time of every article sits beside a small clock, the cloister clock tower seen from
// above, whose two bridges show the visitor's time. Pressing it opens the tower in the page's
// language, where the bridges keep the same time. Escape first closes the tower's settings and only
// then the tower, and a setting changed just before the tower closes is there when it opens again.
for (const lang of ['en', 'zh']) {
  const { page, errors } = await open('', { lang, counts: true, path: ARTICLE[lang] })
  const face = await page
    .waitForFunction(() => document.querySelector('.minread')?.dataset.time, { timeout: 5000 })
    .then(() => page.$eval('.minread', (b) => ({ time: b.dataset.time, hour: b.querySelector('.tower-hour').getAttribute('transform'), minute: b.querySelector('.tower-minute').getAttribute('transform') })), () => null)
  const [h, m] = (face?.time ?? '').split(':').map(Number)
  check(face?.hour === `rotate(${((h * 60 + m) / 2) % 360} 8 8)` && face?.minute === `rotate(${m * 6} 8 8)`, `${lang}: the clock beside the reading time draws ${JSON.stringify(face)}`)
  // The two clocks may read either side of the turn of a minute, so a difference is read again a second later.
  const apart = async (read) => {
    const pair = [await page.$eval('.ltime', (b) => b.textContent), await read()]
    return pair[0] === pair[1] ? null : pair.join(' and ')
  }
  const beside = () => page.$eval('.minread', (b) => b.dataset.time)
  const off = (await apart(beside)) && (await wait(1200), await apart(beside))
  check(!off, `${lang}: the clock in the masthead and the clock beside the reading time show different times, ${off}`)
  const top = await page.evaluate(() => scrollY)
  const frame = await tower(page)
  check(frame, `${lang}: pressing the clock beside the reading time did not open the tower`)
  check(String(await counted(page)) === 'egg-tower', `${lang}: opening the tower counted ${await counted(page)}`)
  if (frame) {
    const words = await frame.evaluate(() => [document.querySelector('h1').textContent, document.getElementById('orbitBtn').textContent])
    check(words.join() === [TOWER[lang].title, TOWER[lang].orbit].join(), `${lang}: the tower reads ${words.join(' and ')}`)
    const inside = () => frame.evaluate(() => window.__CLOCKSTAIRS__.snapshot.time)
    const late = (await apart(inside)) && (await wait(1200), await apart(inside))
    check(!late, `${lang}: the clock in the masthead and the tower show different times, ${late}`)

    await frame.$eval('#settingsBtn', (b) => b.click())
    await frame.select('#weatherSelect', 'snow')
    await page.keyboard.press('Escape')
    await wait(300)
    const settings = await frame.$eval('#settings', (s) => s.hidden)
    const still = await page.evaluate(() => !!document.querySelector('dialog.frdlg')?.open)
    check(settings && still, `${lang}: Escape should close the tower's settings and keep the tower open (settings hidden ${settings}, open ${still})`)
    await page.keyboard.press('Escape')
    const closed = await page.waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 4000 }).then(() => true, () => false)
    check(closed, `${lang}: Escape in the tower did not close it`)
    const moved = await page.evaluate((y) => Math.round(scrollY - y), top)
    check(moved === 0, `${lang}: the tower moved the page under it by ${moved}px`)

    const again = await tower(page)
    check(again, `${lang}: the tower did not open a second time`)
    if (again) {
      const weather = await again.$eval('#weatherSelect', (s) => s.value)
      check(weather === 'snow', `${lang}: the weather chosen just before the tower closed came back as ${weather}`)
      await again.select('#weatherSelect', 'auto')
      await page.click('.fr-close')
      const shut = await page.waitForFunction(() => !document.querySelector('dialog.frdlg'), { timeout: 4000 }).then(() => true, () => false)
      check(shut, `${lang}: the close button did not close the tower`)
    }
  }
  noErrors(errors, `in the ${lang} tower`)
  await page.close()
}

// On the narrowest phones the clock beside the reading time stays, and the article does not scroll sideways.
{
  const { page } = await open('', { width: 320, height: 640, path: ARTICLE.en })
  check((await page.$eval('.minread .tower', (s) => s.getBoundingClientRect().width)) > 0, 'at 320px the clock beside the reading time is missing')
  check(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), 'at 320px the article scrolls sideways')
  await page.close()
}

// On a phone the town and its paper fit without scrolling sideways.
for (const width of [390, 320]) {
  const post = sorter(MAIL.en)
  const day = post.days.find((d) => MAIL.en[d].some((s) => s.kind === 'note' && s.html.length > 1500)) ?? post.days.at(-1)
  const { page, errors } = await open(`clock-open&clock-time=${day}T10:00&clock-pause`, { width, height: 800 })
  await arrival(page, day)
  await page.click('.mu-man')
  await page.waitForSelector('dialog.np[open]', { timeout: 4000 })
  await wait(1000)
  const wide = await page.evaluate(() => {
    const over = (el) => el && el.scrollWidth > el.clientWidth + 1
    return { page: document.documentElement.scrollWidth > innerWidth + 1, town: over(document.querySelector('dialog.clockdlg')), paper: over(document.querySelector('dialog.np')) }
  })
  check(!wide.page && !wide.town && !wide.paper, `at ${width}px something scrolls sideways: ${JSON.stringify(wide)}`)
  noErrors(errors, `at ${width}px`)
  await page.close()
}

if (WANT_SHOTS) {
  fs.mkdirSync(SHOTS, { recursive: true })
  const newest = sorter(MAIL.en).days.at(-1)
  const note = sorter(MAIL.zh).days.findLast((d) => MAIL.zh[d].some((s) => s.kind === 'note'))
  const shots = [
    ['town-dark', 'en', {}, false],
    ['town-light', 'en', { theme: 'light' }, false],
    ['town-zh', 'zh', {}, false],
    ['town-phone', 'en', { width: 390, height: 844 }, false],
    ['town-crossing', 'en', {}, false, D.shift(D.dayOf(D.parts(newest).y, D.parts(newest).m, 1), 1)],
    ['paper', 'en', {}, true],
    ['paper-light', 'en', { theme: 'light' }, true],
    ['paper-zh', 'zh', {}, true],
    ['paper-note-zh', 'zh', {}, true, note],
    ['paper-phone', 'en', { width: 390, height: 844 }, true],
  ]
  for (const [name, lang, opts, paper, day = newest] of shots) {
    const { page } = await open(`clock-open&clock-time=${day}T10:00&clock-pause`, { lang, ...opts })
    await page.evaluate(() => document.fonts.ready)
    await arrival(page, day)
    if (paper) {
      await page.click('.mu-man')
      await page.waitForSelector('dialog.np[open]')
      await page.evaluate(() => document.fonts.ready)
    }
    await wait(1500)
    const el = paper || opts.width ? null : await page.$('dialog.clockdlg')
    const file = path.join(SHOTS, `${name}.png`)
    if (el) await el.screenshot({ path: file })
    else await page.screenshot({ path: file })
    await page.close()
  }
  {
    const { page } = await open('clock-time=14:37&clock-pause', { path: LOST })
    await clock(page)
    await wait(500)
    await page.screenshot({ path: path.join(SHOTS, 'lost.png') })
    await page.close()
  }
  {
    const { page } = await open('eggs=now')
    await page.waitForSelector('.egg-walker')
    await wait(5000)
    await page.screenshot({ path: path.join(SHOTS, 'walk.png') })
    await page.close()
  }
  {
    const { href } = MAIL.en[newest].find((s) => s.kind === 'article')
    const { page } = await open('eggs=now', { path: href })
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))
    await page.waitForSelector('.egg-letter')
    await wait(1500)
    await page.screenshot({ path: path.join(SHOTS, 'letter.png') })
    await page.close()
  }
  for (const [name, lang, opts] of [['woodland', 'en', {}], ['woodland-phone', 'zh', { width: 390, height: 844 }]]) {
    const { page } = await open('', { lang, ...opts })
    if (await woodland(page)) {
      await wait(2500)
      await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
    }
    await page.close()
  }
  for (const [name, lang, opts] of [['garden', 'en', {}], ['garden-phone', 'zh', { width: 390, height: 844 }]]) {
    const { page } = await open('', { lang, ...opts })
    if (await garden(page)) {
      await wait(2500)
      await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
    }
    await page.close()
  }
  for (const [name, lang, opts] of [['tower', 'en', {}], ['tower-phone', 'zh', { width: 390, height: 844 }]]) {
    const { page } = await open('', { lang, path: ARTICLE[lang], ...opts })
    if (await tower(page)) {
      await wait(4000)
      await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
    }
    await page.close()
  }
  console.log(`screenshots -> ${path.relative(process.cwd(), SHOTS)}`)
}

await browser.close()
server.close()

console.log(`\n${checks} checks`)
if (problems.length) {
  console.log(`\n${problems.length} problem(s):`)
  for (const p of problems) console.log('  ' + p)
  process.exitCode = 1
} else console.log('clean: the clock drew the time it was given, and the town brought the right mail')
