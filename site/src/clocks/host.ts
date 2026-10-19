/**
 * The postman's town at the scale of days, which opens over any page, and the postman clock,
 * which stands on the page that does not exist. Both load only when an Easter egg asks for them.
 *
 * The clock is drawn only while the page is visible, so a clock in a background tab does no
 * work and simply shows the right time again when the visitor comes back.
 */
import type { Lang } from '../site.config'
import { adopt, fill, html, svg, type Scene } from './scene'
import { TimeSource, civilToTs, localZone, type Snapshot } from './time'
import type { Day } from './museum/days'
import type { Mail } from './museum/mail'
import type { Museum } from './museum/index'
import UI from './copy'
import TEXT from './scenes/postman/copy'
import css from './host.css?inline'

const motion = matchMedia('(prefers-reduced-motion: reduce)')
/** The clock's drawing code, its styles and the rest of its words load only when the clock is first shown. */
const load = () => import('./scenes/postman/index')
/** The town and its mail load only when it is first opened. */
const loadTown = () => import('./museum/index')
const loadMail = (lang: Lang): Promise<Mail> => fetch(`/${lang}/postman.json`).then((r) => (r.ok ? r.json() : {}))

interface View {
  show(): Promise<void>
  start(): void
  stop(): void
  /** Draws the present state at once, with no transition. */
  jump(): void
}

/**
 * The clock drawn into one element: every second, or every frame while the scene says
 * something is moving.
 */
function view(root: HTMLElement, lang: Lang, time: TimeSource, report: (t: Snapshot, scene: Scene) => void): View {
  let scene: Scene | null = null
  let token = 0
  let running = false
  let jumping = true
  let timer = 0
  let frame = 0
  let last: { ts: number; at: number; rate: number } | null = null

  const clear = () => {
    clearTimeout(timer)
    cancelAnimationFrame(frame)
    timer = frame = 0
  }

  function tick() {
    clear()
    if (!scene || !running) return
    const t = time.snapshot()
    const at = performance.now()
    // Time that moved further than the wall clock did means a jump, because the computer
    // slept, its clock was set, or a preview moved it. The scene then redraws without a flourish.
    if (last && Math.abs(t.ts - (last.ts + (at - last.at) * last.rate)) > 1000) jumping = true
    last = { ts: t.ts, at, rate: time.rate }
    const moving = scene.draw(t, jumping)
    jumping = false
    if (!('ready' in root.dataset)) root.dataset.ready = ''
    report(t, scene)
    if (moving) frame = requestAnimationFrame(tick)
    else timer = window.setTimeout(tick, time.rate ? 1004 - (t.ts % 1000) : 1000)
  }

  return {
    async show() {
      const mine = ++token
      const { default: create } = await load()
      if (mine !== token) return
      scene?.destroy()
      root.replaceChildren()
      delete root.dataset.ready
      scene = create({ root, still: motion.matches, lang })
      jumping = true
      last = null
      tick()
    },
    start() {
      if (running) return
      running = true
      jumping = true
      last = null
      tick()
    },
    stop() {
      running = false
      clear()
    },
    jump() {
      jumping = true
      if (running) tick()
    },
  }
}

/** Reads preview settings from the address, for example ?clock-time=14:37&clock-pause. */
function preview(time: TimeSource, params: URLSearchParams) {
  const zone = params.get('clock-zone')
  if (zone) {
    try {
      new Intl.DateTimeFormat('en-US', { timeZone: zone })
      time.setZone(zone)
    } catch {
      // Not a zone this browser knows; stay in the reader's own.
    }
  }
  const at = params.get('clock-time')
  const ts = at ? parseTime(at, time) : null
  if (ts !== null) time.seek(ts)
  if (params.has('clock-pause')) time.pause(true)
  return Boolean(zone || at || params.has('clock-pause'))
}

/** HH:MM[:SS] today, YYYY-MM-DDTHH:MM[:SS] in the clock's zone, or any instant Date can parse. */
function parseTime(text: string, time: TimeSource) {
  const clock = /^(\d{1,2}):(\d{2})(?::(\d{2}))?$/.exec(text)
  if (clock) return civilToTs(time.snapshot().date, +clock[1], +clock[2], +(clock[3] ?? 0), time.zone)
  const civil = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(text)
  if (civil) return civilToTs(civil[1], +civil[2], +civil[3], +(civil[4] ?? 0), time.zone)
  const instant = Date.parse(text)
  return Number.isNaN(instant) ? null : instant
}

/** The visitor's clock, or the moment a preview address asks for. */
const time = new TimeSource()
const previewing = preview(time, new URLSearchParams(location.search))

interface Shell {
  dialog: HTMLDialogElement
  stage: HTMLElement
}

let shell: Shell | null = null
let town: Museum | null = null
let opening = 0
let mail: Promise<Mail> | null = null

/** The dialog the town opens in, made the first time a visitor finds the town. */
function build(lang: Lang): Shell {
  adopt(css)
  const stage = html('div', { class: 'cd-stage' })
  const shut = html(
    'button',
    { class: 'cd-close', type: 'button', 'aria-label': UI[lang].close },
    svg('svg', { viewBox: '0 0 12 12', width: 12, height: 12, 'aria-hidden': 'true' }, svg('path', { d: 'M2 2L10 10M10 2L2 10' })),
  )
  const dialog = html('dialog', { class: 'clockdlg', 'aria-label': TEXT[lang].name }, stage, shut)
  document.body.append(dialog)

  dialog.addEventListener('close', () => {
    opening++
    town?.destroy()
    town = null
  })
  // The stage fills the dialog, so a click that lands on the dialog itself is on the backdrop.
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close()
  })
  shut.addEventListener('click', () => dialog.close())
  motion.addEventListener('change', () => {
    if (town) enter({ dialog, stage }, lang, town.day()).catch(() => dialog.close())
  })
  return { dialog, stage }
}

/** Builds the town in the open dialog, with the postman at `start` or else today. */
async function enter({ dialog, stage }: Shell, lang: Lang, start?: Day) {
  const mine = ++opening
  mail ??= loadMail(lang).catch(() => ({}))
  const [{ default: create }, letters] = await Promise.all([loadTown(), mail])
  if (mine !== opening || !dialog.open) return
  town?.destroy()
  town = create({ root: stage, lang, today: time.snapshot().date, mail: letters, still: motion.matches, start })
}

/** Opens the postman's town over the page, with him at `start`, or at today. */
export function openTown(lang: Lang, start?: Day) {
  const opened = (shell ??= build(lang))
  if (opened.dialog.open) return
  opened.dialog.showModal()
  enter(opened, lang, start).catch(() => opened.dialog.close())
}

/**
 * Draws the postman clock into `face`, a button that stays hidden until the clock is drawn. The
 * hour is his street and the minute the house he delivers to. Pressing it opens the town.
 */
export function mountClock(face: HTMLElement, lang: Lang) {
  adopt(css)
  // A visitor who travels across zones gets their new zone at the next minute.
  let minute = -1
  const clock = view(face, lang, time, (t, scene) => {
    if (t.minuteIndex !== minute) {
      minute = t.minuteIndex
      if (!previewing && localZone() !== time.zone) time.setZone(localZone())
    }
    const label = fill(UI[lang].label, { name: TEXT[lang].name, reading: scene.reading(t) })
    if (face.getAttribute('aria-label') !== label) face.setAttribute('aria-label', label)
  })
  const sync = () => (document.visibilityState === 'visible' ? clock.start() : clock.stop())
  // Without its drawing code the clock would be an empty frame, so it stays hidden.
  const fail = () => (face.hidden = true)

  face.addEventListener('click', () => openTown(lang))
  time.onChange(() => clock.jump())
  document.addEventListener('visibilitychange', sync)
  motion.addEventListener('change', () => clock.show().catch(fail))
  clock.show().then(() => {
    face.hidden = false
    sync()
  }, fail)
}
