/**
 * The postman's town at the scale of days, in the clock's larger view. Each house is a day
 * and each street a month, and the mailbox of a day with an article or a short post has
 * its flag up. The reader moves the postman by picking a house, by dragging the street or
 * with the keys. At a day with mail he takes it out of the mailbox, and handing it over
 * opens that day's newspaper. With empty hands he sets off for the nearest earlier day
 * that has mail.
 */
import type { Lang } from '../../site.config'
import { adopt, asButton, fill, html, seeded, svg, uid } from '../scene'
import {
  FAR_TILE,
  FEET,
  FIRST,
  GROUND,
  KERB,
  LAST,
  PITCH,
  REACH,
  drawCorner,
  drawFar,
  drawGround,
  drawHouse,
  drawLamp,
  drawPostman,
  plan,
  stride as walk,
  type House,
} from '../scenes/postman/town'
import phrases from '../scenes/postman/phrases'
import townCss from '../scenes/postman/style.css?inline'
import * as cast from './cast'
import copy from './copy'
import { datestamp, WIDTH as STAMP } from './datestamp'
import { between, dayOf, festival, hasCat, monthLength, nearestDay, parts, plotOf, shift, spotAt, type Day } from './days'
import { sorter, type Mail, type Story } from './mail'
import { ink, openPaper, type Paper } from './paper'
import { paintSky } from './sky'
import css from './style.css?inline'

export interface MuseumEnv {
  /** The element the town is drawn into. */
  root: HTMLElement
  lang: Lang
  /** The reader's date. The houses of later days are not built yet. */
  today: Day
  mail: Mail
  /** The reader prefers reduced motion: the postman moves at once and nothing moves by itself. */
  still: boolean
  /** Where the postman starts, if not today. */
  start?: Day
}

export interface Museum {
  /** The day the postman is at or walking to. */
  day(): Day
  destroy(): void
}

const HEIGHT = 240
/** From this width the view is wider than the postman clock and shows more houses. */
const WIDE = 560
const STRIDE = 400
/** The hills behind the town move this much slower than the street, so they read as far away. */
const DEPTH = 0.28
const SCALE = 0.92
/** How far behind the postman the dog keeps. */
const HEEL = 17
/** A trip longer than this many plots is made as a time lapse rather than on foot. */
const FAR = 45
const TRAVEL = 2600
/** The angle of the arm that holds up the mail. */
const OFFER = -64
/** How far from the middle of a garden by a crossing its lamppost stands, toward the crossing. */
const CORNER = 34

/** Days whose mail the reader has been handed, kept while the page stays open. */
const read = new Set<Day>()

const mod = (a: number, n: number) => ((a % n) + n) % n
/** Removes an element and cancels the animations in it, which would otherwise go on running after it is gone. */
function discard(el: Element) {
  for (const a of el.getAnimations({ subtree: true })) a.cancel()
  el.remove()
}
/** A street's seed for the shapes of its houses. */
const street = (y: number, m: number) => y * 100 + m
const planOf = (day: Day) => {
  const { y, m, d } = parts(day)
  return plan(street(y, m), d)
}

/** The clock's easing, cubic-bezier(.45, .05, .35, 1), for movement drawn frame by frame. */
const ease = (() => {
  const curve = (a: number, b: number) => (t: number) => 3 * (1 - t) * (1 - t) * t * a + 3 * (1 - t) * t * t * b + t * t * t
  const x = curve(0.45, 0.35)
  const y = curve(0.05, 1)
  return (k: number) => {
    if (k <= 0) return 0
    if (k >= 1) return 1
    let lo = 0
    let hi = 1
    for (let i = 0; i < 22; i++) {
      const mid = (lo + hi) / 2
      if (x(mid) < k) lo = mid
      else hi = mid
    }
    return y((lo + hi) / 2)
  }
})()

/** Something that moves the street or the postman, frame by frame. */
interface Motion {
  /** Moves things on to `now`. False once the motion is over. */
  step(now: number): boolean
  /** Runs when the motion comes to its end by itself. */
  end(): void
}

export default function create(env: MuseumEnv): Museum {
  adopt(townCss)
  adopt(css)
  const T = copy[env.lang]
  const months = phrases[env.lang].months.split(' ')
  const post = sorter(env.mail)
  const today = env.today
  const since = parts(post.days[0] ?? today).y
  const START = dayOf(since, 1, 1)
  const END = dayOf(parts(today).y + 1, 12, 31)
  const LO = plotOf(START)
  const HI = plotOf(END)
  const clamp = (day: Day) => (day < START ? START : day > END ? END : day)
  const long = new Intl.DateTimeFormat(T.locale, { dateStyle: 'long', timeZone: 'UTC' })
  const spoken = (day: Day) => {
    const { y, m, d } = parts(day)
    return long.format(Date.UTC(y, m - 1, d))
  }
  const ids = { sky: uid('mu-sky'), glow: uid('mu-glow'), moon: uid('mu-moon'), print: uid('mu-print') }
  const paint = { glow: ids.glow }
  const say = html('p', { class: 'mu-say', 'aria-live': 'polite' })

  // The view, rebuilt when the width crosses WIDE.
  let W = 0
  let CENTRE = 0
  let root: SVGSVGElement
  let row: SVGGElement
  let far: SVGGElement
  let night: SVGGElement
  let daylight: SVGGElement
  let shade: SVGRectElement
  let fx: SVGGElement
  let fireworks: SVGGElement
  let road: SVGGElement
  let man: ReturnType<typeof drawPostman>
  let face: SVGGElement
  let hit: SVGGElement
  let held: { letter: SVGGElement; parcel: SVGGElement }
  let dog: cast.Dog
  let stamp: SVGPathElement
  const plots = new Map<number, SVGGElement>()
  /** The pigeons on the lampposts of the plots drawn. */
  const flocks = new Map<number, cast.Bird[]>()

  // Where things are. `at` is the plot under the postman, a fraction while the street moves.
  let at = plotOf(clamp(env.start ?? today))
  let origin = at
  let farPhase = at * PITCH * DEPTH
  let manX = 0
  let dogX = 0
  let facing = 1
  let dogFacing = 1
  /** The day the postman is standing at, or null while he is on his way. */
  let here: Day | null = null
  /** The day he is on his way to. */
  let dest: Day | null = null
  let holding: Story[] | null = null
  let motion: Motion | null = null
  let travelling = false
  /** The date the stamp shows instead of the street's, while it counts through the days of a long trip. */
  let riffle: Day | null = null
  let paper: Paper | null = null

  let raf = 0
  let last = 0
  const timers = new Set<number>()
  const later = (fn: () => void, wait: number) => {
    const id = window.setTimeout(() => {
      timers.delete(id)
      fn()
    }, wait)
    timers.add(id)
  }
  /** Motion that loops until it is cancelled, which a reader who prefers stillness never gets. */
  const loop = (el: Element, keys: Keyframe[], opts: KeyframeAnimationOptions) =>
    env.still ? null : el.animate(keys, { iterations: Infinity, ...opts })

  const manAt = (day: Day) => CENTRE + planOf(day).mailX - REACH
  const homeOf = (day: Day) => plots.get(plotOf(day))?.querySelector<SVGGElement>('[data-house]') ?? null
  const flagOf = (day: Day) => homeOf(day)?.querySelector('.mu-flag') ?? null

  // ---------- the drawing ----------

  function build() {
    W = env.root.clientWidth >= WIDE ? 640 : 360
    CENTRE = W / 2
    env.root.style.aspectRatio = `${W} / ${HEIGHT}`
    plots.clear()
    flocks.clear()
    const stars = seeded('mu-stars')
    far = svg('g', {}, drawFar(), svg('g', { transform: `translate(${3 * FAR_TILE} 0)` }, drawFar()))
    row = svg('g', { class: 'mu-row' })
    man = drawPostman(SCALE)
    face = svg('g', {})
    face.append(man.root.lastChild!)
    man.root.append(face)
    cast.dress(man)
    held = { letter: man.letter, parcel: cast.heldParcel() }
    man.letter.after(held.parcel)
    held.letter.style.display = held.parcel.style.display = 'none'
    hit = svg(
      'g',
      { class: 'mu-man' },
      svg('ellipse', { cx: 0.5, cy: -17, rx: 13, ry: 23, class: 'mu-ring' }),
      svg('rect', { x: -12, y: -42, width: 25, height: 46, class: 'mu-hit' }),
    )
    man.root.append(hit)
    asButton(hit, T.postman, onPostman)
    dog = cast.drawDog()
    stamp = svg('path', { class: 'mu-datestamp', filter: `url(#${ids.print})` })
    fireworks = svg('g', {})
    shade = svg('rect', { width: W, height: HEIGHT, class: 'mu-shade' })
    fx = svg('g', { class: 'mu-fx' })
    road = svg('g', {})
    const clouds = svg(
      'g',
      {},
      ...[0, 300].map((x) =>
        svg(
          'g',
          { transform: `translate(${x} 0)` },
          svg('path', { d: 'M136 34H186C186 28 180 25 174 27C171 20 160 19 156 26C149 23 141 27 142 34Z', class: 'pm-cloud' }),
          svg('path', { d: 'M212 66H248C248 61 243 59 239 61C236 56 228 56 226 61C220 59 214 61 212 66Z', class: 'pm-cloud' }),
        ),
      ),
    )
    night = svg(
      'g',
      { class: 'pm-night' },
      ...Array.from({ length: W / 20 }, () =>
        svg('circle', { cx: (stars() * W).toFixed(1), cy: (4 + stars() * 96).toFixed(1), r: (0.45 + stars() * 0.6).toFixed(2), class: 'pm-star' }),
      ),
      svg('circle', { cx: W - 60, cy: 34, r: 26, fill: `url(#${ids.moon})` }),
      svg('path', { d: `M${W - 60} 22A12 12 0 0 1 ${W - 60} 46A7 12 0 0 0 ${W - 60} 22Z`, class: 'pm-moon' }),
    )
    daylight = svg('g', { class: 'pm-day' }, svg('circle', { cx: W - 60, cy: 36, r: 19, class: 'pm-sun-halo' }), svg('circle', { cx: W - 60, cy: 36, r: 12, class: 'pm-sun' }), clouds)
    root = svg(
      'svg',
      { viewBox: `0 0 ${W} ${HEIGHT}`, class: 'pm mu', role: 'application', 'aria-label': T.scene, tabindex: '0' },
      svg(
        'defs',
        {},
        svg(
          'linearGradient',
          { id: ids.sky, x1: 0, y1: 0, x2: 0, y2: 1 },
          svg('stop', { offset: 0, 'stop-color': 'var(--pm-sky-top)' }),
          svg('stop', { offset: 0.62, 'stop-color': 'var(--pm-sky-mid)' }),
          svg('stop', { offset: 0.8, 'stop-color': 'var(--pm-sky-low)' }),
        ),
        svg(
          'radialGradient',
          { id: ids.glow },
          svg('stop', { offset: 0, 'stop-color': 'var(--pm-lit)', 'stop-opacity': 0.5 }),
          svg('stop', { offset: 1, 'stop-color': 'var(--pm-lit)', 'stop-opacity': 0 }),
        ),
        svg(
          'radialGradient',
          { id: ids.moon },
          svg('stop', { offset: 0, 'stop-color': 'var(--pm-moon)', 'stop-opacity': 0.22 }),
          svg('stop', { offset: 1, 'stop-color': 'var(--pm-moon)', 'stop-opacity': 0 }),
        ),
        // The light of the printed figures runs a little into the film around them.
        svg(
          'filter',
          { id: ids.print, x: '-10%', y: '-30%', width: '120%', height: '160%' },
          svg('feGaussianBlur', { stdDeviation: 0.6, result: 'bleed' }),
          svg('feMerge', {}, svg('feMergeNode', { in: 'bleed' }), svg('feMergeNode', { in: 'SourceGraphic' })),
        ),
      ),
      svg(
        'g',
        { 'aria-hidden': 'true' },
        svg('rect', { width: W, height: HEIGHT, fill: `url(#${ids.sky})` }),
        night,
        daylight,
        svg('g', { class: 'mu-big-moon' }, svg('circle', { cx: W - 60, cy: 34, r: 30, class: 'mu-moon-halo' }), svg('circle', { cx: W - 60, cy: 34, r: 17, class: 'mu-moon-disc' })),
        fireworks,
        far,
        svg('rect', { width: W, height: HEIGHT, class: 'mu-overcast' }),
        shade,
        svg('rect', { x: 0, y: GROUND, width: W, height: KERB - GROUND, class: 'pm-pave' }),
        svg('rect', { x: 0, y: KERB, width: W, height: 3, class: 'pm-kerb' }),
        svg('rect', { x: 0, y: KERB + 3, width: W, height: HEIGHT - KERB - 3, class: 'pm-road' }),
        svg('rect', { x: 0, y: KERB + 3, width: W, height: 1.6, class: 'pm-gutter' }),
      ),
      row,
      dog.root,
      man.root,
      road,
      fx,
      svg('g', { transform: `translate(${W - 12 - STAMP} ${HEIGHT - 16}) skewX(-8)`, 'aria-hidden': 'true' }, stamp),
    )
    env.root.replaceChildren(root, say)
    loop(clouds, [{ transform: 'translateX(-30px)' }, { transform: 'translateX(30px)' }], { duration: 60000, direction: 'alternate', easing: 'ease-in-out' })
    loop(dog.tail, [{ transform: 'rotate(-16deg)' }, { transform: 'rotate(18deg)' }], { duration: 260, direction: 'alternate', easing: 'ease-in-out' })
    listen()
  }

  function addPlot(p: number) {
    const spot = spotAt(p)
    const g = svg('g', { transform: `translate(${(p - origin) * PITCH} ${GROUND})`, 'data-plot': p })
    g.append(drawGround(spot.gap === 1))
    if (spot.gap === undefined) g.append(house(spot.day))
    else {
      // The town draws a street's corners from the house numbers past its ends.
      g.append(drawCorner(spot.gap === 0 ? LAST + 1 : spot.gap === 1 ? LAST + 2 : FIRST - 1, street(spot.y, spot.m)))
      if (spot.gap !== 1) g.append(corner(p, spot.y, spot.m, spot.gap))
    }
    // A later plot is drawn over an earlier one, so a tree in a gap stays behind the next house.
    let next: SVGGElement | null = null
    let nextAt = Infinity
    for (const [q, el] of plots) {
      if (q > p && q < nextAt) {
        nextAt = q
        next = el
      }
    }
    if (next) next.before(g)
    else row.append(g)
    plots.set(p, g)
  }

  /** Takes a plot off the street, with everything that moves on it. */
  function drop(p: number) {
    const g = plots.get(p)
    if (g) discard(g)
    plots.delete(p)
    flocks.delete(p)
  }

  /**
   * The lamppost on one corner of a crossing. Its blade names the street on that side: the
   * month that ends at the crossing, or the month that begins there. Pigeons sit on the second.
   */
  function corner(p: number, y: number, m: number, gap: 0 | 2) {
    const side = gap === 0 ? -1 : 1
    const named = gap === 0 ? { y, m } : parts(shift(dayOf(y, m, monthLength(y, m)), 1))
    const x = -side * CORNER
    const lamp = drawLamp(ids.glow, x, side)
    lamp.label.replaceChildren(`${months[named.m - 1]} `, svg('tspan', { class: 'pm-sign-num' }, String(named.y)))
    const g = svg('g', { transform: `translate(0 ${-GROUND})` }, lamp.root)
    if (side > 0) {
      const flock = cast.drawBirds([x + 14, x + 32, x + 60], 49)
      flock.root.addEventListener('click', (e) => {
        e.stopPropagation()
        scatter()
      })
      for (const b of flock.list) peck(b)
      flocks.set(p, flock.list)
      g.append(flock.root)
    }
    return g
  }

  /** The house of a day, with what that day brings: a flag for mail, a parcel for an article, a festival. */
  function house(day: Day) {
    const { d } = parts(day)
    const h = planOf(day)
    const g = drawHouse(d, h, paint)
    g.dataset.day = day
    if (day > today) {
      g.classList.add('mu-future')
      return g
    }
    if (day === here) g.setAttribute('data-dest', '')
    const stories = post.get(day)
    if (stories.length) {
      const flag = cast.drawFlag(h)
      if (read.has(day) || (day === here && holding)) flag.classList.add('mu-down')
      g.append(flag)
      if (stories.some((s) => s.kind === 'article')) g.append(cast.drawDoorParcel(h))
    }
    const fest = festival(day)
    if (fest === 'spring' || fest === 'spring-day' || fest === 'spring-eve') {
      const top = h.roof === 'flat' ? -h.height - 6 : -h.height
      for (const x of [-h.width / 2 + 5, h.width / 2 - 5]) {
        const lantern = cast.drawLantern(x, top, 0.8)
        g.append(lantern.root)
        loop(lantern.swing, [{ transform: 'rotate(-5deg)' }, { transform: 'rotate(5deg)' }], {
          duration: 1800 + seeded(`${day}${x}`)() * 600,
          direction: 'alternate',
          easing: 'ease-in-out',
        })
      }
    }
    if (fest === 'halloween') g.append(cast.drawPumpkin(h.mailX))
    if (fest === 'christmas') g.append(cast.drawWreath(h.doorX))
    if (hasCat(day)) g.append(cat(h))
    return g
  }

  /** A cat asleep on the roof of a house, who looks up when clicked. */
  function cat(h: House) {
    const c = cast.drawCat()
    const y = h.roof === 'flat' ? -h.height - 8 : -h.height - h.rise + 1.6
    const x = h.roof === 'hip' ? -5 : h.roof === 'flat' ? -h.width / 4 : 0
    c.root.setAttribute('transform', `translate(${x} ${y})`)
    loop(c.z, [{ transform: 'translate(0, 0)', opacity: 0 }, { opacity: 1, offset: 0.3 }, { transform: 'translate(4px, -9px)', opacity: 0 }], { duration: 2600 })
    loop(c.tail, [{ transform: 'rotate(0deg)' }, { transform: 'rotate(0deg)', offset: 0.8 }, { transform: 'rotate(-14deg)', offset: 0.9 }, { transform: 'rotate(0deg)' }], {
      duration: 3800,
    })
    c.root.addEventListener('click', (e) => {
      e.stopPropagation()
      c.root.classList.add('awake')
      c.head.animate([{ transform: 'translateY(0)' }, { transform: 'translateY(-1.4px)' }], { duration: 300, fill: 'forwards' })
      bubble(c.root, T.mew, -4, -14)
      later(() => c.root.classList.remove('awake'), 6000)
    })
    return c.root
  }

  /** Puts the street and the hills where `at` says, and prints the date. */
  function place() {
    if (Math.abs(at - origin) > 400) {
      origin = Math.round(at)
      for (const [p, g] of plots) g.setAttribute('transform', `translate(${(p - origin) * PITCH} ${GROUND})`)
    }
    row.style.transform = `translateX(${CENTRE - (at - origin) * PITCH}px)`
    const from = Math.floor(at - (CENTRE + PITCH) / PITCH)
    const to = Math.ceil(at + (W - CENTRE + PITCH) / PITCH)
    for (const p of plots.keys()) if (p < from || p > to) drop(p)
    for (let p = from; p <= to; p++) if (!plots.has(p)) addPlot(p)
    if (!travelling) farPhase = at * PITCH * DEPTH
    far.style.transform = `translateX(${-FAR_TILE - mod(farPhase, FAR_TILE)}px)`
    show(riffle ?? nearest(at, facing))
  }

  function pose() {
    man.root.style.transform = `translate(${manX}px, ${FEET}px)`
    face.style.transform = `scaleX(${facing})`
    dog.root.style.transform = `translate(${dogX}px, ${FEET}px)`
    dog.face.style.transform = `scaleX(${dogFacing})`
  }

  let shown = ''
  /** The date stamp in the corner, as on an old photo, shows the day of the house the postman is at or passing. */
  function show(day: Day) {
    if (day === shown) return
    shown = day
    stamp.setAttribute('d', datestamp(day))
    stamp.dataset.date = day
  }

  /** The day of the house nearest to plot `p`, within the town's span of days. */
  const nearest = (p: number, dir: number) => clamp(nearestDay(p, dir))

  // ---------- moving ----------

  function frame(now: number) {
    raf = 0
    const dt = Math.min(50, now - last)
    last = now
    if (motion && !motion.step(now)) {
      const done = motion
      motion = null
      done.end()
    }
    place()
    const trotting = follow(dt)
    pose()
    if (motion || trotting) raf = requestAnimationFrame(frame)
  }

  function wake() {
    if (raf) return
    last = performance.now()
    raf = requestAnimationFrame(frame)
  }

  let trot: Animation[] = []
  /** The dog keeps to the postman's heel and runs round him when he turns. */
  function follow(dt: number) {
    const want = manX - facing * HEEL
    if (env.still) {
      dogX = want
      dogFacing = facing
      return false
    }
    const gap = want - dogX
    const busy = Math.abs(gap) > 0.3
    if (busy) dogX += gap * (1 - Math.exp(-dt / 130))
    else dogX = want
    dogFacing = Math.abs(gap) > 3 ? Math.sign(gap) : facing
    const moving = busy || motion !== null || dragging
    if (moving && !trot.length && !env.still) {
      trot = dog.legs.map((leg, i) =>
        leg.animate([{ transform: 'rotate(22deg)' }, { transform: 'rotate(-22deg)' }], {
          duration: 180,
          iterations: Infinity,
          direction: 'alternate',
          delay: (i % 2) * 90,
        }),
      )
    } else if (!moving && trot.length) {
      for (const a of trot) a.cancel()
      trot = []
    }
    return busy
  }

  let strides: Animation[] = []
  /** Legs, arms and a slight rise and fall, a step every `beat` milliseconds until he stops. */
  function stride(beat: number) {
    halt()
    if (env.still) return
    strides = walk(man, beat)
  }

  function halt() {
    for (const a of strides) a.cancel()
    strides = []
  }

  const turn = (dir: number) => {
    if (dir) facing = Math.sign(dir)
  }

  /** Sends the postman to a day: on foot when it is near, as a time lapse when it is far. */
  function go(target: Day) {
    if (travelling) return
    const day = clamp(target)
    if (here === day && !motion) return knock()
    const to = plotOf(day)
    motion = null
    leave()
    dest = day
    if (env.still) return jump(day)
    const n = Math.abs(to - at)
    if (n > FAR) return travel(day)
    turn(to - at)
    const duration = Math.min(3600, 600 + 420 * n ** 0.75)
    stride(Math.max(140, Math.min(STRIDE, duration / (n * 2.2 + 1))))
    motion = tween(to, manAt(day), duration)
    wake()
  }

  function tween(to: number, manTo: number, duration: number): Motion {
    const from = at
    const manFrom = manX
    const t0 = performance.now()
    return {
      step(now) {
        const k = Math.min(1, (now - t0) / duration)
        const e = ease(k)
        at = from + (to - from) * e
        manX = manFrom + (manTo - manFrom) * e
        return k < 1
      },
      end: arrive,
    }
  }

  /** Straight to a day, with no walk, for a reader who prefers stillness. */
  function jump(day: Day) {
    dest = day
    at = plotOf(day)
    manX = manAt(day)
    arrive()
    wake()
  }

  /**
   * A time lapse to a far day: the street fades, the sky turns from day to night and back,
   * the hills race by and the date stamp counts through the days between.
   */
  function travel(day: Day) {
    const to = plotOf(day)
    const dir = Math.sign(to - at)
    const fromDay = shown
    const span = between(fromDay, day)
    const phase = farPhase
    const ahead = mod(to * PITCH * DEPTH - phase, FAR_TILE)
    const drift = dir > 0 ? ahead + 3 * FAR_TILE : ahead - 4 * FAR_TILE
    const manFrom = manX
    const manTo = manAt(day)
    const t0 = performance.now()
    let arrived = false
    travelling = true
    turn(dir)
    stride(150)
    scatter()
    const fade = row.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 420, fill: 'forwards' })
    const pulse = (k: number) => k % 2
    night.animate(Array.from({ length: 13 }, (_, k) => ({ opacity: pulse(k) })), { duration: TRAVEL })
    daylight.animate(Array.from({ length: 13 }, (_, k) => ({ opacity: 1 - pulse(k) })), { duration: TRAVEL })
    shade.animate([{ opacity: 0 }, ...Array.from({ length: 11 }, (_, k) => ({ opacity: k % 2 ? 0.55 : 0.05 })), { opacity: 0 }], {
      duration: TRAVEL,
      easing: 'ease-in-out',
    })
    motion = {
      step(now) {
        const k = Math.min(1, (now - t0) / TRAVEL)
        farPhase = phase + drift * ease(k)
        manX = manFrom + (manTo - manFrom) * ease(k)
        riffle = shift(fromDay, Math.round(span * ease(k)))
        if (!arrived && k >= 0.62) {
          // The street is rebuilt at the far day while it cannot be seen.
          arrived = true
          for (const p of plots.keys()) drop(p)
          at = origin = to
          world(day)
          fade.cancel()
          row.animate([{ opacity: 0 }, { opacity: 1 }], { duration: TRAVEL * 0.38, easing: 'ease-out' })
        }
        return k < 1
      },
      end() {
        travelling = false
        riffle = null
        arrive()
      },
    }
    wake()
  }

  /** Where the street comes to rest after a drag: at the nearest house, carried on by a flick. */
  function glide(speed: number) {
    let v = speed
    let t = performance.now()
    motion = {
      step(now) {
        const dt = Math.min(50, now - t)
        t = now
        if (Math.abs(v) < 0.004) return false
        at = Math.min(HI, Math.max(LO, at + v * dt))
        if (at === LO || at === HI) return false
        v *= Math.exp(-dt / 350)
        return true
      },
      end: settle,
    }
    wake()
  }

  /** Walks the last bit to the house nearest to where the street stopped. */
  function settle() {
    const day = nearest(at, facing)
    if (env.still) return jump(day)
    const to = plotOf(day)
    dest = day
    turn(to - at)
    motion = tween(to, manAt(day), 260 + 300 * Math.min(2, Math.abs(to - at)))
    wake()
  }

  /** The postman stops at a house. If its day has mail, he takes it out of the mailbox. */
  function arrive() {
    halt()
    const day = dest ?? nearest(at, facing)
    at = plotOf(day)
    dest = null
    here = day
    // He stands just before the mailbox, so he turns to face it.
    facing = 1
    place()
    homeOf(day)?.setAttribute('data-dest', '')
    world(day)
    decorate(day)
    if (post.get(day).length && day <= today) collect()
    else say.textContent = fill(T.arrived, { date: spoken(day) })
    idle()
  }

  /** The postman sets off: anything he holds goes back in the mailbox, and the day's figures leave. */
  function leave() {
    halt()
    if (here) {
      if (holding && !read.has(here)) flagOf(here)?.classList.remove('mu-down')
      homeOf(here)?.removeAttribute('data-dest')
    }
    release()
    for (const el of row.querySelectorAll('.mu-of-day')) discard(el)
    here = null
    dest = null
  }

  // ---------- the postman and the mail ----------

  let offer: Animation | null = null

  function collect() {
    if (!here) return
    const stories = post.get(here)
    holding = stories
    ink({ lang: env.lang, day: here, since, stories })
    const parcel = stories.some((s) => s.kind === 'article')
    held.parcel.style.display = parcel ? '' : 'none'
    held.letter.style.display = parcel ? 'none' : ''
    flagOf(here)?.classList.add('mu-down')
    man.arms[1].style.transform = `rotate(${OFFER}deg)`
    hit.setAttribute('aria-label', fill(T.holding, { date: spoken(here) }))
    say.textContent = fill(T.mail, { date: spoken(here) })
    if (env.still) return
    // He reaches up to the mailbox, takes the mail and holds it out.
    man.arms[1].animate(
      [
        { transform: 'rotate(0deg)' },
        { transform: 'rotate(-112deg)', offset: 0.4 },
        { transform: 'rotate(-112deg)', offset: 0.55 },
        { transform: `rotate(${OFFER}deg)` },
      ],
      { duration: 900, easing: 'ease-in-out' },
    )
    const item = parcel ? held.parcel : held.letter
    item.animate([{ opacity: 0 }, { opacity: 0, offset: 0.45 }, { opacity: 1, offset: 0.55 }, { opacity: 1 }], { duration: 900 })
    dog.body.animate(
      [
        { transform: 'translateY(0)' },
        { transform: 'translateY(-3.4px)', offset: 0.25 },
        { transform: 'translateY(0)', offset: 0.5 },
        { transform: 'translateY(-2.4px)', offset: 0.75 },
        { transform: 'translateY(0)' },
      ],
      { duration: 650, delay: 500 },
    )
    offer = man.arms[1].animate(
      [
        { transform: `rotate(${OFFER}deg)` },
        { transform: `rotate(${OFFER - 16}deg)`, offset: 0.1 },
        { transform: `rotate(${OFFER}deg)`, offset: 0.2 },
        { transform: `rotate(${OFFER}deg)` },
      ],
      { duration: 2600, delay: 1500, iterations: Infinity, easing: 'ease-in-out' },
    )
  }

  /** Empty hands again. */
  function release() {
    holding = null
    offer?.cancel()
    offer = null
    man.arms[1].style.transform = ''
    held.letter.style.display = held.parcel.style.display = 'none'
    hit.setAttribute('aria-label', T.postman)
  }

  /** He hands the mail over, and the day's paper opens. */
  function hand(from: Element) {
    if (!here || paper) return
    const day = here
    const stories = post.get(day)
    if (!stories.length || day > today) return
    const box = from.getBoundingClientRect()
    read.add(day)
    flagOf(day)?.classList.add('mu-down')
    release()
    paper = openPaper({
      host: env.root,
      lang: env.lang,
      day,
      since,
      stories,
      from: box,
      still: env.still,
      onClose: () => {
        paper = null
        idle()
      },
    })
  }

  function onPostman() {
    if (travelling || paper) return
    if (holding) return hand(holding.some((s) => s.kind === 'article') ? held.parcel : held.letter)
    const from = dest ?? here ?? nearest(at, facing)
    // From a day not built yet, the nearest earlier mail is the newest there is.
    go(post.before(from > today ? shift(today, 1) : from) ?? today)
  }

  /** The house the postman is already at: he takes its mail out again, or hands it over. */
  function knock() {
    if (holding) return onPostman()
    if (here && post.get(here).length && here <= today) collect()
  }

  // ---------- the world of the day ----------

  let worldDay = ''
  /** The season, the weather and any festival of the day the postman is at. */
  function world(day: Day) {
    if (day === worldDay) return
    worldDay = day
    paintSky({ root, fx, fireworks, width: W, still: env.still }, day, day > today)
  }

  /** The figures that come out while the postman is at a day: a reader on article days, a piano on music days. */
  function decorate(day: Day) {
    if (day > today) return
    const p = plotOf(day)
    const pastHouse = (q: number) => {
      const spot = spotAt(q)
      return spot.gap === undefined && spot.day <= today ? spot : null
    }
    const stories = post.get(day)
    if (stories.some((s) => s.kind === 'article')) {
      const q = pastHouse(p + 1) ? p + 1 : p - 1
      const home = pastHouse(q) && plots.get(q)?.querySelector('[data-house]')
      if (home) {
        const reader = cast.drawReader()
        reader.root.classList.add('mu-of-day')
        reader.root.setAttribute('transform', 'translate(-2 5)')
        home.append(reader.root)
        loop(
          reader.paper,
          [
            { transform: 'translateY(0)' },
            { transform: 'translateY(0)', offset: 0.86 },
            { transform: 'translateY(1.2px)', offset: 0.9 },
            { transform: 'translateY(0)', offset: 0.95 },
          ],
          { duration: 5200 },
        )
        reader.root.addEventListener('click', (e) => {
          e.stopPropagation()
          if (env.still) return hand(reader.paper)
          reader.paper
            .animate([{ transform: 'translateY(0)' }, { transform: 'translateY(4px)', offset: 0.5 }, { transform: 'translateY(0)' }], { duration: 500 })
            .finished.then(() => hand(reader.paper), () => {})
        })
      }
    }
    if (stories.some((s) => s.kind === 'article' && s.category === 'music')) {
      const home = homeOf(day)
      const h = planOf(day)
      const pane = h.panes.find((q) => q.y < -50) ?? h.panes[0]
      if (home && pane) {
        const notes = svg('g', { class: 'mu-of-day' }, ...['♪', '♫', '♪'].map((ch) => svg('text', { x: pane.x, y: pane.y + 4, class: 'mu-note' }, ch)))
        home.append(notes)
        ;[...notes.children].forEach((el, i) =>
          loop(
            el,
            [
              { transform: 'translate(0, 0)', opacity: 0 },
              { opacity: 1, offset: 0.25 },
              { transform: `translate(${i % 2 ? -7 : 8}px, -30px)`, opacity: 0 },
            ],
            { duration: 2600, delay: i * 870 },
          ),
        )
      }
    }
  }

  // ---------- the townsfolk ----------

  function bubble(parent: SVGGElement, text: string, x: number, y: number) {
    parent.querySelector('.mu-bubble')?.remove()
    const b = cast.drawBubble(text, x, y)
    parent.append(b)
    b.animate([{ opacity: 0, transform: 'translateY(2px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 220, fill: 'backwards' })
    later(() => b.remove(), 2200)
  }

  function peck(b: cast.Bird) {
    const r = Math.random
    loop(
      b.head,
      [{ transform: 'translate(0, 0)' }, { transform: 'translate(-0.7px, 1.7px)', offset: 0.05 }, { transform: 'translate(0, 0)', offset: 0.1 }, { transform: 'translate(0, 0)' }],
      { duration: 2400 + r() * 2200, delay: r() * 2000 },
    )
  }

  let away = false
  /** The pigeons fly off when the postman sets out, and come back a few seconds later. */
  function scatter() {
    if (env.still || away) return
    away = true
    for (const flock of flocks.values()) {
      flock.forEach((b, i) => {
        const dx = 40 + i * 14
        b.fly.animate(
          [
            { transform: 'translate(0, 0)', opacity: 1 },
            { transform: `translate(${dx * 0.4}px, -26px)`, opacity: 1, offset: 0.08 },
            { transform: `translate(${dx}px, -90px)`, opacity: 0, offset: 0.2 },
            { transform: `translate(${-dx}px, -80px)`, opacity: 0, offset: 0.78 },
            { transform: `translate(${-dx * 0.3}px, -20px)`, opacity: 1, offset: 0.92 },
            { transform: 'translate(0, 0)', opacity: 1 },
          ],
          { duration: 7000 + i * 600, delay: i * 90, easing: 'ease-in-out' },
        )
        b.wing.animate([{ transform: 'scaleY(1)' }, { transform: 'scaleY(-1.4)' }], { duration: 120, iterations: 58, direction: 'alternate', delay: i * 90 })
      })
    }
    later(() => (away = false), 8400)
  }

  /** Now and then someone rides past on a bicycle. */
  function cyclist() {
    later(cyclist, 14000 + Math.random() * 20000)
    if (motion || paper || document.hidden) return
    const c = cast.drawCyclist()
    const dir = Math.random() < 0.5 ? 1 : -1
    const y = 229
    const ends = [-30, W + 30]
    const [from, to] = dir > 0 ? ends : ends.reverse()
    road.append(c.root)
    for (const s of c.spins) s.animate([{ transform: 'rotate(0deg)' }, { transform: 'rotate(360deg)' }], { duration: 520, iterations: Infinity })
    c.legs.forEach((leg, i) =>
      leg.animate([{ transform: 'rotate(-28deg)' }, { transform: 'rotate(18deg)' }], { duration: 260, iterations: Infinity, direction: 'alternate', delay: i * 260 }),
    )
    c.root
      .animate([{ transform: `translate(${from}px, ${y}px) scale(${dir * 1.1}, 1.1)` }, { transform: `translate(${to}px, ${y}px) scale(${dir * 1.1}, 1.1)` }], {
        duration: 5200 + Math.random() * 1600,
      })
      .finished.then(
        () => discard(c.root),
        () => discard(c.root),
      )
  }

  let idler = 0
  /** At rest with nothing to hand over, the postman looks round now and then. */
  function idle() {
    clearTimeout(idler)
    if (env.still) return
    idler = window.setTimeout(() => {
      if (here && !holding && !paper && !motion) {
        face.animate(
          [
            { transform: `scaleX(${facing})` },
            { transform: `scaleX(${-facing})`, offset: 0.1 },
            { transform: `scaleX(${-facing})`, offset: 0.9 },
            { transform: `scaleX(${facing})` },
          ],
          { duration: 1700 },
        )
      }
      idle()
    }, 9000 + Math.random() * 6000)
  }

  // ---------- the reader's hands ----------

  let dragging = false
  let swallow = false
  let wheeling = 0

  /** Stops whatever the postman is doing so the reader can move the street by hand. */
  function grab() {
    motion = null
    leave()
    stride(300)
    scatter()
  }

  function listen() {
    root.addEventListener('pointerdown', (e) => {
      swallow = false
      if (e.button !== 0 || travelling || paper) return
      const x0 = e.clientX
      const at0 = at
      const units = W / root.getBoundingClientRect().width
      const trail: { t: number; at: number }[] = []
      const move = (ev: PointerEvent) => {
        if (!dragging) {
          if (Math.abs(ev.clientX - x0) < 6) return
          dragging = true
          root.setPointerCapture(ev.pointerId)
          root.classList.add('mu-dragging')
          grab()
        }
        const raw = at0 - ((ev.clientX - x0) * units) / PITCH
        // Past either end of time the street gives a little and springs back.
        const next = raw < LO ? LO - (LO - raw) * 0.3 : raw > HI ? HI + (raw - HI) * 0.3 : raw
        if (Math.abs(next - at) > 0.02) turn(next - at)
        at = next
        trail.push({ t: ev.timeStamp, at })
        while (trail.length > 2 && ev.timeStamp - trail[0].t > 90) trail.shift()
        wake()
      }
      const up = (ev: PointerEvent) => {
        root.removeEventListener('pointermove', move)
        root.removeEventListener('pointerup', up)
        root.removeEventListener('pointercancel', up)
        if (!dragging) return
        dragging = false
        swallow = true
        root.classList.remove('mu-dragging')
        const a = trail[0]
        const b = trail.at(-1)
        const v = ev.type === 'pointerup' && a && b && b.t > a.t ? (b.at - a.at) / (b.t - a.t) : 0
        at = Math.min(HI, Math.max(LO, at))
        glide(Math.abs(v) > 0.006 && !env.still ? v : 0)
      }
      root.addEventListener('pointermove', move)
      root.addEventListener('pointerup', up)
      root.addEventListener('pointercancel', up)
    })
    // A drag ends with a click on whatever it started on, which must not count as picking it.
    root.addEventListener(
      'click',
      (e) => {
        if (!swallow) return
        swallow = false
        e.stopPropagation()
      },
      true,
    )
    root.addEventListener(
      'wheel',
      (e) => {
        if (travelling || paper) return
        e.preventDefault()
        const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY
        const px = d * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1)
        if (!wheeling) grab()
        const next = Math.min(HI, Math.max(LO, at + px / PITCH / 2))
        turn(next - at)
        at = next
        wake()
        clearTimeout(wheeling)
        wheeling = window.setTimeout(() => {
          wheeling = 0
          settle()
        }, 180)
      },
      { passive: false },
    )
    row.addEventListener('click', (e) => {
      const g = (e.target as Element).closest<SVGGElement>('[data-plot]')
      if (!g || travelling || paper) return
      const p = Number(g.dataset.plot)
      go(nearest(p, p - at))
    })
    dog.root.addEventListener('click', () => {
      bubble(dog.root, T.woof, dogFacing * 9, -17)
      if (!env.still) {
        dog.body.animate([{ transform: 'translateY(0)' }, { transform: 'translateY(-3px)', offset: 0.3 }, { transform: 'translateY(0)' }], { duration: 360 })
      }
    })
    root.addEventListener('keydown', (e) => {
      if (travelling || paper) return
      const base = dest ?? here ?? nearest(at, facing)
      const { y, m, d } = parts(base)
      const month = (n: number) => {
        const index = y * 12 + m - 1 + n
        const yy = Math.floor(index / 12)
        const mm = (index % 12) + 1
        return dayOf(yy, mm, Math.min(d, monthLength(yy, mm)))
      }
      const keys: Record<string, () => Day | undefined> = {
        ArrowLeft: () => (e.shiftKey ? post.before(base) : shift(base, -1)),
        ArrowRight: () => (e.shiftKey ? post.after(base) : shift(base, 1)),
        PageUp: () => month(-1),
        PageDown: () => month(1),
        Home: () => today,
      }
      if (e.key in keys) {
        e.preventDefault()
        const to = keys[e.key]()
        if (to) go(to)
      } else if ((e.key === 'Enter' || e.key === ' ') && e.target === root) {
        e.preventDefault()
        onPostman()
      }
    })
  }

  // ---------- start ----------

  function start(day: Day) {
    build()
    at = origin = plotOf(day)
    dest = day
    manX = manAt(day)
    dogX = manX - HEEL
    facing = dogFacing = 1
    shown = ''
    worldDay = ''
    place()
    pose()
    arrive()
  }

  /** The first moments in the town: the keys work at once, the postman waves, and later someone rides by. */
  function greet() {
    root.focus({ preventScroll: true })
    if (env.still) return
    if (!holding) {
      man.arms[1].animate(
        [0, -150, -118, -150, -118, 0].map((deg) => ({ transform: `rotate(${deg}deg)` })),
        { duration: 1700, delay: 450, easing: 'ease-in-out' },
      )
    }
    later(cyclist, 6000 + Math.random() * 8000)
  }

  start(clamp(env.start ?? today))
  greet()

  const watch = new ResizeObserver(() => {
    if ((env.root.clientWidth >= WIDE ? 640 : 360) === W) return
    const day = dest ?? here ?? nearest(at, facing)
    for (const a of root.getAnimations({ subtree: true })) a.cancel()
    cancelAnimationFrame(raf)
    raf = 0
    motion = null
    travelling = false
    riffle = null
    holding = null
    here = null
    start(day)
  })
  watch.observe(env.root)

  return {
    day: () => dest ?? here ?? nearest(at, facing),
    destroy() {
      // First, because closing the paper sets the postman's next look round.
      paper?.close(true)
      watch.disconnect()
      cancelAnimationFrame(raf)
      clearTimeout(idler)
      clearTimeout(wheeling)
      for (const id of timers) clearTimeout(id)
      timers.clear()
      for (const a of root.getAnimations({ subtree: true })) a.cancel()
      env.root.replaceChildren()
      env.root.style.aspectRatio = ''
    },
  }
}
