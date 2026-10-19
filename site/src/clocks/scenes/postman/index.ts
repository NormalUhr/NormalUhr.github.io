/**
 * A postman walks a small town whose streets are the hours and whose house numbers are
 * the minutes, so the house he is delivering to tells the time. Only the few houses
 * around him are drawn. The rest of the town exists only as addresses.
 */
import { postman } from '../../encode'
import { adopt, compose, fill, seeded, svg, uid, type Scene, type SceneEnv } from '../../scene'
import { pad, type Snapshot } from '../../time'
import phrases from './phrases'
import {
  FAR_TILE,
  FEET,
  FIRST,
  GROUND,
  KERB,
  LAST,
  PITCH,
  REACH,
  beyond,
  drawCalendar,
  drawCorner,
  drawFar,
  drawGround,
  drawHouse,
  drawLamp,
  drawPostman,
  plan,
  stride,
} from './town'
import css from './style.css?inline'

/** Where the destination's plot comes to rest on screen. */
const CENTRE = 180
/** Plots drawn on each side of the destination. */
const NEAR = 3
const STRIDE = 400
const WALK = 3 * STRIDE
const DELIVER = 820
const TURN = 1500
const EASE = 'cubic-bezier(.45, .05, .35, 1)'
/** The hills behind the town move this much slower than the street, so they read as far away. */
const DEPTH = 0.28
const SCALE = 0.92

interface Plot {
  plot: SVGGElement
  home: SVGGElement | null
}

/** One street's row of plots, moved as a whole when the postman walks. */
interface Row {
  street: number
  g: SVGGElement
  plots: Map<number, Plot>
}

const shift = (n: number) => CENTRE - n * PITCH
const along = (x: number) => `translateX(${x}px)`
const standAt = (street: number, n: number) => CENTRE + plan(street, n).mailX - REACH

export default function create(env: SceneEnv): Scene {
  adopt(css)
  const text = phrases[env.lang]
  const ids = { sky: uid('pm-sky'), glow: uid('pm-glow'), moon: uid('pm-moon') }
  const paint = { glow: ids.glow }

  const stars = seeded('pm-stars')
  const far = drawFar()
  const rows = svg('g', {})
  const man = drawPostman(SCALE)
  const lamp = drawLamp(ids.glow)
  const calendar = drawCalendar()
  lamp.root.append(calendar.root)

  const art = svg(
    'g',
    { 'aria-hidden': 'true' },
    svg('rect', { width: 360, height: 240, fill: `url(#${ids.sky})` }),
    svg(
      'g',
      { class: 'pm-night' },
      ...Array.from({ length: 18 }, () =>
        svg('circle', { cx: (stars() * 360).toFixed(1), cy: (4 + stars() * 96).toFixed(1), r: (0.45 + stars() * 0.6).toFixed(2), class: 'pm-star' }),
      ),
      svg('circle', { cx: 300, cy: 34, r: 26, fill: `url(#${ids.moon})` }),
      svg('path', { d: 'M300 22A12 12 0 0 1 300 46A7 12 0 0 0 300 22Z', class: 'pm-moon' }),
    ),
    svg(
      'g',
      { class: 'pm-day' },
      svg('circle', { cx: 300, cy: 36, r: 19, class: 'pm-sun-halo' }),
      svg('circle', { cx: 300, cy: 36, r: 12, class: 'pm-sun' }),
      svg('path', { d: 'M136 34H186C186 28 180 25 174 27C171 20 160 19 156 26C149 23 141 27 142 34Z', class: 'pm-cloud' }),
      svg('path', { d: 'M212 66H248C248 61 243 59 239 61C236 56 228 56 226 61C220 59 214 61 212 66Z', class: 'pm-cloud' }),
    ),
    far,
    svg('rect', { x: 0, y: GROUND, width: 360, height: KERB - GROUND, class: 'pm-pave' }),
    svg('rect', { x: 0, y: KERB, width: 360, height: 3, class: 'pm-kerb' }),
    svg('rect', { x: 0, y: KERB + 3, width: 360, height: 240 - KERB - 3, class: 'pm-road' }),
    svg('rect', { x: 0, y: KERB + 3, width: 360, height: 1.6, class: 'pm-gutter' }),
    rows,
    man.root,
    lamp.root,
  )

  const root = svg(
    'svg',
    { viewBox: '0 0 360 240', class: 'pm', 'aria-hidden': 'true' },
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
    ),
    art,
  )
  env.root.append(root)

  /** The plots of one street from n - NEAR to n + NEAR, in street order. */
  function row(street: number, n: number): Row {
    const r: Row = { street, g: svg('g', {}), plots: new Map() }
    rows.append(r.g)
    extend(r, n)
    return r
  }

  function extend(r: Row, n: number) {
    for (let k = n - NEAR; k <= n + NEAR; k++) {
      if (r.plots.has(k)) continue
      const plot = svg('g', { transform: `translate(${k * PITCH} ${GROUND})` })
      const out = beyond(k)
      plot.append(drawGround(out === 2))
      const home = out ? null : drawHouse(k, plan(r.street, k), paint)
      if (home) plot.append(home)
      else plot.append(drawCorner(k, r.street))
      // A later plot is drawn over an earlier one, so a tree in a gap stays behind the next house.
      const after = [...r.plots.entries()].filter(([j]) => j > k).sort(([a], [b]) => a - b)[0]
      if (after) after[1].plot.before(plot)
      else r.g.append(plot)
      r.plots.set(k, { plot, home })
    }
  }

  function prune(r: Row, n: number) {
    for (const [k, p] of r.plots) {
      if (k >= n - NEAR && k <= n + NEAR) continue
      p.plot.remove()
      r.plots.delete(k)
    }
  }

  /** Marks the destination with data-dest. Houses visited earlier this hour keep their letter. */
  function mark(r: Row, n: number) {
    for (const [k, { home }] of r.plots) {
      if (!home) continue
      home.toggleAttribute('data-dest', k === n)
      home.classList.toggle('pm-done', k < n)
    }
  }

  let current: Row | null = null
  let shown: { street: number; number: number; date: string } | null = null
  /** How far the far hills have moved, within one tile. */
  let farAt = 0

  const placeFar = (at: number) => (far.style.transform = along(-FAR_TILE - at))

  function moveFar(by: number, duration: number) {
    const from = farAt
    farAt = (farAt + by) % FAR_TILE
    placeFar(farAt)
    far.animate([{ transform: along(-FAR_TILE - from) }, { transform: along(-FAR_TILE - from - by) }], { duration, easing: EASE })
  }


  function setSign(street: number) {
    lamp.sign.dataset.street = String(street)
    lamp.label.replaceChildren(...compose(text.sign, { n: svg('tspan', { class: 'pm-sign-num' }, pad(street)) }))
  }

  function setDay(date: string) {
    const [, m, d] = date.split('-').map(Number)
    calendar.month.textContent = text.months.split(' ')[m - 1]
    calendar.day.textContent = String(d)
  }

  /** Yesterday's page comes off the calendar and falls away, showing the new day under it. */
  function tear(date: string) {
    const page = calendar.page.cloneNode(true) as SVGGElement
    calendar.page.after(page)
    setDay(date)
    page
      .animate(
        [
          { transform: 'translate(0, 0) rotate(0deg)', opacity: 1 },
          { transform: 'translate(1px, 2px) rotate(-12deg)', opacity: 1, offset: 0.3 },
          { transform: 'translate(6px, 26px) rotate(-40deg)', opacity: 0 },
        ],
        { duration: 1100, delay: 300, easing: 'ease-in', fill: 'both' },
      )
      .finished.then(
        () => page.remove(),
        () => page.remove(),
      )
  }

  /** Turns a plate over. A copy with the old words folds away, then the plate unfolds with the new words. */
  function flip(el: SVGGElement, change: () => void) {
    const old = el.cloneNode(true) as SVGGElement
    old.removeAttribute('data-street')
    el.before(old)
    change()
    const fold = [{ transform: 'scaleY(1)' }, { transform: 'scaleY(0)' }]
    old.animate(fold, { duration: 260, easing: 'ease-in', fill: 'forwards' }).finished.then(
      () => old.remove(),
      () => old.remove(),
    )
    el.animate([...fold].reverse(), { duration: 300, delay: 260, easing: 'ease-out', fill: 'backwards' })
  }

  function placeMan(street: number, n: number) {
    man.root.style.transform = `translate(${standAt(street, n)}px, ${FEET}px)`
  }

  /** The postman reaches up, posts the letter and takes the next one from his bag. */
  function deliver(home: SVGGElement, delay: number) {
    const timing = { duration: DELIVER, delay }
    man.arms[1].animate(
      [
        { transform: 'rotate(0deg)' },
        { transform: 'rotate(-112deg)', offset: 0.34 },
        { transform: 'rotate(-112deg)', offset: 0.6 },
        { transform: 'rotate(0deg)', offset: 0.86 },
        { transform: 'rotate(0deg)' },
      ],
      { ...timing, easing: 'ease-in-out' },
    )
    man.letter.animate(
      [{ opacity: 1 }, { opacity: 1, offset: 0.4 }, { opacity: 0, offset: 0.5 }, { opacity: 0, offset: 0.86 }, { opacity: 1 }],
      timing,
    )
    home.querySelector('.pm-peek')?.animate([{ opacity: 0 }, { opacity: 0, offset: 0.46 }, { opacity: 1, offset: 0.54 }, { opacity: 1 }], {
      ...timing,
      fill: 'backwards',
    })
    home.querySelector('.pm-flap')?.animate([{ opacity: 1 }, { opacity: 0.2, offset: 0.46 }, { opacity: 1, offset: 0.62 }, { opacity: 1 }], timing)
  }

  /** Draws the present state at once, with nothing in motion. */
  function settle(t: Snapshot) {
    for (const a of root.getAnimations({ subtree: true })) a.cancel()
    const { street, number } = postman(t)
    rows.replaceChildren()
    current = row(street, number)
    current.g.style.transform = along(shift(number))
    mark(current, number)
    placeMan(street, number)
    farAt = ((t.minuteIndex * PITCH * DEPTH) % FAR_TILE + FAR_TILE) % FAR_TILE
    placeFar(farAt)
    setSign(street)
    setDay(t.date)
  }

  /** On to the next house of the same street. */
  function walk(t: Snapshot) {
    if (!current || !shown || env.still) return settle(t)
    const { street, number } = postman(t)
    const r = current
    extend(r, number)
    mark(r, number)
    r.g.style.transform = along(shift(number))
    r.g.animate([{ transform: along(shift(shown.number)) }, { transform: along(shift(number)) }], { duration: WALK, easing: EASE }).finished.then(
      () => prune(r, number),
      () => {},
    )
    const from = standAt(street, shown.number)
    placeMan(street, number)
    man.root.animate([{ transform: `translate(${from}px, ${FEET}px)` }, { transform: `translate(${standAt(street, number)}px, ${FEET}px)` }], {
      duration: WALK,
      easing: EASE,
    })
    stride(man, STRIDE, 3)
    moveFar(PITCH * DEPTH, WALK)
    const home = r.plots.get(number)?.home
    if (home) deliver(home, WALK)
  }

  /** Round the corner from the last house of one street to the first house of the next. */
  function turn(t: Snapshot) {
    if (!current || !shown || env.still) return settle(t)
    const { street, number } = postman(t)
    const old = current
    const away = 2.5 * PITCH
    for (const el of old.g.querySelectorAll('[data-house]')) {
      el.removeAttribute('data-house')
      el.removeAttribute('data-dest')
    }
    old.g.animate(
      [
        { transform: along(shift(shown.number)), opacity: 1 },
        { opacity: 1, offset: 0.35 },
        { transform: along(shift(shown.number) - away), opacity: 0 },
      ],
      { duration: TURN * 0.6, easing: 'ease-in', fill: 'forwards' },
    ).finished.then(
      () => old.g.remove(),
      () => old.g.remove(),
    )
    current = row(street, number)
    current.g.style.transform = along(shift(number))
    mark(current, number)
    current.g.animate(
      [
        { transform: along(shift(number) + away), opacity: 0 },
        { transform: along(shift(number)), opacity: 1 },
      ],
      { duration: TURN * 0.62, delay: TURN * 0.38, easing: 'ease-out', fill: 'backwards' },
    )
    const from = standAt(old.street, shown.number)
    placeMan(street, number)
    man.root.animate([{ transform: `translate(${from}px, ${FEET}px)` }, { transform: `translate(${standAt(street, number)}px, ${FEET}px)` }], {
      duration: TURN,
      easing: EASE,
    })
    stride(man, STRIDE, 5)
    moveFar(PITCH * DEPTH, TURN)
    flip(lamp.sign, () => setSign(street))
    if (t.date !== shown.date) tear(t.date)
    const home = current.plots.get(number)?.home
    if (home) deliver(home, TURN)
  }

  return {
    draw(t, jump) {
      const { street, number } = postman(t)
      if (jump || !shown) settle(t)
      else if (street === shown.street && number === shown.number + 1 && t.date === shown.date) walk(t)
      else if (number === FIRST && shown.number === LAST && street === (shown.street + 1) % 24) turn(t)
      else if (street !== shown.street || number !== shown.number || t.date !== shown.date) settle(t)
      else return false
      shown = { street, number, date: t.date }
      return false
    },
    reading(t) {
      const { street, number } = postman(t)
      return fill(text.reading, { number: pad(number), street: pad(street) })
    },
    destroy() {
      for (const a of root.getAnimations({ subtree: true })) a.cancel()
    },
  }
}
