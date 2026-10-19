/**
 * The townsfolk and the things on the houses, drawn in the same flat shapes as the town.
 * Each figure stands on y = 0 and faces right. What they do is decided by the view.
 */
import { svg } from '../scene'
import { BOX, type Figure, type House } from '../scenes/postman/town'

/** An umbrella for rain and a scarf for the cold, hidden until the weather calls for them. */
export function dress(man: Figure) {
  man.bob.prepend(
    svg(
      'g',
      { class: 'mu-umbrella' },
      svg('path', { d: 'M-1 -16V-47', class: 'mu-handle' }),
      svg('path', { d: 'M-14 -45Q-1 -61 12 -45Q8.8 -47.6 5.6 -45Q2.4 -47.6 -0.8 -45Q-4 -47.6 -7.4 -45Q-10.6 -47.6 -14 -45Z', class: 'mu-canopy' }),
    ),
  )
  const scarf = svg(
    'g',
    { class: 'mu-scarf' },
    svg('rect', { x: -2.8, y: -28.8, width: 6.8, height: 2.7, rx: 1.2, class: 'mu-wool' }),
    svg('path', { d: 'M-2 -27.2L-3.8 -21.4L-1.6 -21.2L-0.2 -26.6Z', class: 'mu-wool' }),
  )
  // Under the front arm, which is drawn over the body.
  man.arms[1].parentElement!.before(scarf)
}

/** A small parcel, held in the postman's hand next to where he holds a letter. */
export function heldParcel() {
  return svg(
    'g',
    { transform: 'translate(0.4 11.4) rotate(-6)', class: 'mu-held' },
    svg('rect', { x: -1.6, y: -3.4, width: 8.4, height: 6.4, rx: 0.7, class: 'mu-parcel' }),
    svg('path', { d: 'M2.6 -3.4V3M-1.6 -0.2H6.8', class: 'mu-twine' }),
  )
}

export interface Dog {
  root: SVGGElement
  /** Turns round with the postman. */
  face: SVGGElement
  body: SVGGElement
  legs: SVGElement[]
  tail: SVGElement
}

export function drawDog(): Dog {
  const leg = (x: number) => svg('rect', { x: x - 0.75, y: -5.8, width: 1.5, height: 5.8, rx: 0.7, class: 'mu-dog-leg' })
  const legs = [leg(-4.4), leg(-2.6), leg(3.2), leg(5)]
  const tail = svg('path', { d: 'M-6.2 -9C-8.4 -10.4 -9 -12.4 -8.6 -14', class: 'mu-dog-tail' })
  const body = svg(
    'g',
    {},
    ...legs,
    tail,
    svg('rect', { x: -6.8, y: -11, width: 13.4, height: 6.4, rx: 3.2, class: 'mu-dog-fur' }),
    svg('rect', { x: 4.4, y: -12, width: 1.4, height: 4.6, rx: 0.5, class: 'mu-collar' }),
    svg('circle', { cx: 7.6, cy: -12.2, r: 3.3, class: 'mu-dog-fur' }),
    svg('ellipse', { cx: 10.6, cy: -11.4, rx: 2.1, ry: 1.5, class: 'mu-dog-fur' }),
    svg('circle', { cx: 12.5, cy: -11.8, r: 0.65, class: 'mu-dog-dark' }),
    svg('circle', { cx: 8.7, cy: -13.1, r: 0.5, class: 'mu-dog-dark' }),
    svg('path', { d: 'M5.8 -14.6C4.6 -14 4.4 -11.4 5.4 -10C6.4 -10.8 6.8 -13 6.6 -14.4Z', class: 'mu-dog-dark' }),
  )
  const face = svg('g', {}, body)
  const root = svg(
    'g',
    { class: 'mu-dog' },
    svg('ellipse', { cx: 0, cy: 0.4, rx: 7.5, ry: 1.3, class: 'pm-shadow' }),
    svg('circle', { cx: 2, cy: -8, r: 11, class: 'mu-hit' }),
    face,
  )
  return { root, face, body, legs, tail }
}

export interface Bird {
  g: SVGGElement
  fly: SVGGElement
  wing: SVGElement
  head: SVGGElement
}

/** Pigeons along the top of the street sign. */
export function drawBirds(xs: number[], y: number) {
  const list: Bird[] = xs.map((x, i) => {
    const wing = svg('path', { d: 'M-1.6 -3.6Q1 -5.8 3.2 -3.4Q1 -2.4 -1.6 -3.6Z', class: 'mu-wing' })
    const head = svg(
      'g',
      {},
      svg('circle', { cx: -2.6, cy: -5.4, r: 1.35, class: 'mu-bird' }),
      svg('path', { d: 'M-3.9 -5.6L-5 -5.2L-3.9 -4.9Z', class: 'mu-beak' }),
    )
    const fly = svg(
      'g',
      {},
      svg('path', { d: 'M-0.6 -1.3V0M0.6 -1.3V0', class: 'mu-bird-leg' }),
      svg('ellipse', { cx: 0, cy: -3.2, rx: 3, ry: 2, class: 'mu-bird' }),
      svg('path', { d: 'M2.4 -3.6L4.8 -2.4L2.6 -2.2Z', class: 'mu-bird' }),
      wing,
      head,
    )
    const g = svg('g', { transform: `translate(${x} ${y})` }, svg('g', { transform: i % 2 ? 'scale(-1 1)' : null }, fly))
    return { g, fly, wing, head }
  })
  return { root: svg('g', { class: 'mu-birds' }, svg('rect', { x: xs[0] - 8, y: y - 12, width: xs.at(-1)! - xs[0] + 16, height: 14, class: 'mu-hit' }), ...list.map((b) => b.g)), list }
}

export interface Cat {
  root: SVGGElement
  head: SVGGElement
  tail: SVGElement
  z: SVGElement
}

/** A cat asleep on a roof. */
export function drawCat(): Cat {
  const head = svg(
    'g',
    {},
    svg('path', { d: 'M-7.3 -6.4L-7.1 -10L-5.2 -8Z M-3.6 -8.2L-1.9 -10.2L-1.7 -6.6Z', class: 'mu-cat-fur' }),
    svg('circle', { cx: -4.7, cy: -5.8, r: 2.9, class: 'mu-cat-fur' }),
    svg('path', { d: 'M-6.4 -5.2q.7 .5 1.4 0M-4.4 -5.2q.7 .5 1.4 0', class: 'mu-cat-lid' }),
    svg('g', { class: 'mu-cat-eyes' }, svg('circle', { cx: -5.7, cy: -5.4, r: 0.55 }), svg('circle', { cx: -3.7, cy: -5.4, r: 0.55 })),
  )
  const tail = svg('path', { d: 'M4.6 -1.8C7.6 -2 8.4 1.2 4.8 0.9H-0.6', class: 'mu-cat-tail' })
  const z = svg('text', { x: -2, y: -11, class: 'mu-z' }, 'z')
  const root = svg(
    'g',
    { class: 'mu-cat' },
    svg('circle', { cx: -1, cy: -4.4, r: 12, class: 'mu-hit' }),
    svg('ellipse', { cx: 0.6, cy: -3.2, rx: 5.6, ry: 3.4, class: 'mu-cat-fur' }),
    tail,
    head,
    z,
  )
  return { root, head, tail, z }
}

/** Someone on a bench, reading the day's paper. */
export function drawReader() {
  const paper = svg(
    'g',
    {},
    svg('rect', { x: -6.6, y: -14, width: 13.2, height: 10, rx: 0.4, class: 'mu-paper' }),
    svg('path', { d: 'M0 -14V-4', class: 'mu-paper-fold' }),
    svg('path', { d: 'M-5.6 -12.6H5.6', class: 'mu-headline' }),
    svg('path', { d: 'M-5.6 -10.6h4.4M1 -10.6h4.6M-5.6 -8.8h4.4M1 -8.8h3.8M-5.6 -7h3.8M1 -7h4.6M-5.6 -5.2h4.4', class: 'mu-paper-ink' }),
  )
  const root = svg(
    'g',
    { class: 'mu-reader' },
    svg('circle', { cx: 0, cy: -2, r: 15, class: 'mu-hit' }),
    svg('rect', { x: -11, y: -5, width: 22, height: 1.5, rx: 0.6, class: 'mu-bench' }),
    svg('path', { d: 'M-9.4 4V10M9.4 4V10', class: 'mu-bench-leg' }),
    svg('rect', { x: -3.8, y: -8.4, width: 7.6, height: 11, rx: 2.5, class: 'mu-coat' }),
    svg('circle', { cx: 0, cy: -11.6, r: 3, class: 'pm-skin' }),
    svg('path', { d: 'M-2.4 3V9.6M2 3V9.6', class: 'mu-trousers' }),
    svg('rect', { x: -11, y: 2.4, width: 22, height: 2, rx: 0.6, class: 'mu-bench' }),
    paper,
  )
  return { root, paper }
}

/** Someone riding a bicycle along the road. */
export function drawCyclist() {
  const wheel = (cx: number) =>
    svg(
      'g',
      { transform: `translate(${cx} -5.2)` },
      svg('g', { class: 'mu-spin' }, svg('circle', { r: 5, class: 'mu-tyre' }), svg('path', { d: 'M-4.6 0H4.6M0 -4.6V4.6', class: 'mu-spoke' })),
    )
  const wheels = [wheel(-9), wheel(9)]
  const leg = (cls: string) => svg('g', { transform: 'translate(-3.4 -13)' }, svg('rect', { x: -1.2, y: 0, width: 2.4, height: 9.4, rx: 1.1, class: cls }))
  const legs = [leg('mu-rider-back'), leg('mu-rider')]
  const root = svg(
    'g',
    { class: 'mu-cyclist' },
    svg('ellipse', { cx: 0, cy: 0.4, rx: 15, ry: 1.4, class: 'pm-shadow' }),
    ...wheels,
    legs[0],
    svg('path', { d: 'M-9 -5.2L-2.4 -5.2L5 -11.6L9 -5.2M-2.4 -5.2L-4.6 -12.6M5 -11.6L6 -15.4', class: 'mu-frame' }),
    svg('path', { d: 'M-6.6 -13H-2.4M4.6 -15.6H8', class: 'mu-frame-line' }),
    svg('path', { d: 'M-4.4 -13.4C-3.6 -17 -1.4 -21 0.8 -22.6L3.6 -21C2.6 -18.6 1.4 -15.6 -1 -13Z', class: 'mu-rider' }),
    svg('path', { d: 'M1.6 -21L6.4 -15.6', class: 'mu-arm' }),
    svg('circle', { cx: 2.8, cy: -25.2, r: 2.6, class: 'pm-skin' }),
    svg('path', { d: 'M0.2 -26.2C0.4 -28.4 4.6 -28.8 5.6 -26.4Z', class: 'mu-cap' }),
    legs[1],
  )
  return { root, spins: wheels.map((w) => w.firstElementChild as SVGGElement), legs: legs.map((l) => l.firstElementChild as SVGElement) }
}

/** A word in a bubble, as from one of the townsfolk. */
export function drawBubble(text: string, x: number, y: number) {
  const w = [...text].reduce((n, ch) => n + (/[⺀-鿿]/.test(ch) ? 6.4 : 3.4), 7)
  return svg(
    'g',
    { class: 'mu-bubble', transform: `translate(${x} ${y})` },
    svg('rect', { x: -w / 2, y: -9, width: w, height: 9.5, rx: 3 }),
    svg('text', { x: 0, y: -2.6 }, text),
  )
}

/** The red flag on a mailbox that has mail to collect. */
export function drawFlag(h: House) {
  const x = h.mailX + BOX.w / 2 + 0.4
  return svg(
    'g',
    { class: 'mu-flag' },
    svg('rect', { x, y: BOX.y - 11, width: 1, height: 12, class: 'mu-pole' }),
    svg('path', { d: `M${x + 1} ${BOX.y - 11}h6.6v4.4h-6.6Z`, class: 'mu-flag-cloth' }),
  )
}

/** A parcel on the step, on a day an article came out. */
export function drawDoorParcel(h: House) {
  const x = h.doorX
  return svg(
    'g',
    { class: 'mu-step-parcel' },
    svg('rect', { x: x - 4, y: -6.8, width: 8, height: 5.2, rx: 0.6, class: 'mu-parcel' }),
    svg('path', { d: `M${x} -6.8V-1.6M${x - 4} -4.2H${x + 4}`, class: 'mu-twine' }),
  )
}

export function drawLantern(x: number, y: number, size = 1) {
  const swing = svg(
    'g',
    { class: 'mu-swing' },
    svg('path', { d: 'M0 0V3', class: 'mu-string' }),
    svg('rect', { x: -1.6, y: 2.6, width: 3.2, height: 1, class: 'mu-gold' }),
    svg('ellipse', { cx: 0, cy: 6.4, rx: 3, ry: 3.4, class: 'mu-red' }),
    svg('path', { d: 'M0 3.1V9.7M-1.6 3.6Q-2.6 6.4 -1.6 9.2M1.6 3.6Q2.6 6.4 1.6 9.2', class: 'mu-rib' }),
    svg('rect', { x: -1.6, y: 9.4, width: 3.2, height: 1, class: 'mu-gold' }),
    svg('path', { d: 'M0 10.4V13.6', class: 'mu-tassel' }),
  )
  return { root: svg('g', { transform: `translate(${x} ${y}) scale(${size})` }, swing), swing }
}

export const drawPumpkin = (x: number) =>
  svg(
    'g',
    {},
    svg('ellipse', { cx: x, cy: -3, rx: 4, ry: 3, class: 'mu-pumpkin' }),
    svg('rect', { x: x - 0.5, y: -7.2, width: 1, height: 1.8, class: 'mu-stem' }),
    svg('path', { d: `M${x - 2} -3.6l.7-.9.7.9ZM${x + 0.6} -3.6l.7-.9.7.9ZM${x - 1.8} -1.8h3.6l-.9.7h-1.8Z`, class: 'mu-carve' }),
  )

export const drawWreath = (x: number) =>
  svg('g', {}, svg('circle', { cx: x, cy: -17.5, r: 3.1, class: 'mu-wreath' }), svg('circle', { cx: x, cy: -14.6, r: 0.9, class: 'mu-bow' }))

/** A burst of fireworks, drawn at full size around its centre. */
export function drawBurst(cx: number, cy: number, colour: string) {
  const sparks = svg(
    'g',
    {},
    ...Array.from({ length: 12 }, (_, k) => {
      const t = (k / 12) * Math.PI * 2
      return svg('path', { d: `M${Math.cos(t) * 4} ${Math.sin(t) * 4}L${Math.cos(t) * 13} ${Math.sin(t) * 13}`, class: 'mu-spark', stroke: colour })
    }),
  )
  return { root: svg('g', { transform: `translate(${cx} ${cy})` }, sparks), sparks }
}
