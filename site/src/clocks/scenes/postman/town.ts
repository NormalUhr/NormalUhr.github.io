/**
 * The postman's town, drawn in flat shapes. A house is drawn from its address alone, so the
 * same address looks the same every time the postman comes back to it.
 */
import { seeded, svg } from '../../scene'
import { pad } from '../../time'

/** Distance between neighbouring plots along a street. */
export const PITCH = 76
/** The line the houses stand on. */
export const GROUND = 180
/** Where the pavement ends and the kerb begins. */
export const KERB = 204
/** Where the postman's feet touch the pavement. */
export const FEET = 191
/** How far left of a mailbox the postman stands to reach it. */
export const REACH = 10.5
/** The lamppost stands in the gap between two plots, where it never hides a house number. */
const LAMP_X = 66

/** How many wall, roof and door colours the stylesheet defines. */
const WALLS = 7
const TILES = 3
const DOORS = 5

/** The first and last house of every street. */
export const FIRST = 0
export const LAST = 59

/** How many plots past either end of its street plot `k` lies. It is 0 for a house of the street. */
export const beyond = (k: number) => (k < FIRST ? FIRST - k : k > LAST ? k - LAST : 0)

type Roof = 'gable' | 'hip' | 'flat'
type Extra = 'pot' | 'box' | 'tree' | 'vine'

interface Pane {
  x: number
  y: number
  w: number
  h: number
  lit: boolean
  /** Which side a curtain is drawn to in a lit window, if any. */
  curtain: -1 | 0 | 1
  /** A tall upper window that opens onto a railing. */
  rail: boolean
}

export interface House {
  width: number
  height: number
  wall: number
  roof: Roof
  tile: number
  rise: number
  chimney: number | null
  door: number
  doorX: number
  arch: boolean
  arched: boolean
  shutters: boolean
  balcony: boolean
  panes: Pane[]
  mailX: number
  extra: Extra
  plant: number
}

/** Ids the drawings share within one scene. */
export interface Paint {
  glow: string
}

const DOOR = { w: 13, h: 26 }
const PANE = { w: 12, h: 17 }
const PLAQUE = { w: 25, h: 15, y: -36 }
/** The letterbox beside each door, measured from the ground the house stands on. */
export const BOX = { w: 10, h: 12, y: -22 }

/** Everything about the house at an address, chosen once from the address. */
export function plan(street: number, number: number): House {
  const r = seeded(`${street}-${number}`)
  const int = (lo: number, hi: number) => lo + Math.floor(r() * (hi - lo + 1))
  const two = r() < 0.55
  const width = int(27, 32) * 2
  const height = two ? int(86, 94) : int(52, 58)
  const wall = int(0, WALLS - 1)
  const roof = (['gable', 'gable', 'hip', 'flat'] as const)[int(0, 3)]
  const tile = int(0, TILES - 1)
  const rise = roof === 'flat' ? 0 : int(15, 24)
  const chimney = roof !== 'flat' && r() < 0.6 ? (r() < 0.5 ? -1 : 1) * Math.round(width / 4) : null
  const door = int(0, DOORS - 1)
  const side = int(-1, 1)
  const doorX = side * (width / 2 - 13)
  const mailX = side < 1 ? doorX + 14 : doorX - 14
  const arch = r() < 0.4
  const arched = r() < 0.35
  const shutters = r() < 0.45
  const balcony = two && r() < 0.35
  const extra = (['pot', 'box', 'tree', 'vine'] as const)[int(0, 3)]
  const plant = int(0, 2)

  const panes: Pane[] = []
  // One window on the ground floor, in the widest stretch of wall the door and mailbox leave.
  const taken = [
    [doorX - DOOR.w / 2 - 2, doorX + DOOR.w / 2 + 2],
    [mailX - BOX.w / 2 - 1, mailX + BOX.w / 2 + 1],
  ].sort((a, b) => a[0] - b[0])
  let best = { from: 0, to: 0 }
  let at = -width / 2 + 4
  for (const [from, to] of [...taken, [width / 2 - 4, width / 2 - 4]]) {
    if (from - at > best.to - best.from) best = { from: at, to: from }
    at = Math.max(at, to)
  }
  const curtain = () => ([0, 0, -1, 1] as const)[int(0, 3)]
  if (best.to - best.from >= PANE.w + 6) {
    panes.push({ x: (best.from + best.to) / 2, y: -42, w: PANE.w, h: PANE.h, lit: r() < 0.5, curtain: curtain(), rail: false })
  }
  if (two) {
    const y = -height + 15
    if (balcony) panes.push({ x: 0, y: y - 4, w: 14, h: PANE.h + 6, lit: r() < 0.6, curtain: curtain(), rail: true })
    else for (const x of [-width / 4 - 1, width / 4 + 1]) panes.push({ x, y, w: PANE.w - 1, h: PANE.h, lit: r() < 0.5, curtain: curtain(), rail: false })
  }
  return { width, height, wall, roof, tile, rise, chimney, door, doorX, arch, arched, shutters, balcony, panes, mailX, extra, plant }
}

/** The height of the roof's top edge above `x`, for a roof seen from the front. */
function roofTop(h: House, x: number) {
  const half = h.width / 2 + 3
  if (h.roof === 'gable') return -h.height - h.rise * (1 - Math.abs(x) / half)
  return -h.height - h.rise * Math.min(1, (half - Math.abs(x)) / 11)
}

function roofShape(h: House) {
  const L = -h.width / 2 - 3
  const R = h.width / 2 + 3
  const H = h.height
  const top = -H - h.rise
  if (h.roof === 'gable') return `M${L} ${-H}L0 ${top}L${R} ${-H}Z`
  return `M${L} ${-H}L${L + 11} ${top}L${R - 11} ${top}L${R} ${-H}Z`
}

/** Courses of tiles, drawn as short lines across the roof that follow its slope. */
function tiles(h: House) {
  const lines: string[] = []
  const half = h.width / 2 + 3
  for (let k = 4.5; k < h.rise - 2; k += 4.5) {
    const y = -h.height - k
    const inset = h.roof === 'gable' ? half * (k / h.rise) : 11 * (k / h.rise)
    lines.push(`M${-half + inset + 1.5} ${y}H${half - inset - 1.5}`)
  }
  return svg('path', { d: lines.join(''), class: 'pm-tile-line' })
}

function roof(h: House) {
  const tone = `var(--pm-roof-${h.tile})`
  const L = -h.width / 2
  if (h.roof === 'flat') {
    // A parapet with a coping of tiles, as on a southern house with a roof terrace.
    return svg(
      'g',
      {},
      svg('rect', { x: L, y: -h.height - 6, width: h.width, height: 6, fill: `var(--pm-wall-${h.wall})` }),
      svg('rect', { x: L - 1.5, y: -h.height - 8, width: h.width + 3, height: 2.6, fill: tone }),
      svg('rect', { x: L, y: -h.height - 1.5, width: h.width, height: 1.5, class: 'pm-shade' }),
    )
  }
  return svg(
    'g',
    {},
    svg('path', { d: roofShape(h), fill: tone }),
    tiles(h),
    svg('rect', { x: L - 3, y: -h.height - 2.2, width: h.width + 6, height: 2.2, class: 'pm-eave' }),
  )
}

function chimney(h: House, x: number) {
  const base = roofTop(h, x)
  return svg(
    'g',
    {},
    svg('rect', { x: x - 3.5, y: base - 11, width: 7, height: 14, fill: `var(--pm-roof-${(h.tile + 1) % TILES})` }),
    svg('rect', { x: x - 4.5, y: base - 13, width: 9, height: 2.4, class: 'pm-cap' }),
  )
}

/** A window with its frame and sill. Its glass glows in the evening when someone is home. */
function pane(p: Pane, h: House, paint: Paint) {
  const g = svg('g', {})
  const { x, y, w, h: ph } = p
  if (p.lit) {
    g.append(svg('ellipse', { cx: x, cy: y + ph / 2, rx: w * 1.3, ry: ph, fill: `url(#${paint.glow})`, class: 'pm-glow' }))
  }
  if (h.shutters && !h.balcony) {
    for (const sx of [x - w / 2 - 5.6, x + w / 2 + 1.1]) {
      g.append(svg('rect', { x: sx, y: y - 0.5, width: 4.5, height: ph + 1, fill: `var(--pm-door-${h.door})` }))
      g.append(svg('path', { d: `M${sx + 0.8} ${y + 4}h2.9M${sx + 0.8} ${y + 8}h2.9M${sx + 0.8} ${y + 12}h2.9`, class: 'pm-slats' }))
    }
  }
  const shape = (inset: number) => {
    const l = x - w / 2 + inset
    const r = x + w / 2 - inset
    const b = y + ph - inset
    const t = y + inset
    if (!h.arched) return `M${l} ${b}V${t}H${r}V${b}Z`
    const rad = (r - l) / 2
    return `M${l} ${b}V${t + rad}A${rad} ${rad} 0 0 1 ${r} ${t + rad}V${b}Z`
  }
  g.append(svg('path', { d: shape(-1.3), class: 'pm-frame' }))
  g.append(svg('path', { d: shape(0), class: p.lit ? 'pm-glass pm-lit' : 'pm-glass' }))
  if (p.lit && p.curtain) {
    const cx = p.curtain < 0 ? x - w / 2 : x + w / 2 - w * 0.34
    g.append(svg('rect', { x: cx, y: y + (h.arched ? w / 2 : 0), width: w * 0.34, height: ph - (h.arched ? w / 2 : 0), class: 'pm-curtain' }))
  }
  g.append(svg('path', { d: `M${x} ${y + (h.arched ? w / 2 : 0)}V${y + ph}M${x - w / 2} ${y + ph * 0.48}H${x + w / 2}`, class: 'pm-mullion' }))
  g.append(svg('rect', { x: x - w / 2 - 2.2, y: y + ph + 0.6, width: w + 4.4, height: 2, class: 'pm-frame' }))
  if (p.rail) {
    const top = y + ph - 8
    const bars = Array.from({ length: 7 }, (_, i) => `M${x - w / 2 - 3 + i * ((w + 6) / 6)} ${top}V${y + ph + 2}`).join('')
    g.append(svg('path', { d: `M${x - w / 2 - 4} ${top}H${x + w / 2 + 4}${bars}`, class: 'pm-rail' }))
  }
  return g
}

function door(h: House) {
  const x = h.doorX
  const { w, h: dh } = DOOR
  const l = x - w / 2
  const outline = (inset: number) => {
    const a = l - inset
    const b = l + w + inset
    if (!h.arch) return `M${a} 0V${-dh - inset}H${b}V0Z`
    const rad = (b - a) / 2
    return `M${a} 0V${-dh + rad - inset}A${rad} ${rad} 0 0 1 ${b} ${-dh + rad - inset}V0Z`
  }
  return svg(
    'g',
    {},
    svg('path', { d: outline(1.6), class: 'pm-frame' }),
    svg('path', { d: outline(0), fill: `var(--pm-door-${h.door})` }),
    svg('path', { d: `M${l + 2.5} -3V${-dh + 9}M${l + w - 2.5} -3V${-dh + 9}M${l + 2.5} ${-dh + 9}H${l + w - 2.5}M${l + 2.5} -3H${l + w - 2.5}`, class: 'pm-panel' }),
    svg('circle', { cx: x + 3.8, cy: -12.5, r: 0.9, class: 'pm-brass' }),
    svg('rect', { x: x - 9, y: -1.6, width: 18, height: 1.6, class: 'pm-step' }),
  )
}

/** The enamel plate with the house number, over the door. */
function plaque(n: number, x: number) {
  const { w, h, y } = PLAQUE
  return svg(
    'g',
    {},
    svg('rect', { x: x - w / 2, y: y - h / 2, width: w, height: h, rx: 2.2, class: 'pm-enamel' }),
    svg('rect', { x: x - w / 2 + 1.4, y: y - h / 2 + 1.4, width: w - 2.8, height: h - 2.8, rx: 1.4, class: 'pm-rim' }),
    svg('text', { x, y: y + 4.7, class: 'pm-num' }, pad(n)),
  )
}

/** A letterbox on the wall. A delivered letter sticks out of its slot. */
function mailbox(x: number) {
  const { w, h, y } = BOX
  return svg(
    'g',
    {},
    svg('rect', { x: x - w / 2, y, width: w, height: h, rx: 1.6, class: 'pm-box' }),
    svg('rect', { x: x - w / 2 + 1.6, y: y + 3, width: w - 3.2, height: 1.8, rx: 0.6, class: 'pm-flap' }),
    svg('rect', { x: x - 3, y: y + 1.2, width: 6, height: 3.4, class: 'pm-peek' }),
  )
}

const leaf = (cx: number, cy: number, rx: number, ry = rx) => svg('ellipse', { cx, cy, rx, ry, class: 'pm-leaf' })

function pot(x: number, plant: number) {
  const g = svg('g', {}, svg('path', { d: `M${x - 3.6} 0L${x - 4.6} -7H${x + 4.6}L${x + 3.6} 0Z`, class: 'pm-pot' }))
  if (plant === 0) g.prepend(leaf(x, -12, 6))
  else if (plant === 1) g.prepend(leaf(x, -16, 3.6, 10))
  else {
    g.prepend(leaf(x, -11.5, 5.4))
    for (const [dx, dy] of [[-2.5, -14], [2.2, -12.5], [0, -9.5]]) g.append(svg('circle', { cx: x + dx, cy: dy, r: 1.2, class: 'pm-flower' }))
  }
  return g
}

function windowBox(p: Pane, plant: number) {
  const y = p.y + p.h + 2.6
  const g = svg('g', {})
  const blooms = plant === 1 ? 'pm-flower-2' : 'pm-flower'
  for (let i = 0; i < 5; i++) {
    const cx = p.x - p.w / 2 + (i + 0.5) * (p.w / 5)
    g.append(leaf(cx, y, 1.9))
    g.append(svg('circle', { cx, cy: y - 1.6, r: 1.1, class: blooms }))
  }
  g.append(svg('rect', { x: p.x - p.w / 2 - 1, y, width: p.w + 2, height: 3.4, class: 'pm-pot' }))
  return g
}

function tree(x: number, plant: number) {
  const g = svg('g', {}, svg('rect', { x: x - 1.2, y: -22, width: 2.4, height: 22, class: 'pm-trunk' }))
  if (plant === 1) g.append(svg('ellipse', { cx: x, cy: -34, rx: 6, ry: 20, class: 'pm-tree' }))
  else g.append(svg('circle', { cx: x, cy: -31, r: 12, class: 'pm-tree' }), svg('circle', { cx: x + 5, cy: -25, r: 7.5, class: 'pm-tree' }))
  return g
}

function vine(x: number, height: number) {
  const g = svg('g', {})
  const r = seeded(`vine-${x}-${height}`)
  for (let y = -3; y > -Math.min(height - 10, 52); y -= 3.2) g.append(leaf(x + 1.5 + r() * 4, y, 2 + r() * 1.4))
  return g
}

/** One house, standing on y = 0 and centred on x = 0. */
export function drawHouse(n: number, h: House, paint: Paint) {
  const L = -h.width / 2
  const g = svg('g', { 'data-house': n })
  if (h.extra === 'tree') g.append(tree(h.width / 2 + 7, h.plant))
  if (h.chimney !== null) g.append(chimney(h, h.chimney))
  g.append(
    roof(h),
    svg('rect', { x: L, y: -h.height, width: h.width, height: h.height, fill: `var(--pm-wall-${h.wall})` }),
    // Light comes from the left, so each wall darkens a little towards its right edge.
    svg('rect', { x: h.width / 2 - 4, y: -h.height, width: 4, height: h.height, class: 'pm-shade' }),
    svg('rect', { x: L, y: -4, width: h.width, height: 4, class: 'pm-shade' }),
  )
  if (h.roof !== 'flat') g.append(svg('rect', { x: L, y: -h.height, width: h.width, height: 3, class: 'pm-shade' }))
  if (h.height > 70) g.append(svg('rect', { x: L, y: -50, width: h.width, height: 1.6, class: 'pm-band' }))
  for (const p of h.panes) g.append(pane(p, h, paint))
  if (h.extra === 'box' && h.panes.length) g.append(windowBox(h.panes[0], h.plant))
  if (h.extra === 'vine') g.append(vine(L, h.height))
  g.append(
    svg('ellipse', { cx: h.doorX, cy: -24, rx: 15, ry: 13, fill: `url(#${paint.glow})`, class: 'pm-porch' }),
    door(h),
    plaque(n, h.doorX),
    mailbox(h.mailX),
  )
  if (h.extra === 'pot') g.append(pot(h.doorX + (h.mailX > h.doorX ? -12 : 12), h.plant))
  return g
}

/**
 * Past either end of a street lie a corner garden, then the cross street, then the first
 * garden of the next block.
 */
export function drawCorner(k: number, street: number) {
  const g = svg('g', {})
  const out = beyond(k)
  const r = seeded(`corner-${street}-${k}`)
  if (out === 2) {
    // At the cross street the road takes the place of the pavement for the width of a street.
    g.append(svg('rect', { x: -26, y: 0, width: 52, height: KERB - GROUND + 3, class: 'pm-road' }))
    g.append(svg('path', { d: `M-26 0V${KERB - GROUND}M26 0V${KERB - GROUND}`, class: 'pm-kerb-edge' }))
    return g
  }
  const toward = k < FIRST ? 1 : -1
  const wall = svg('g', {})
  wall.append(svg('rect', { x: -PITCH / 2 + (toward > 0 ? 8 : 0), y: -11, width: PITCH - 8, height: 11, class: 'pm-stone' }))
  wall.append(svg('rect', { x: -PITCH / 2 + (toward > 0 ? 7 : -1), y: -12.5, width: PITCH - 6, height: 2.2, class: 'pm-cap' }))
  g.append(tree(-8 + r() * 16, out === 1 ? 0 : 1), wall)
  if (out === 1) g.append(leaf(toward * 18, -15, 9, 6), leaf(toward * 26, -12, 6, 5))
  return g
}

/** Slabs of the pavement and the dash down the middle of the road, which move with the street. */
export function drawGround(cross: boolean) {
  const joints = Array.from({ length: 3 }, (_, i) => -PITCH / 2 + (i * PITCH) / 3)
    .filter((x) => !cross || Math.abs(x) > 28)
    .map((x) => `M${x} 1V${KERB - GROUND - 1}`)
    .join('')
  return svg(
    'g',
    {},
    joints && svg('path', { d: joints, class: 'pm-joint' }),
    svg('rect', { x: -11, y: KERB - GROUND + 17, width: 22, height: 1.8, rx: 0.9, class: 'pm-dash' }),
  )
}

/** Hills, cypresses and a bell tower far behind the street, drawn as one repeating tile. */
export const FAR_TILE = 360

export function drawFar() {
  const tile = (x: number) =>
    svg(
      'g',
      { transform: `translate(${x} 0)` },
      svg('path', {
        d: 'M0 150C40 128 70 132 104 140C140 148 166 120 210 122C252 124 282 146 318 140C338 136 350 134 360 150V190H0Z',
        class: 'pm-far-back',
      }),
      svg('path', { d: 'M232 150V112H240V104L244 98L248 104V112H256V150Z', class: 'pm-far-back' }),
      svg('rect', { x: 241.6, y: 116, width: 4.8, height: 7, rx: 2.4, class: 'pm-far-window' }),
      ...[34, 42, 150, 300, 309].map((cx, i) => svg('ellipse', { cx, cy: 150 - (i % 2) * 4, rx: 4, ry: 15 + (i % 2) * 4, class: 'pm-far-tree' })),
      svg('path', {
        d: 'M0 166C30 160 60 158 92 162C130 167 170 156 206 158C240 160 280 168 316 163C334 161 348 160 360 166V190H0Z',
        class: 'pm-far-front',
      }),
    )
  return svg('g', {}, tile(0), tile(FAR_TILE), tile(2 * FAR_TILE))
}

export interface Figure {
  root: SVGGElement
  bob: SVGGElement
  legs: [SVGGElement, SVGGElement]
  arms: [SVGGElement, SVGGElement]
  letter: SVGGElement
}

/** A limb hanging from a joint, so it can swing about that joint. */
function limb(x: number, y: number, cls: string, length: number, end: SVGElement) {
  const swing = svg('g', {}, svg('rect', { x: -1.7, y: 0, width: 3.4, height: length, rx: 1.6, class: cls }), end)
  return { pivot: svg('g', { transform: `translate(${x} ${y})` }, swing), swing }
}

/** The postman, facing the way the house numbers grow, with his feet at y = 0. */
export function drawPostman(scale: number): Figure {
  const shoe = (cls: string) => svg('path', { d: 'M-2 12.4H3.6C5.4 12.4 5.8 14.6 4.4 14.6H-2Z', class: cls })
  const back = limb(-0.6, -14, 'pm-uniform-back', 13, shoe('pm-shoe'))
  const front = limb(0.8, -14, 'pm-uniform', 13, shoe('pm-shoe'))
  const letter = svg(
    'g',
    { transform: 'translate(0.6 11.2) rotate(-8)' },
    svg('rect', { x: -1.2, y: -2.6, width: 8, height: 5.4, rx: 0.6, class: 'pm-envelope' }),
    svg('path', { d: 'M-1.2 -2.6L2.8 0.6L6.8 -2.6', class: 'pm-envelope-fold' }),
    svg('rect', { x: 4.4, y: -2, width: 1.7, height: 2, class: 'pm-stamp' }),
  )
  const armBack = limb(-0.4, -25, 'pm-uniform-back', 11, svg('circle', { cx: 0, cy: 11.4, r: 1.9, class: 'pm-skin-shade' }))
  const armFront = limb(1, -25, 'pm-uniform', 11, svg('g', {}, letter, svg('circle', { cx: 0, cy: 11.4, r: 1.9, class: 'pm-skin' })))
  const bob = svg(
    'g',
    {},
    armBack.pivot,
    back.pivot,
    front.pivot,
    svg('path', { d: 'M-4.6 -25.4C-5.2 -20 -5.6 -15.6 -5.4 -12H5.6C5.6 -16 5.2 -20.6 4.4 -25.4C1.6 -27.2 -1.8 -27.2 -4.6 -25.4Z', class: 'pm-uniform' }),
    svg('rect', { x: -5.5, y: -15.4, width: 11.1, height: 1.6, class: 'pm-belt' }),
    svg('circle', { cx: 3.1, cy: -22.6, r: 0.6, class: 'pm-brass' }),
    svg('circle', { cx: 3.3, cy: -18.6, r: 0.6, class: 'pm-brass' }),
    svg('path', { d: 'M3.8 -25.6L-4.2 -17.6', class: 'pm-strap' }),
    svg('rect', { x: -8.4, y: -19.4, width: 8.6, height: 8, rx: 1.6, class: 'pm-bag' }),
    svg('path', { d: 'M-8.4 -17.8C-8.4 -19 -7.6 -19.4 -6.8 -19.4H-1.4C-0.6 -19.4 0.2 -19 0.2 -17.8V-15.2H-8.4Z', class: 'pm-bag-flap' }),
    svg('rect', { x: -4.9, y: -16, width: 1.6, height: 1.6, rx: 0.3, class: 'pm-brass' }),
    svg('rect', { x: -0.2, y: -28.4, width: 2.8, height: 3, class: 'pm-skin' }),
    svg('path', { d: 'M-2.8 -33.4C-4 -31.4 -3.6 -29.4 -2.2 -28.2L-0.4 -29.6Z', class: 'pm-hair' }),
    svg('circle', { cx: 1.2, cy: -31.2, r: 4.6, class: 'pm-skin' }),
    svg('circle', { cx: 5.6, cy: -30.6, r: 1.1, class: 'pm-skin' }),
    svg('circle', { cx: -0.2, cy: -31, r: 1, class: 'pm-skin-shade' }),
    svg('path', { d: 'M3.6 -28.9C4.4 -29.6 5.4 -29.6 6.2 -28.9', class: 'pm-moustache' }),
    svg('path', { d: 'M-3.9 -33.6L-3.5 -37.4C-1 -39.4 3 -39.4 5.6 -37.4L5.7 -33.6Z', class: 'pm-uniform' }),
    svg('rect', { x: -4, y: -35, width: 9.8, height: 1.4, class: 'pm-band-red' }),
    svg('path', { d: 'M5.4 -34L9.6 -33.2C8.2 -32.4 6.6 -32.4 5.4 -32.6Z', class: 'pm-visor' }),
    svg('circle', { cx: 1.6, cy: -36.4, r: 0.8, class: 'pm-brass' }),
    armFront.pivot,
  )
  const root = svg(
    'g',
    {},
    svg('ellipse', { cx: 0.5, cy: 0.6, rx: 8 * scale, ry: 1.6, class: 'pm-shadow' }),
    svg('g', { transform: `scale(${scale})` }, bob),
  )
  return { root, bob, legs: [back.swing, front.swing], arms: [armBack.swing, armFront.swing], letter }
}

/**
 * The postman's walk: his legs and arms swing and he rises a little with each step, for `steps`
 * steps or until the animations are cancelled. `beat` is the time one step takes.
 */
export function stride(man: Figure, beat: number, steps = Infinity) {
  const swing = (el: Element, a: number) =>
    el.animate(
      [0, a, 0, -a, 0].map((deg) => ({ transform: `rotate(${deg}deg)`, easing: 'ease-in-out' })),
      { duration: 2 * beat, iterations: steps / 2 },
    )
  return [
    swing(man.legs[0], -26),
    swing(man.legs[1], 26),
    swing(man.arms[0], 20),
    swing(man.arms[1], -7),
    man.bob.animate([{ transform: 'translateY(0)' }, { transform: 'translateY(-1.2px)' }, { transform: 'translateY(0)' }], {
      duration: beat,
      iterations: steps,
      easing: 'ease-in-out',
    }),
  ]
}

export interface Lamp {
  root: SVGGElement
  sign: SVGGElement
  label: SVGTextElement
}

/**
 * A lamppost at a corner, with the street's name on an enamel blade. The blade points along
 * the street from the post at `x`: to the right, or to the left when `side` is -1.
 */
export function drawLamp(glow: string, x = LAMP_X, side: 1 | -1 = 1): Lamp {
  const blade = side > 0 ? x + 2 : x - 72
  const label = svg('text', { x: blade + 35, y: 65.4, class: 'pm-sign-text' })
  const sign = svg(
    'g',
    { class: 'pm-sign' },
    svg('path', { d: `M${x} 53h${4 * side}M${x} 68h${4 * side}`, class: 'pm-iron-line' }),
    svg('rect', { x: blade, y: 49, width: 70, height: 23, rx: 2.6, class: 'pm-sign-plate' }),
    svg('rect', { x: blade + 1.6, y: 50.6, width: 66.8, height: 19.8, rx: 1.6, class: 'pm-sign-rim' }),
    label,
  )
  const root = svg(
    'g',
    {},
    svg('circle', { cx: x, cy: 34, r: 30, fill: `url(#${glow})`, class: 'pm-glow' }),
    svg('ellipse', { cx: x + 4 * side, cy: KERB - 4, rx: 46, ry: 9, fill: `url(#${glow})`, class: 'pm-glow' }),
    svg('path', { d: `M${x - 5} ${KERB - 2}L${x - 3.4} ${KERB - 14}H${x + 3.4}L${x + 5} ${KERB - 2}Z`, class: 'pm-iron' }),
    svg('rect', { x: x - 1.5, y: 44, width: 3, height: KERB - 58, class: 'pm-iron' }),
    svg('rect', { x: x - 2.6, y: 128, width: 5.2, height: 2.4, rx: 1, class: 'pm-iron' }),
    svg('rect', { x: x - 2.6, y: 170, width: 5.2, height: 2.4, rx: 1, class: 'pm-iron' }),
    svg('path', { d: `M${x - 7} 44H${x + 7}L${x + 5} 46.5H${x - 5}Z`, class: 'pm-iron' }),
    svg('path', { d: `M${x - 6} 43.6L${x - 7.4} 28H${x + 7.4}L${x + 6} 43.6Z`, class: 'pm-lantern' }),
    svg('path', { d: `M${x} 28V43.6`, class: 'pm-iron-line' }),
    svg('path', { d: `M${x - 9} 28.4L${x} 20.6L${x + 9} 28.4Z`, class: 'pm-iron' }),
    svg('circle', { cx: x, cy: 19.4, r: 1.6, class: 'pm-iron' }),
    sign,
  )
  return { root, sign, label }
}

export interface Calendar {
  /** The hook and the calendar hanging from it. */
  root: SVGGElement
  /** The page that shows the day. */
  page: SVGGElement
  month: SVGTextElement
  day: SVGTextElement
}

/** A small tear-off calendar for the day, to hang under the blade of the lamppost at `x`. */
export function drawCalendar(x = LAMP_X): Calendar {
  const top = 84
  const month = svg('text', { x, y: top + 4.3, class: 'pm-cal-month' })
  const day = svg('text', { x, y: top + 15.6, class: 'pm-cal-day' })
  const page = svg(
    'g',
    { class: 'pm-calendar' },
    svg('rect', { x: x - 7, y: top, width: 14, height: 18, rx: 1.2, class: 'pm-cal-page' }),
    svg('path', { d: `M${x - 7} ${top + 5.8}V${top + 1.2}Q${x - 7} ${top} ${x - 5.8} ${top}H${x + 5.8}Q${x + 7} ${top} ${x + 7} ${top + 1.2}V${top + 5.8}Z`, class: 'pm-cal-band' }),
    month,
    day,
  )
  const root = svg('g', {}, svg('rect', { x: x - 3.4, y: top - 2, width: 6.8, height: 2.6, rx: 0.8, class: 'pm-iron' }), page)
  return { root, page, month, day }
}
