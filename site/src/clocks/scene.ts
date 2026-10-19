/**
 * What the clock's scene provides, and small helpers for drawing it and the larger view.
 *
 * A scene draws into an element whose size CSS fixes at 3:2, so it can work in an SVG
 * viewBox of 360 by 240 and never move the page around it.
 */
import type { Lang } from '../site.config'
import type { Snapshot } from './time'

export interface SceneEnv {
  /** The element the scene draws into. */
  root: HTMLElement
  /** The reader prefers reduced motion: no continuous movement and no flourishes. */
  still: boolean
  lang: Lang
}

export interface Scene {
  /**
   * Shows the state at `t`. `jump` is true on the first draw and after a resume or a
   * preview seek, when the state must appear directly with no transition.
   * Returns true while something is moving and another frame is wanted soon; otherwise
   * the next draw comes at the next whole second.
   */
  draw(t: Snapshot, jump: boolean): boolean
  /** What the artwork shows right now, in words, for the accessible label and the readout. */
  reading(t: Snapshot): string
  destroy(): void
}

/** The clock's own words, in each language. */
export interface ClockText {
  name: string
}

const SVG_NS = 'http://www.w3.org/2000/svg'
type Attrs = Record<string, string | number | boolean | null | undefined>
type Child = Node | string | null | undefined | false

function set(el: Element, attrs: Attrs, kids: Child[]) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue
    el.setAttribute(k, v === true ? '' : String(v))
  }
  for (const kid of kids) if (kid) el.append(kid)
  return el
}

/** Creates an SVG element: `svg('rect', { x: 1, y: 2 })`. */
export function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, ...kids: Child[]) {
  return set(document.createElementNS(SVG_NS, tag), attrs, kids) as SVGElementTagNameMap[K]
}

/** Creates an HTML element: `html('p', { class: 'x' }, 'text')`. */
export function html<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...kids: Child[]) {
  return set(document.createElement(tag), attrs, kids) as HTMLElementTagNameMap[K]
}

/**
 * Makes part of a drawing behave as a button: focusable, named for screen readers, and
 * worked by a click, a tap, Enter or Space.
 */
export function asButton(el: Element, label: string, onActivate: () => void) {
  el.setAttribute('role', 'button')
  el.setAttribute('tabindex', '0')
  el.setAttribute('aria-label', label)
  el.addEventListener('click', onActivate)
  el.addEventListener('keydown', (e) => {
    const key = (e as KeyboardEvent).key
    if (key !== 'Enter' && key !== ' ') return
    e.preventDefault()
    onActivate()
  })
}

/** A repeatable sequence of numbers in [0, 1), the same for the same seed. */
export function seeded(seed: string) {
  let h = 2166136261
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 16777619)
  return () => {
    h = (h + 0x6d2b79f5) | 0
    let x = Math.imul(h ^ (h >>> 15), 1 | h)
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296
  }
}

/** Fills the template's {name} slots. */
export const fill = (s: string, vars: Record<string, string | number>) =>
  s.replace(/\{(\w+)\}/g, (all, k: string) => (k in vars ? String(vars[k]) : all))

let ids = 0
/**
 * An id no other element on the page has. The postman clock and his town can be on the
 * page at once, so gradients, filters and clip paths need ids of their own.
 */
export const uid = (prefix: string) => `${prefix}-${++ids}`

/** Like fill, but a slot can take an element, for words that need their own styling. */
export const compose = (template: string, vars: Record<string, string | number | Node>) =>
  template.split(/(\{\w+\})/).map((part) => {
    const k = /^\{(\w+)\}$/.exec(part)?.[1]
    if (k === undefined || !(k in vars)) return part
    const v = vars[k]
    return v instanceof Node ? v : String(v)
  })

const adopted = new Set<string>()
/**
 * Adds a scene's styles to the page the first time the scene is shown. A scene imports its
 * stylesheet with ?inline and passes it here, so its styles load with its code and only on
 * pages where the clock is actually shown.
 */
export function adopt(css: string) {
  if (adopted.has(css)) return
  adopted.add(css)
  document.head.append(html('style', {}, css))
}
