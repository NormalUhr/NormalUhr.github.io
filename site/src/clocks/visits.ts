/**
 * The postman's visits to the rest of the site: he walks along the bottom of the window, or a
 * letter is slipped in at its edge. Either one, pressed, opens his town.
 */
import type { Lang } from '../site.config'
import { openTown } from './host'
import copy from './museum/copy'
import type { Day } from './museum/days'
import { adopt, asButton, html, svg } from './scene'
import { drawPostman, stride } from './scenes/postman/town'
import townCss from './scenes/postman/style.css?inline'
import css from './visits.css?inline'

const motion = matchMedia('(prefers-reduced-motion: reduce)')
/** His pace in pixels a second, and the time one step takes at that pace. */
const PACE = 60
const BEAT = 260
/** A visitor who prefers stillness sees him stand at the corner for this long instead. */
const STAY = 20_000

let walking = false
let posted = false

/** The postman walks in at the left along the bottom of the window and out at the right. */
export function walkPast(lang: Lang) {
  if (walking) return
  walking = true
  adopt(townCss)
  adopt(css)
  const man = drawPostman(1)
  const figure = svg('svg', { viewBox: '-15 -43 30 47', class: 'pm egg-walker' }, man.root)
  const street = html('div', { class: 'egg-street' }, figure)
  document.body.append(street)

  const gone = () => {
    if (!street.isConnected) return
    for (const a of street.getAnimations({ subtree: true })) a.cancel()
    street.remove()
    walking = false
  }
  asButton(figure, copy[lang].postman, () => {
    gone()
    openTown(lang)
  })
  if (motion.matches) {
    figure.classList.add('egg-still')
    setTimeout(gone, STAY)
    return
  }
  const width = street.clientWidth
  const duration = ((width + figure.clientWidth) / PACE) * 1000
  stride(man, BEAT, Math.ceil(duration / BEAT))
  figure
    .animate([{ transform: 'translateX(-100%)' }, { transform: `translateX(${width}px)` }], { duration, easing: 'linear' })
    .finished.then(gone, () => {})
}

/** A letter from `day` is slipped in at the right edge of the window, once a page. */
export function slipLetter(lang: Lang, day: Day) {
  if (posted) return
  posted = true
  adopt(townCss)
  adopt(css)
  const letter = svg(
    'svg',
    { viewBox: '0 0 64 44', class: 'pm egg-letter' },
    svg('rect', { x: 1, y: 1, width: 62, height: 42, rx: 2.4, class: 'egg-paper' }),
    svg('path', { d: 'M2 2.6L32 24L62 2.6', class: 'egg-fold' }),
    svg('rect', { x: 49, y: 6, width: 9, height: 11, rx: 0.6, class: 'egg-stamp' }),
    // The address, written by hand.
    svg('path', { d: 'M13 30.4Q20 28.8 27 30.2T41 29.6M13 35.6Q18 34.4 24 35.4T33 35', class: 'egg-ink' }),
  )
  const edge = html('div', { class: 'egg-edge' }, letter)
  document.body.append(edge)
  asButton(letter, copy[lang].letter, () => {
    edge.remove()
    openTown(lang, day)
  })
  if (!motion.matches) letter.animate([{ transform: 'translateX(120%) rotate(-20deg)' }], { duration: 900, easing: 'cubic-bezier(.2, .7, .3, 1)' })
}
