/**
 * The season, the weather and any festival of a day, painted onto the town: classes on the
 * scene for the colours of the trees and for the postman's umbrella or scarf, and the
 * things that fall from or light up the sky.
 */
import { seeded, svg } from '../scene'
import { drawBurst } from './cast'
import { festival, season, weather, type Day } from './days'

export interface Sky {
  root: SVGSVGElement
  /** Rain, snow, leaves, petals and fireflies. */
  fx: SVGGElement
  fireworks: SVGGElement
  width: number
  /** The reader prefers reduced motion: colours only, with nothing falling. */
  still: boolean
}

const KINDS = ['mu-spring', 'mu-summer', 'mu-autumn', 'mu-winter', 'mu-rain', 'mu-snow', 'mu-cold', 'mu-moonday']
const COLOURS = ['#f6c977', '#e2857a', '#9fd3ff', '#c7f0a8']

/** A day not built yet has clear weather and no festival. */
export function paintSky(sky: Sky, day: Day, future: boolean) {
  const { root, fx, fireworks, width: W } = sky
  const s = season(day)
  const w = future ? 'clear' : weather(day)
  const fest = future ? null : festival(day)
  root.classList.remove(...KINDS)
  root.classList.add(`mu-${s}`)
  if (w !== 'clear') root.classList.add(`mu-${w}`)
  if (s === 'winter') root.classList.add('mu-cold')
  if (fest === 'mid-autumn') root.classList.add('mu-moonday')
  for (const a of fx.getAnimations({ subtree: true })) a.cancel()
  for (const a of fireworks.getAnimations({ subtree: true })) a.cancel()
  fx.replaceChildren()
  fireworks.replaceChildren()
  if (sky.still) return

  // The same day always looks the same, because the date is the seed.
  const r = seeded(`fx-${day}`)
  const fall = (el: SVGElement, keys: Keyframe[], opts: KeyframeAnimationOptions & { duration: number }) => {
    fx.append(el)
    el.animate(keys, { iterations: Infinity, delay: -r() * opts.duration, ...opts })
  }
  if (w === 'rain') {
    for (let i = 0; i < W / 9; i++) {
      const x = r() * (W + 40)
      fall(svg('path', { d: `M${x} -12l-2 7`, class: 'mu-drop' }), [{ transform: 'translate(0, 0)' }, { transform: 'translate(-16px, 262px)' }], {
        duration: 620 + r() * 380,
        easing: 'linear',
      })
    }
  } else if (w === 'snow') {
    for (let i = 0; i < W / 12; i++) {
      const x = r() * W
      const sway = (r() - 0.5) * 30
      fall(
        svg('circle', { cx: x, cy: -6, r: 0.7 + r() * 0.9, class: 'mu-flake' }),
        [{ transform: 'translate(0, 0)' }, { transform: `translate(${sway}px, 124px)` }, { transform: `translate(${-sway * 0.4}px, 250px)` }],
        { duration: 7000 + r() * 5000, easing: 'linear' },
      )
    }
  } else if (s === 'autumn' || s === 'spring') {
    for (let i = 0; i < W / 40; i++) {
      const x = r() * W
      const sway = 20 + r() * 40
      fall(
        svg('ellipse', { cx: x, cy: -6, rx: s === 'autumn' ? 2 : 1.5, ry: 1.1, class: s === 'autumn' ? 'mu-fall' : 'mu-petal' }),
        [
          { transform: 'translate(0, 0) rotate(0deg)' },
          { transform: `translate(${sway}px, 80px) rotate(200deg)` },
          { transform: `translate(${-sway * 0.3}px, 170px) rotate(380deg)` },
          { transform: `translate(${sway * 0.6}px, 252px) rotate(560deg)` },
        ],
        { duration: 9000 + r() * 6000, easing: 'ease-in-out' },
      )
    }
  } else if (s === 'summer') {
    for (let i = 0; i < W / 45; i++) {
      const dx = (r() - 0.5) * 30
      const dy = (r() - 0.5) * 20
      fall(
        svg('circle', { cx: r() * W, cy: 110 + r() * 80, r: 0.9, class: 'mu-firefly' }),
        [{ transform: 'translate(0, 0)', opacity: 0.2 }, { transform: `translate(${dx}px, ${dy}px)`, opacity: 1 }],
        { duration: 3000 + r() * 3000, direction: 'alternate', easing: 'ease-in-out' },
      )
    }
  }
  if (fest === 'new-year-eve' || fest === 'spring-eve' || fest === 'spring-day') {
    COLOURS.forEach((colour, i) => {
      const burst = drawBurst(50 + r() * (W - 160), 26 + r() * 50, colour)
      fireworks.append(burst.root)
      burst.sparks.animate(
        [
          { transform: 'scale(.2)', opacity: 0 },
          { transform: 'scale(.3)', opacity: 1, offset: 0.1 },
          { transform: 'scale(1)', opacity: 0.9, offset: 0.5 },
          { transform: 'scale(1.15)', opacity: 0 },
        ],
        { duration: 1900, delay: i * 520, iterations: Infinity },
      )
    })
  }
}
