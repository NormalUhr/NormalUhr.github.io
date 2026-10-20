/**
 * The dish of raked sand in the footer keeps the hour the way the karesansui garden does. A stone
 * stands for each hour of the twelve-hour clock, set clockwise from one o'clock, so that midnight
 * and noon have all twelve. A groove raked from the middle grows one turn an hour.
 */
import { everyMinute, hhmm } from './clock'

/** The groove after `minutes` of the hour, one point a minute, in the dish's 40 by 40 drawing. */
function groove(minutes: number) {
  return Array.from({ length: minutes + 1 }, (_, i) => {
    const turn = i / 60
    const r = 1.5 + 8 * turn
    return `${i ? 'L' : 'M'}${(20 + r * Math.sin(2 * Math.PI * turn)).toFixed(2)} ${(20 - r * Math.cos(2 * Math.PI * turn)).toFixed(2)}`
  }).join('')
}

export function keepDish(dish: HTMLElement, label: string) {
  const stones = dish.querySelectorAll('.dish-stone')
  const line = dish.querySelector('.dish-groove')!
  everyMinute((now) => {
    const hours = now.getHours() % 12 || 12
    stones.forEach((stone, i) => stone.classList.toggle('on', i < hours))
    line.setAttribute('d', groove(now.getMinutes()))
    dish.dataset.hours = String(hours)
    dish.setAttribute('aria-label', `${label} ${hhmm(now)}`)
  })
}
