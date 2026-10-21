/**
 * The small clock beside the reading time of an article is the cloister clock tower seen from
 * above. Its hands are the tower's two bridges, the short, wide hour bridge and the long, narrow
 * minute bridge, which turn about the lift in the middle as they do in the tower.
 */
import { everyMinute, hhmm } from './clock'

export function keepBridges(clock: HTMLElement, label: string) {
  const hour = clock.querySelector('.tower-hour')!
  const minute = clock.querySelector('.tower-minute')!
  everyMinute((now) => {
    const minutes = now.getHours() * 60 + now.getMinutes()
    hour.setAttribute('transform', `rotate(${(minutes / 2) % 360} 8 8)`)
    minute.setAttribute('transform', `rotate(${now.getMinutes() * 6} 8 8)`)
    clock.dataset.time = hhmm(now)
    clock.setAttribute('aria-label', `${label} ${hhmm(now)}`)
  })
}
