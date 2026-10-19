/**
 * The moments that set off an Easter egg. Each one only watches the page until its moment
 * comes, so every page carries the eggs at almost no cost.
 */
import type { Lang } from '../site.config'

export const lang: Lang = document.documentElement.lang.startsWith('zh') ? 'zh' : 'en'

/** The visitor is not looking at the page, or is already inside something that opened over it. */
const away = () => document.hidden || document.querySelector('dialog[open]') !== null

/** Runs `fire` when the visitor's own clock passes each hour, if they are looking at the page. */
export function onHourTurn(fire: () => void) {
  const now = new Date()
  const turn = new Date(now.getFullYear(), now.getMonth(), now.getDate(), now.getHours() + 1).getTime()
  setTimeout(() => {
    // A background tab runs its timers late. An hour that turned minutes ago is no longer the moment.
    if (Date.now() - turn < 5000 && !away()) fire()
    onHourTurn(fire)
  }, turn - now.getTime())
}

/**
 * On a share `chance` of page views, runs `fire` once, at a random moment between `from` and
 * `to` seconds after the page opens. If the visitor has left the tab, it waits for them.
 */
export function byChance(chance: number, [from, to]: [number, number], fire: () => void) {
  if (Math.random() >= chance) return
  const go = () => {
    if (document.hidden) return document.addEventListener('visibilitychange', go, { once: true })
    if (!away()) fire()
  }
  setTimeout(go, (from + Math.random() * (to - from)) * 1000)
}

/** Runs `fire` with the day of a date the visitor clicks, unless the date is part of a link or a dialog. */
export function onDate(fire: (day: string) => void) {
  document.addEventListener('click', (e) => {
    const time = (e.target as Element).closest?.('time[datetime]')
    if (!time || time.closest('a, dialog') || !getSelection()?.isCollapsed) return
    const day = time.getAttribute('datetime')!
    if (/^\d{4}-\d{2}-\d{2}$/.test(day)) fire(day)
  })
}

/** Runs `fire` with each element the page marks data-egg="name". */
export function onMark(name: string, fire: (el: HTMLElement) => void) {
  for (const el of document.querySelectorAll<HTMLElement>(`[data-egg="${name}"]`)) fire(el)
}

/**
 * Runs `fire` once the visitor has scrolled to the end of `el` and has had the page in front of
 * them for `ms` in all.
 */
export function afterReading(el: HTMLElement, ms: number, fire: () => void) {
  let seen = 0
  let since = document.hidden ? null : performance.now()
  let reached = false
  let timer = 0
  const spent = () => seen + (since === null ? 0 : performance.now() - since)

  function check() {
    clearTimeout(timer)
    if (!reached) return
    const left = ms - spent()
    if (left > 0) {
      if (since !== null) timer = window.setTimeout(check, left)
      return
    }
    document.removeEventListener('visibilitychange', visibility)
    fire()
  }
  function visibility() {
    if (document.hidden) {
      seen = spent()
      since = null
    } else since = performance.now()
    check()
  }
  document.addEventListener('visibilitychange', visibility)

  // An empty mark just after the element. It counts as reached once it is anywhere above the
  // bottom of the window, so a reader who jumps past it with one key still reaches it.
  const end = document.createElement('div')
  el.after(end)
  const watch = new IntersectionObserver(
    ([entry]) => {
      if (!entry.isIntersecting) return
      watch.disconnect()
      end.remove()
      reached = true
      check()
    },
    { rootMargin: '100000px 0px 0px 0px' },
  )
  watch.observe(end)
}
