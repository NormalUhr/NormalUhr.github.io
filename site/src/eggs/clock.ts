/** Runs `show` with the visitor's time now and again at the start of every minute. */
export function everyMinute(show: (now: Date) => void) {
  let timer = 0
  const tick = () => {
    clearTimeout(timer)
    const now = new Date()
    show(now)
    timer = window.setTimeout(tick, 60_000 - (now.getTime() % 60_000))
  }
  // A background tab runs its timers late, so the time is read again when the tab comes back.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) tick()
  })
  tick()
}

const pad = (n: number) => String(n).padStart(2, '0')
export const hhmm = (now: Date) => `${pad(now.getHours())}:${pad(now.getMinutes())}`

/** The visitor's local time as HH:MM, the way the ants in the woodland lay it out in stones. */
export function keepTime(face: HTMLElement, label: string) {
  everyMinute((now) => {
    face.textContent = hhmm(now)
    face.setAttribute('aria-label', `${label} ${hhmm(now)}`)
  })
}
