/** The visitor's local time as HH:MM, the way the ants in the woodland lay it out in stones. */
export function keepTime(face: HTMLElement, label: string) {
  let timer = 0
  const show = () => {
    clearTimeout(timer)
    const now = new Date()
    const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
    face.textContent = time
    face.setAttribute('aria-label', `${label} ${time}`)
    timer = window.setTimeout(show, 60_000 - (now.getTime() % 60_000))
  }
  // A background tab runs its timers late, so the time is read again when the tab comes back.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) show()
  })
  show()
}
