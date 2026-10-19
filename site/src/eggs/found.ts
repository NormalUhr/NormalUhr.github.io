/**
 * Counts each Easter egg a visitor opens, as a GoatCounter event. GoatCounter lists the events
 * with the pages, and counts each one at most once per visit, which also adds to the site total.
 * Its script counts nothing on a local address, so previews and checks send nothing.
 */
const EGGS = {
  'egg-postman-walk': 'Postman town, from the postman walking past',
  'egg-postman-date': 'Postman town, from a date',
  'egg-postman-lost': 'Postman town, from the clock on the missing page',
  'egg-postman-letter': 'Postman town, from the letter after an article',
  'egg-woodland': 'Woodland clock, from the clock in the masthead',
} as const

export type Egg = keyof typeof EGGS

declare global {
  interface Window {
    goatcounter?: { count?: (vars: { path: string; title: string; event: boolean }) => void }
  }
}

export function found(egg: Egg) {
  window.goatcounter?.count?.({ path: egg, title: EGGS[egg], event: true })
}
