/**
 * What the clock shows, derived from the time alone. The scene draws these values and
 * animates between them; nothing here depends on what was shown before, so any moment
 * can be drawn directly, on load or after a jump.
 */
import type { Snapshot } from './time'

/** The Postman: the hour is the street, the minute the house number. */
export function postman(t: Snapshot) {
  return { street: t.h, number: t.m }
}
