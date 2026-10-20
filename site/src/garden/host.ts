/** Opens the karesansui garden over the page. Without WebGL 2 its doors in the footer open nothing. */
import type { Lang } from '../site.config'
import { hasWebGL2, openFrame } from '../eggs/frame'
import type { Egg } from '../eggs/found'
import { WORDS } from './app/words.js'

/** `door` is the Easter egg that opened it. */
export function openGarden(lang: Lang, door: Egg) {
  if (!hasWebGL2()) return
  const { title, close } = WORDS[lang]
  openFrame({ egg: door, src: `/${lang}/garden/`, title, close, ground: '#273b2e' })
}
