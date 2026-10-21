/** Opens the cloister clock tower over the page. Without WebGL 2 the clock beside the reading time opens nothing. */
import type { Lang } from '../site.config'
import { hasWebGL2, openFrame } from '../eggs/frame'
import { WORDS } from './app/words.js'

export function openTower(lang: Lang) {
  if (!hasWebGL2()) return
  const { title, close } = WORDS[lang]
  openFrame({ egg: 'egg-tower', src: `/${lang}/tower/`, title, close, ground: '#eee9df' })
}
