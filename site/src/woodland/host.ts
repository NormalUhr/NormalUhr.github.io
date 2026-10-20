/** Opens the woodland clock over the page. Without WebGL 2 and WebAssembly the clock in the masthead stays a clock. */
import type { Lang } from '../site.config'
import { hasWebGL2, openFrame } from '../eggs/frame'
import { WORDS } from './app/words.js'

export function openWoodland(lang: Lang) {
  if (typeof WebAssembly !== 'object' || !hasWebGL2()) return
  const { title, close } = WORDS[lang]
  openFrame({ egg: 'egg-woodland', src: `/${lang}/woodland/`, title, close, ground: '#242c21' })
}
