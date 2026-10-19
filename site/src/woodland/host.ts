/**
 * Opens the woodland clock over the page, in a frame that fills the window. The woodland runs
 * as a page of its own because it takes over its window and has no way to stop. Closing the
 * frame removes it, which stops the woodland and frees its graphics memory at once.
 */
import type { Lang } from '../site.config'
import { adopt, html, svg } from '../clocks/scene'
import { found } from '../eggs/found'
import { WORDS } from './app/words.js'
import css from './host.css?inline'

let dialog: HTMLDialogElement | null = null
let frame: HTMLIFrameElement | null = null

/** The woodland needs WebGL 2 and WebAssembly. Without them the clock in the masthead stays a clock. */
function supported() {
  if (typeof WebAssembly !== 'object') return false
  const gl = document.createElement('canvas').getContext('webgl2')
  gl?.getExtension('WEBGL_lose_context')?.loseContext()
  return gl !== null
}

function build(lang: Lang) {
  adopt(css)
  const shut = html(
    'button',
    { class: 'wd-close', type: 'button', 'aria-label': WORDS[lang].close },
    svg('svg', { viewBox: '0 0 12 12', width: 12, height: 12, 'aria-hidden': 'true' }, svg('path', { d: 'M2 2L10 10M10 2L2 10' })),
  )
  const box = html('dialog', { class: 'wooddlg', 'aria-label': WORDS[lang].title }, shut)
  document.body.append(box)
  shut.addEventListener('click', () => box.close())
  box.addEventListener('close', () => {
    frame?.remove()
    frame = null
  })
  // Escape pressed inside the woodland reaches the woodland's own page, which passes it on.
  window.addEventListener('message', (e) => {
    if (e.origin === location.origin && e.source === frame?.contentWindow && e.data === 'woodland:close') box.close()
  })
  return box
}

/** Opens a new woodland in the reader's language. */
export function openWoodland(lang: Lang) {
  if (!supported()) return
  const box = (dialog ??= build(lang))
  if (box.open) return
  frame = html('iframe', { class: 'wd-frame', src: `/${lang}/woodland/`, title: WORDS[lang].title })
  box.prepend(frame)
  box.showModal()
  found('egg-woodland')
  // The keys of the woodland, such as H for the full view, work once its page has the focus.
  const opened = frame
  opened.addEventListener('load', () => opened.focus(), { once: true })
}
