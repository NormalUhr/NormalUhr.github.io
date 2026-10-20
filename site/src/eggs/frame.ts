/**
 * Opens an Easter egg that runs as a page of its own, such as the woodland clock, over the page
 * in a frame that fills the window. Such a page takes over its window and has no way to stop, so
 * closing removes the frame, which stops it and frees its graphics memory. The egg's page talks
 * to the site with messages:
 * - 'frame:close' asks the site to close the frame, because Escape pressed inside the frame
 *   reaches only the egg's page.
 * - 'frame:hold' says that the page has work to finish, such as saving, before it is removed.
 *   The site then answers a close with 'frame:leave', and removes the frame when the page
 *   answers 'frame:left', or after LEAVE_MS.
 */
import { adopt, html, svg } from '../clocks/scene'
import { found, type Egg } from './found'
import css from './frame.css?inline'

/** How long a page that asked the site to hold may take to finish after the visitor closes it. */
const LEAVE_MS = 3000

export interface Framed {
  egg: Egg
  /** The address of the egg's page. */
  src: string
  /** The name of the egg's page, for the dialog and the frame. */
  title: string
  /** The name of the close button. */
  close: string
  /** The colour of the egg's page while it loads, so that opening does not flash. */
  ground: string
}

/** The eggs that open in a frame draw with WebGL 2. Without it their trigger stays what it looks like. */
export function hasWebGL2() {
  const gl = document.createElement('canvas').getContext('webgl2')
  gl?.getExtension('WEBGL_lose_context')?.loseContext()
  return gl !== null
}

export function openFrame({ egg, src, title, close, ground }: Framed) {
  // A second click while the egg's code is still loading would open a second frame over the first.
  // A frame that is still finishing after a close holds the next one back for the same reason.
  if (document.querySelector('.frdlg')) return
  adopt(css)
  const frame = html('iframe', { class: 'fr-frame', src, title })
  const shut = html(
    'button',
    { class: 'fr-close', type: 'button', 'aria-label': close },
    svg('svg', { viewBox: '0 0 12 12', width: 12, height: 12, 'aria-hidden': 'true' }, svg('path', { d: 'M2 2L10 10M10 2L2 10' })),
  )
  const box = html('dialog', { class: 'frdlg', 'aria-label': title, style: `--ground: ${ground}` }, frame, shut)
  let hold = false
  let timer = 0
  const gone = () => {
    clearTimeout(timer)
    window.removeEventListener('message', heard)
    box.remove()
  }
  const heard = (e: MessageEvent) => {
    if (e.origin !== location.origin || e.source !== frame.contentWindow) return
    if (e.data === 'frame:close') box.close()
    else if (e.data === 'frame:hold') hold = true
    else if (e.data === 'frame:left' && !box.open) gone()
  }
  shut.addEventListener('click', () => box.close())
  box.addEventListener('close', () => {
    if (!hold) return gone()
    frame.contentWindow?.postMessage('frame:leave', location.origin)
    timer = window.setTimeout(gone, LEAVE_MS)
  })
  window.addEventListener('message', heard)
  document.body.append(box)
  box.showModal()
  found(egg)
  // The egg's keys, such as H for the full view, work once its page has the focus.
  frame.addEventListener('load', () => frame.focus(), { once: true })
}
