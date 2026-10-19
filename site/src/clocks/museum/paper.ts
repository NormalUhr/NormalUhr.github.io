/**
 * The newspaper the postman hands over: the front page of one day, with the articles and
 * short posts that came out on it, written by hand. It opens over the town as a dialog of
 * its own, so the Escape key and the backdrop close the paper before they close the town.
 */
import type { Lang } from '../../site.config'
import { fill, html, svg, uid } from '../scene'
import copy from './copy'
import { between, parts, weather, type Day } from './days'
import type { Article, Note, Story } from './mail'

/** One day's front page. */
export interface Issue {
  lang: Lang
  day: Day
  /** The year of the first issue, which numbers the volumes. */
  since: number
  stories: Story[]
}

export interface PaperOptions extends Issue {
  /** The element the dialog is added to. */
  host: HTMLElement
  /** Where on screen the paper comes from. */
  from: DOMRect
  still: boolean
  onClose(): void
}

export interface Paper {
  /** Closes the paper, folding it away unless `now` is true. */
  close(now?: boolean): void
}

/**
 * The hands the paper is written in, from Google Fonts like the site's own type: one for Latin
 * letters and figures and one for Chinese. English is written with a pen. Chinese is written with
 * a brush pen, and a brush script writes the English words and figures among it. An English paper
 * keeps the brush pen for a story that exists only in Chinese.
 */
const HANDS: Record<Lang, { latin: string; han: string }> = {
  en: { latin: 'Grape Nuts', han: 'Ma Shan Zheng' },
  zh: { latin: 'Caveat Brush', han: 'Ma Shan Zheng' },
}
/** Chinese characters and the punctuation that goes with them, which the Latin hands lack. */
const HAN = /[\p{Script=Han}\u3000-\u303f\uff00-\uffef]/gu
const GOOGLE = 'https://fonts.googleapis.com/css2'
const family = (name: string) => `family=${name.replaceAll(' ', '+')}`

/** Adds a stylesheet. `ready` settles once it has loaded or failed. */
function stylesheet(href: string) {
  const link = html('link', { rel: 'stylesheet', href })
  const ready = new Promise((done) => {
    link.addEventListener('load', done)
    link.addEventListener('error', done)
  })
  document.head.append(link)
  return { href, link, ready }
}

const pens: Partial<Record<Lang, Promise<unknown>>> = {}
let brush: ReturnType<typeof stylesheet> | undefined

/**
 * Fetches the handwriting an issue needs, so its paper opens already written. The whole Chinese
 * hand is megabytes, so Google sends a font with only the characters on this front page, or its
 * usual slices when the page has too many. Every paper's font has the same family name and one
 * paper is open at a time, so the last paper's font is removed.
 */
export function ink(issue: Issue) {
  const { latin, han } = HANDS[issue.lang]
  const words = write(issue).textContent ?? ''
  const pen = (pens[issue.lang] ??= stylesheet(`${GOOGLE}?${family(latin)}&display=swap`).ready)
  pen.then(() => document.fonts.load(`16px "${latin}"`, words)).catch(() => {})

  const chinese = [...new Set(words.match(HAN))].join('')
  if (!chinese) return
  const href = `${GOOGLE}?${family(han)}&text=${encodeURIComponent(chinese)}&display=swap`
  if (brush?.href !== href) {
    brush?.link.remove()
    brush = stylesheet(href)
  }
  brush.ready.then(() => document.fonts.load(`16px "${han}"`, chinese)).catch(() => {})
}

/** Writes out an issue's front page. `name` is the id of its masthead, which titles the dialog. */
function write(o: Issue, name?: string) {
  const T = copy[o.lang]
  const { y, m, d } = parts(o.day)
  const date = new Intl.DateTimeFormat(T.locale, { dateStyle: 'full', timeZone: 'UTC' }).format(Date.UTC(y, m - 1, d))

  const article = (a: Article) =>
    html(
      'section',
      { class: 'np-story np-article' },
      html('p', { class: 'np-kicker' }, html('span', {}, T.article), html('span', {}, a.topic)),
      html('h3', { class: 'np-head' }, html('a', { href: a.href }, a.title)),
      a.deck && html('p', { class: 'np-deck' }, a.deck),
      a.lede && html('p', { class: 'np-lede', lang: o.lang }, a.lede),
      a.translated && html('p', { class: 'np-xlat' }, a.translated),
      html('p', { class: 'np-meta' }, html('span', {}, fill(T.minutes, { n: a.minutes })), a.only && html('span', {}, a.only)),
      html('a', { class: 'np-go', href: a.href }, T.read),
    )

  const note = (n: Note) => {
    const body = html('div', { class: 'np-body', lang: o.lang })
    // The post's own markup, rendered by the site at build time.
    body.innerHTML = n.html
    return html(
      'section',
      { class: 'np-story np-note' },
      html('p', { class: 'np-kicker' }, html('span', {}, T.note)),
      body,
      n.translated && html('p', { class: 'np-xlat' }, n.translated),
      html('a', { class: 'np-go np-quiet', href: n.href }, T.open),
    )
  }

  return html(
    'article',
    { class: 'np-sheet', tabindex: '-1', autofocus: true },
    html(
      'header',
      { class: 'np-mast' },
      html(
        'p',
        { class: 'np-ears' },
        html('span', {}, fill(T.issue, { vol: y - o.since + 1, no: between(`${y}-01-01`, o.day) + 1 })),
        html('span', {}, T.weather[weather(o.day)]),
      ),
      html('h2', { class: 'np-name', id: name }, T.paper),
      html('p', { class: 'np-date' }, html('time', { datetime: o.day }, date)),
    ),
    ...o.stories.map((s) => (s.kind === 'article' ? article(s) : note(s))),
  )
}

export function openPaper(o: PaperOptions): Paper {
  const name = uid('np-name')
  const shut = html(
    'button',
    { class: 'np-close', type: 'button', 'aria-label': copy[o.lang].close },
    svg('svg', { viewBox: '0 0 12 12', width: 12, height: 12, 'aria-hidden': 'true' }, svg('path', { d: 'M2 2L10 10M10 2L2 10' })),
  )
  const sheet = write(o, name)
  sheet.prepend(shut)
  const dialog = html('dialog', { class: 'np', 'aria-labelledby': name }, sheet)
  o.host.append(dialog)
  dialog.showModal()

  // Out of the postman's hand, spinning, as a paper comes off the press in an old film.
  const box = dialog.getBoundingClientRect()
  const dx = o.from.left + o.from.width / 2 - (box.left + box.width / 2)
  const dy = o.from.top + o.from.height / 2 - (box.top + box.height / 2)
  if (o.still) dialog.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160 })
  else {
    dialog.animate(
      [
        { transform: `translate(${dx}px, ${dy}px) scale(0.03) rotate(-560deg)`, opacity: 0 },
        { opacity: 1, offset: 0.18 },
        { transform: 'translate(0, 0) scale(1.04) rotate(3deg)', offset: 0.78 },
        { transform: 'none' },
      ],
      { duration: 880, easing: 'cubic-bezier(.2, .7, .3, 1)' },
    )
  }

  let closing = false
  function close(now = false) {
    if (closing) return
    closing = true
    const done = () => {
      dialog.close()
      dialog.remove()
      o.onClose()
    }
    if (now || o.still) return done()
    for (const a of dialog.getAnimations()) a.cancel()
    dialog
      .animate([{ transform: 'none', opacity: 1 }, { transform: 'translateY(24px) scale(0.94) rotate(-2deg)', opacity: 0 }], { duration: 220, easing: 'ease-in' })
      .finished.then(done, done)
  }

  shut.addEventListener('click', () => close())
  // The sheet fills the dialog, so a click on the dialog itself landed on the backdrop.
  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) close()
  })
  dialog.addEventListener('cancel', (e) => {
    e.preventDefault()
    close()
  })
  return { close }
}
