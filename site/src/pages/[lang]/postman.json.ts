/**
 * Everything the postman can bring back, by day, in one language: the articles and the
 * short posts. The footer clock fetches this only when its larger view is opened.
 */
import type { APIRoute, GetStaticPaths } from 'astro'
import { LANGS, type Lang } from '../../site.config'
import { allNotes, dateISO, resolved, subtitleFor, titleFor } from '../../lib/articles'
import { category, fill, t } from '../../lib/taxonomy'
import type { Mail, Story } from '../../clocks/museum/mail'

export const getStaticPaths = (() => LANGS.map((lang) => ({ params: { lang } }))) satisfies GetStaticPaths

/** How an article says that its text in this language was translated by AI. */
const MACHINE = /translated (?:from \w+ )?by AI/i

/**
 * The first paragraph of plain prose, cut to a few lines. Headings, quotes, lists, code,
 * media, equations and a line in italics (such as a translation note) are passed over.
 */
function lede(markdown: string, lang: Lang) {
  const max = lang === 'zh' ? 110 : 240
  const paragraphs = markdown.split(/\n\s*\n/).map((p) => p.trim())
  for (const p of paragraphs) {
    if (!p || /^(#|>|<|`{3}|~{3}|\$\$|\||!\[|[-*+] |\d+\. )/.test(p) || p.includes('$')) continue
    if (/^([*_])[^\n]*\1$/.test(p)) continue
    const plain = p
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/<[^>]+>/g, '')
      .replace(/[*_`]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
    if (!plain) continue
    if (plain.length <= max) return plain
    const cut = plain.slice(0, max)
    const space = cut.lastIndexOf(' ')
    return (lang === 'zh' || space < max * 0.7 ? cut : cut.slice(0, space)).replace(/[,.;:，。；：、]$/, '') + '…'
  }
  return undefined
}

export const GET: APIRoute = async ({ params }) => {
  const lang = params.lang as Lang
  const L = t(lang)
  const other = LANGS.find((l) => l !== lang)!
  const mail: Mail = {}
  const post = (day: string, story: Story) => (mail[day] ??= []).push(story)

  for (const a of await resolved(lang)) {
    const own = a.lang === lang
    const body = a.entry.body ?? ''
    post(dateISO(a.entry.data.date), {
      kind: 'article',
      title: titleFor(a, lang),
      deck: subtitleFor(a, lang),
      category: a.entry.data.category,
      topic: category(a.entry.data.category).name[lang],
      minutes: a.minutes,
      href: `/${lang}/p/${a.slug}/`,
      lede: own ? lede(body, lang) : undefined,
      only: own ? undefined : fill(L.onlyIn, { lang: L.langName[a.lang] }),
      translated: own && MACHINE.test(body) ? fill(L.translated, { lang: L.langName[other] }) : undefined,
    })
  }

  for (const n of await allNotes(lang)) {
    const id = n.id.split('/').slice(1).join('/')
    const from = n.data.origin === 'both' || n.data.origin === lang ? null : n.data.origin
    post(dateISO(n.data.date), {
      kind: 'note',
      html: n.rendered?.html ?? '',
      href: `/${lang}/posts#${id}`,
      translated: from ? fill(L.translated, { lang: L.langName[from] }) : undefined,
    })
  }

  return new Response(JSON.stringify(mail), { headers: { 'content-type': 'application/json' } })
}
