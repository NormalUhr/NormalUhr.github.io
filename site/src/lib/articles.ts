import { getCollection, type CollectionEntry } from 'astro:content'
import type { Lang } from '../site.config'
import { LANGS } from '../site.config'
import { SERIES } from './taxonomy'

export type ArticleEntry = CollectionEntry<'articles'>
export type NoteEntry = CollectionEntry<'notes'>

export interface Article {
  slug: string
  lang: Lang
  entry: ArticleEntry
  words: number
  minutes: number
  /** Display equations, counted from the source so a list row can advertise them. */
  eqns: number
}

/** "en/grpo" -> { lang: "en", slug: "grpo" } */
function split(id: string): { lang: Lang; slug: string } | null {
  const [lang, ...rest] = id.split('/')
  if (!LANGS.includes(lang as Lang) || !rest.length) return null
  return { lang: lang as Lang, slug: rest.join('/') }
}

/** Counts CJK characters and Latin words, so both languages get a sane reading time. */
function measure(markdown: string) {
  const plain = markdown
    .replace(/(^|\n)(```|~~~)[\s\S]*?\n\2/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[#*_>|`]/g, ' ')
  const cjk = (plain.match(/[一-鿿]/g) || []).length
  const latin = (plain.match(/[A-Za-z][A-Za-z'’-]*/g) || []).length
  const words = cjk + latin
  const eqns = Math.floor((markdown.match(/\$\$/g) || []).length / 2)
  return { words, eqns, minutes: Math.max(1, Math.round(words / (cjk > latin ? 400 : 220))) }
}

let cache: Article[] | null = null

export async function allArticles(): Promise<Article[]> {
  if (cache) return cache
  const entries = await getCollection('articles', ({ data }) => !data.draft)
  const out: Article[] = []
  for (const entry of entries) {
    const parts = split(entry.id)
    if (!parts) throw new Error(`article id "${entry.id}" is not <lang>/<slug>`)
    out.push({ ...parts, entry, ...measure(entry.body ?? '') })
  }
  out.sort((a, b) => b.entry.data.date.getTime() - a.entry.data.date.getTime())
  cache = out
  return out
}

/** All languages a slug exists in, keyed by slug. */
export async function bySlug(): Promise<Map<string, Partial<Record<Lang, Article>>>> {
  const map = new Map<string, Partial<Record<Lang, Article>>>()
  for (const a of await allArticles()) {
    const bucket = map.get(a.slug) ?? {}
    bucket[a.lang] = a
    map.set(a.slug, bucket)
  }
  return map
}

/**
 * One row per slug, in the reader's language where it exists and in whatever language
 * it does exist otherwise. This is the fallback rule the whole site runs on.
 */
export async function resolved(lang: Lang): Promise<Array<Article & { langs: Lang[] }>> {
  const map = await bySlug()
  const out: Array<Article & { langs: Lang[] }> = []
  for (const [, bucket] of map) {
    const langs = LANGS.filter((l) => bucket[l])
    const pick = bucket[lang] ?? bucket[langs[0]]
    if (pick) out.push({ ...pick, langs })
  }
  out.sort((a, b) => b.entry.data.date.getTime() - a.entry.data.date.getTime())
  return out
}

/**
 * The article's name in the reader's language. Every article carries a name in both,
 * even when its body exists in only one, so a list never mixes scripts.
 */
export function titleFor(a: Article, lang: Lang): string {
  if (a.lang === lang) return a.entry.data.title
  return a.entry.data.titleAlt ?? a.entry.data.title
}

export const isArchive = (a: Article) => a.entry.data.category === 'archive'

export async function liveArticles(lang: Lang) {
  return (await resolved(lang)).filter((a) => !isArchive(a))
}
export async function archiveArticles(lang: Lang) {
  return (await resolved(lang)).filter(isArchive)
}

export async function seriesMembers(lang: Lang, id: string) {
  return (await resolved(lang))
    .filter((a) => a.entry.data.series === id)
    .sort((a, b) => (a.entry.data.part ?? 99) - (b.entry.data.part ?? 99) || a.entry.data.date.getTime() - b.entry.data.date.getTime())
}

export async function seriesCounts(lang: Lang) {
  const counts = new Map<string, number>()
  for (const s of SERIES) counts.set(s.id, (await seriesMembers(lang, s.id)).length)
  return counts
}

export async function categoryCounts(lang: Lang) {
  const counts = new Map<string, number>()
  for (const a of await resolved(lang)) {
    counts.set(a.entry.data.category, (counts.get(a.entry.data.category) ?? 0) + 1)
  }
  return counts
}

export async function tagCounts(lang: Lang) {
  const counts = new Map<string, number>()
  for (const a of await resolved(lang)) {
    for (const tag of a.entry.data.tags) counts.set(tag, (counts.get(tag) ?? 0) + 1)
  }
  return counts
}

/**
 * Short posts in one language. Every id must exist in both, so a post is never
 * published to half the audience; a missing translation fails the build.
 */
export async function allNotes(lang: Lang): Promise<NoteEntry[]> {
  const notes = await getCollection('notes', ({ data }) => !data.draft)
  const byId = new Map<string, Set<Lang>>()
  for (const nt of notes) {
    const parts = split(nt.id)
    if (!parts) throw new Error(`note id "${nt.id}" is not <lang>/<id>`)
    const set = byId.get(parts.slug) ?? new Set<Lang>()
    set.add(parts.lang)
    byId.set(parts.slug, set)
  }
  for (const [id, langs] of byId) {
    const missing = LANGS.filter((l) => !langs.has(l))
    if (missing.length) throw new Error(`short post "${id}" is missing its ${missing.join(' and ')} version`)
    const origins = new Set(notes.filter((nt) => nt.id.endsWith(`/${id}`)).map((nt) => nt.data.origin))
    if (origins.size > 1) throw new Error(`short post "${id}" disagrees about its origin language: ${[...origins].join(', ')}`)
  }
  return notes
    .filter((nt) => nt.id.startsWith(`${lang}/`))
    .sort((a, b) => b.data.date.getTime() - a.data.date.getTime())
}

/** Sibling articles in the reader's language, newest first, for the prev/next pager. */
export async function neighbours(lang: Lang, slug: string) {
  const list = await resolved(lang)
  const i = list.findIndex((a) => a.slug === slug)
  return { newer: i > 0 ? list[i - 1] : null, older: i >= 0 && i < list.length - 1 ? list[i + 1] : null }
}

export const dateISO = (d: Date) => d.toISOString().slice(0, 10)
