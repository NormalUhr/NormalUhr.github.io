import type { APIRoute } from 'astro'
import { site, DEFAULT_LANG } from '../site.config'
import { resolved, dateISO } from '../lib/articles'

const escape = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]!))

export const GET: APIRoute = async ({ site: base }) => {
  const origin = (base ?? new URL('https://normaluhr.github.io')).origin
  const items = (await resolved(DEFAULT_LANG))
    .map((a) => `    <item>
      <title>${escape(a.entry.data.title)}</title>
      <link>${origin}/${DEFAULT_LANG}/p/${a.slug}</link>
      <guid isPermaLink="true">${origin}/${DEFAULT_LANG}/p/${a.slug}</guid>
      <pubDate>${a.entry.data.date.toUTCString()}</pubDate>
      ${a.entry.data.subtitle ? `<description>${escape(a.entry.data.subtitle)}</description>` : ''}
    </item>`)
    .join('\n')

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
    <title>${escape(site.name[DEFAULT_LANG])}</title>
    <link>${origin}/</link>
    <description>${escape(site.tagline[DEFAULT_LANG])}</description>
    <language>en</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
${items}
</channel></rss>
`
  return new Response(xml, { headers: { 'content-type': 'application/xml; charset=utf-8' } })
}
