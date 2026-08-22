// Assembles each theme template into a self-contained HTML file: injects the content
// payload, the fully rendered posts, and a font-inlined KaTeX stylesheet.
import fs from 'node:fs'
import path from 'node:path'
import { renderMarkdown } from './render.mjs'
import { CATEGORIES, ARCHIVE, SERIES, NOTES, COMMENTS, UI, SITE, REPO, loadPosts } from './content.mjs'

const HERE = path.dirname(new URL(import.meta.url).pathname)
const OUT = path.resolve(HERE, '..', 'demos')

/** KaTeX ships woff/woff2/ttf; keep only woff2 and inline it so the page needs no network. */
function katexCss() {
  const dist = path.join(HERE, 'node_modules', 'katex', 'dist')
  const css = fs.readFileSync(path.join(dist, 'katex.min.css'), 'utf8')
  return css
    .replace(/url\(fonts\/([^)]+?)\)\s*format\("([^"]+)"\)/g, (whole, file, format) => {
      if (format !== 'woff2') return 'url(about:invalid)'
      return `url(data:font/woff2;base64,${fs.readFileSync(path.join(dist, 'fonts', file)).toString('base64')}) format("woff2")`
    })
    .replace(/,\s*url\(about:invalid\)/g, '')
    .replace(/src:url\(about:invalid\);?/g, '')
}

async function renderFullPost(key) {
  const out = {}
  for (const lang of ['en', 'zh']) {
    const file = path.join(REPO, '_includes', 'posts', `${key}_${lang}.md`)
    if (fs.existsSync(file)) out[lang] = await renderMarkdown(fs.readFileSync(file, 'utf8'), lang)
  }
  return out
}

export async function build({ quiet = false } = {}) {
  const posts = loadPosts()

  const notes = []
  for (const n of NOTES) notes.push({ ...n, html: (await renderMarkdown(n.body, n.lang)).html })

  const bodies = {
    grpo: await renderFullPost('2025-02-07-grpo'),
    decorators: await renderFullPost('2025-04-10-decorators'),
  }

  const live = posts.filter((p) => p.category !== 'archive')
  const stats = {
    articles: live.length,
    archive: posts.length - live.length,
    notes: notes.length,
    bilingual: posts.filter((p) => p.langs.length > 1).length,
    since: SITE.since,
    visits: SITE.visits,
  }

  const payload = { site: SITE, ui: UI, categories: CATEGORIES, archive: ARCHIVE, series: SERIES, posts, notes, comments: COMMENTS, bodies, stats }
  const json = JSON.stringify(payload).replace(/<\/script/gi, '<\\/script').replace(/<!--/g, '<\\!--')
  const katex = katexCss()

  fs.mkdirSync(OUT, { recursive: true })
  const themes = fs.readdirSync(path.join(HERE, 'themes')).filter((f) => f.endsWith('.html'))
  const written = []
  for (const theme of themes) {
    const src = fs.readFileSync(path.join(HERE, 'themes', theme), 'utf8')
    if (!src.includes('__PAYLOAD__') || !src.includes('__KATEX_CSS__')) {
      throw new Error(`${theme} is missing __PAYLOAD__ or __KATEX_CSS__`)
    }
    const html = src.replace('__KATEX_CSS__', () => katex).replace('__PAYLOAD__', () => json)
    fs.writeFileSync(path.join(OUT, theme), html)
    written.push([theme, Buffer.byteLength(html)])
  }

  if (!quiet) {
    for (const [name, bytes] of written) console.log(`${name.padEnd(14)} ${(bytes / 1024).toFixed(0)} KB`)
    console.log(`\narticles ${stats.articles} · archive ${stats.archive} · short posts ${stats.notes} · bilingual ${stats.bilingual}`)
    console.log('categories:', CATEGORIES.map((c) => `${c.id}(${posts.filter((p) => p.category === c.id).length})`).join(' '))
    console.log('tags:', [...new Set(posts.flatMap((p) => p.tags))].sort().join(', ') || '(none)')
    console.log('grpo:', bodies.grpo.en.eqnCount, 'equations,', bodies.grpo.en.toc.length, 'headings')
  }
  return { posts, stats, written }
}

if (path.resolve(process.argv[1] ?? '') === path.resolve(new URL(import.meta.url).pathname)) {
  await build()
}
