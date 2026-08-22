// One-off migration: Jekyll _posts/ + _includes/posts/ -> Astro content collections.
//
// Article bodies are copied byte for byte. The only things this script decides are
// front-matter fields, and it writes a report of every decision so they can be checked.
//
//   node scripts/migrate.mjs [--dry]
import fs from 'node:fs'
import path from 'node:path'

const SITE = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..')
const REPO = path.resolve(SITE, '..')
const OUT = path.join(SITE, 'src', 'content', 'articles')
const DRY = process.argv.includes('--dry')

/** Practice notes the author asked to leave behind. */
const DROP = [/刷题/, /LeetCode/i, /单调栈/, /Bitset/]

/**
 * Per-slug taxonomy. Category and series are editorial calls that do not exist in the
 * old front matter, so they live here; everything else is read from the source.
 * Tags are deliberately few: at most two, and most posts have none.
 */
const TAX = {
  'vogel-im-kafig':      { cat: 'music',         tags: ['Sawano'] },
  'ai-infra':            { cat: 'systems',       tags: [],       lang: 'en' },
  'grpo-dapo-gspo':      { cat: 'post-training', tags: ['GRPO'],   series: ['grpo-line', 2] },
  'kl':                  { cat: 'post-training', tags: ['GRPO'], series: ['grpo-line', 3] },
  'decorators':          { cat: 'systems',       tags: ['Python'] },
  'bauklotze':           { cat: 'music',         tags: ['Sawano'] },
  'dualpipe':            { cat: 'systems',       tags: ['Distributed'],   series: ['deepseek', 3] },
  'rlhf':                { cat: 'post-training', tags: ['RLHF'] },
  'grpo':                { cat: 'post-training', tags: ['GRPO', 'RLHF'],   series: ['grpo-line', 1] },
  'mla':                 { cat: 'systems',       tags: ['KV-Cache'],   series: ['deepseek', 2] },
  'deepseek-r1':         { cat: 'post-training', tags: ['GRPO'],   series: ['deepseek', 1] },
  'moe-load-balancing':  { cat: 'systems',       tags: ['MoE'] },
  'unlearning-pitfalls': { cat: 'post-training', tags: ['Unlearning'] },
  'effective-c':         { cat: 'archive',       tags: ['C++'] },
  'lie-group':           { cat: 'archive',       tags: [] },
}


/**
 * The other language's title, for the 21 articles that exist in one language only.
 * Every article's name is shown in both languages even when its body is not, so these
 * are translations of the author's own titles and are the one place in the import that
 * contains words he did not write. They are listed together so they can be reviewed
 * and edited in one pass.
 */
const TITLE_ALT = {
  // Chinese-only articles: the English name.
  'vogel-im-kafig':  'Vogel im Käfig: How Sawano Builds Hope and Despair from the Smallest Interval',
  'bauklotze':       'Bauklötze: A Musical Dissection',
  'lie-group':       'Visual SLAM Notes: Lie Groups and Lie Algebras',
  'auto':            'Using auto for Variables',
  'auto-function':   'Using auto for Functions',
  'dynamic':         'Dynamic and Static in C++',
  'enum':            'How Strong Is a Strongly Typed enum?',
  'nullptr':         'Do You Really Understand nullptr?',
  'optional':        'std::optional in C++17',
  'override-final':  'override and final Make Virtual Functions Safer',
  'passkey-idiom':   'The Passkey Idiom in the Factory Pattern',
  'rvalue':          'Rvalue References and Move Semantics',
  'top-const':       'Top-level and Low-level const',
  'void':            'Getting to the Bottom of void*',
  'const-1':         'The const Truth: Syntactic and Semantic const',
  'const-2':         'The const Truth: mutable',
  'const-3':         'The const Truth: Thread Safety of const',
  'constexpr-1':     'From Compile-time Constants to constexpr (1)',
  'constexpr-2':     'From Compile-time Constants to constexpr (2)',
  'constexpr-3':     'From Compile-time Constants to constexpr (3)',
  'constexpr-4':     'constexpr in C++17',
  // English-only article: the Chinese name.
  'ai-infra':        'AI Infra 的角色转变：从辅助转为主力输出',
}

const SERIES_BY_TITLE = [
  [/从编译期常量到constexpr（一）/, 'constexpr', 1],
  [/从编译期常量到constexpr（二）/, 'constexpr', 2],
  [/从编译期常量到constexpr（三）/, 'constexpr', 3],
  [/C\+\+17中的constexpr/, 'constexpr', 4],
  [/const真理大讨论之 语法和语义const/, 'const', 1],
  [/const真理大讨论之 mutable/, 'const', 2],
  [/const真理大讨论之 const的线程安全/, 'const', 3],
]

function parseFrontMatter(raw) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(raw)
  if (!m) return { data: {}, body: raw }
  const data = {}
  let key = null
  for (const line of m[1].split('\n')) {
    const item = /^\s+-\s+(.*)$/.exec(line)
    if (item && key) { (data[key] ||= []).push(item[1].trim()); continue }
    const kv = /^([A-Za-z_][\w-]*):\s*(.*)$/.exec(line)
    if (!kv) continue
    key = kv[1]
    const v = kv[2].trim().replace(/^["'](.*)["']$/, '$1')
    data[key] = v === '' ? [] : v
  }
  return { data, body: m[2] }
}

/**
 * The old front matter packed both languages into `title` + `subtitle`, with a marker
 * saying which order: "… [En/中]" means title is English and subtitle Chinese,
 * "… [中/En]" the reverse. Both strings are the author's own words; this only routes
 * them to the right language file.
 */
function splitTitles(title, subtitle) {
  const marker = /\s*\[(En\/中|中\/En|en\/zh|zh\/en)\]\s*$/i.exec(title)
  if (!marker) return null
  const clean = title.replace(marker[0], '').trim()
  const englishFirst = /^en/i.test(marker[1])
  return englishFirst
    ? { en: { title: clean, subtitle: '' }, zh: { title: subtitle, subtitle: '' } }
    : { zh: { title: clean, subtitle: '' }, en: { title: subtitle, subtitle: '' } }
}

const yamlString = (s) => `"${String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`

function frontMatter(f) {
  const lines = [`title: ${yamlString(f.title)}`]
  if (f.titleAlt) lines.push(`titleAlt: ${yamlString(f.titleAlt)}`)
  if (f.subtitle) lines.push(`subtitle: ${yamlString(f.subtitle)}`)
  lines.push(`date: ${f.date}`)
  lines.push(`category: ${f.category}`)
  if (f.series) lines.push(`series: ${f.series.id}`, `part: ${f.series.part ?? 'null'}`)
  if (f.tags.length) lines.push(`tags: [${f.tags.map(yamlString).join(', ')}]`)
  return `---\n${lines.join('\n')}\n---\n`
}

const report = []
const redirects = {}
let written = 0, dropped = 0

/**
 * The only edit made to any article body. Four <source> tags in the music post are
 * written ".//videos/..." instead of "/videos/...", which resolves relative to the page
 * and 404s. Every other media path in the archive is already absolute.
 */
function fixMediaPaths(body, key) {
  let out = body
  const rel = /(src=")\.\/+(videos|img)\//g
  if (rel.test(out)) {
    const n = (out.match(/src="\.\/+(videos|img)\//g) || []).length
    out = out.replace(/(src=")\.\/+(videos|img)\//g, '$1/$2/')
    report.push(`FIXPATH ${key}  ${n} path(s) corrected from ".//" to "/"`)
  }
  const spaced = /(src="[^"]*)\.m p4"/g
  if (spaced.test(out)) {
    out = out.replace(/(src="[^"]*)\.m p4"/g, '$1.mp4"')
    report.push(`FIXPATH ${key}  extension typo ".m p4" corrected to ".mp4"`)
  }
  return out
}

if (!DRY) {
  for (const lang of ['en', 'zh']) fs.rmSync(path.join(OUT, lang), { recursive: true, force: true })
  for (const lang of ['en', 'zh']) fs.mkdirSync(path.join(OUT, lang), { recursive: true })
}

for (const file of fs.readdirSync(path.join(REPO, '_posts')).filter((f) => f.endsWith('.md')).sort()) {
  const key = file.replace(/\.md$/, '')
  const raw = fs.readFileSync(path.join(REPO, '_posts', file), 'utf8')
  const { data, body } = parseFrontMatter(raw)
  const title = data.title || key

  if (DROP.some((re) => re.test(title))) {
    report.push(`DROP    ${key}  (${title})`)
    dropped += 1
    continue
  }

  const slug = key.replace(/^\d{4}-\d{1,2}-\d{1,2}-/, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
  const date = String(data.date).slice(0, 10)
  // Jekyll's `permalink: pretty` produced /YYYY/MM/DD/<file-slug>/ from the filename,
  // which is what any existing inbound link points at.
  const fileSlug = key.replace(/^\d{4}-\d{1,2}-\d{1,2}-/, '')
  const [y, m, d] = date.split('-')
  const landing = data.layout === 'post_lang' || (TAX[slug] ?? {}).lang === 'en' ? 'en' : 'zh'
  redirects[`/${y}/${m}/${d}/${fileSlug}`] = `/${landing}/p/${slug}`
  const tax = TAX[slug] ?? {}

  let series = tax.series ? { id: tax.series[0], part: tax.series[1] } : null
  if (!series) {
    for (const [re, id, part] of SERIES_BY_TITLE) if (re.test(title)) series = { id, part }
  }
  if (!series && /^漫谈C\+\+/.test(title)) series = { id: 'mantan-cpp', part: null }

  const base = {
    date,
    category: tax.cat ?? 'archive',
    series,
    tags: tax.tags ?? [],
  }

  const split = splitTitles(title, data.subtitle || '')
  const bilingual = data.layout === 'post_lang'

  if (bilingual) {
    if (!split) throw new Error(`${key}: bilingual post without an [En/中] marker in the title`)
    for (const lang of ['en', 'zh']) {
      const src = path.join(REPO, '_includes', 'posts', `${key}_${lang}.md`)
      if (!fs.existsSync(src)) { report.push(`WARN    ${key}  missing ${lang} body`); continue }
      const text = frontMatter({ ...base, ...split[lang] }) + '\n' + fixMediaPaths(fs.readFileSync(src, 'utf8'), `${key}_${lang}`)
      if (!DRY) fs.writeFileSync(path.join(OUT, lang, `${slug}.md`), text)
      written += 1
    }
    report.push(`OK      ${slug.padEnd(22)} en+zh  ${base.category.padEnd(14)} ${series ? series.id + (series.part ? '#' + series.part : '') : '-'}`)
  } else {
    const lang = tax.lang ?? 'zh'
    const titleAlt = TITLE_ALT[slug]
    if (!titleAlt) throw new Error(`${slug}: single-language article with no TITLE_ALT entry`)
    const text = frontMatter({ ...base, title, titleAlt, subtitle: data.subtitle || '' }) + '\n' + fixMediaPaths(body, key)
    if (!DRY) fs.writeFileSync(path.join(OUT, lang, `${slug}.md`), text)
    written += 1
    report.push(`OK      ${slug.padEnd(22)} ${lang}     ${base.category.padEnd(14)} ${series ? series.id + (series.part ? '#' + series.part : '') : '-'}`)
  }
}

const summary = [
  `files written: ${written}`,
  `posts dropped: ${dropped}`,
  '',
  ...report.sort(),
].join('\n')

console.log(summary)
if (!DRY) {
  fs.writeFileSync(path.join(SITE, 'scripts', 'migration-report.txt'), summary + '\n')
  fs.mkdirSync(path.join(SITE, 'src', 'generated'), { recursive: true })
  fs.writeFileSync(
    path.join(SITE, 'src', 'generated', 'legacy-redirects.mjs'),
    '// Generated by scripts/migrate.mjs. Old Jekyll permalinks -> new article URLs.\n' +
      'export default ' + JSON.stringify(redirects, null, 2) + '\n',
  )
  console.log(`\nlegacy redirects written: ${Object.keys(redirects).length}`)
}
