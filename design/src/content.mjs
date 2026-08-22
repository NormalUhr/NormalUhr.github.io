// Content model for the redesign.
// Posts come from the repository's own _posts/ front matter, so browsing is real.
// Short posts, read counts and comments are placeholders: none of them exist on the site yet.
import fs from 'node:fs'
import path from 'node:path'

const HERE = path.dirname(new URL(import.meta.url).pathname)
export const REPO = path.resolve(HERE, '..', '..')

/**
 * Three living categories plus a separate archive. Anything that would end up with a
 * single post does not get its own category; unlearning sits under post-training,
 * which is where it actually belongs.
 */
export const CATEGORIES = [
  {
    id: 'post-training',
    accent: '#5ee39b',
    name: { en: 'Post-training', zh: '后训练' },
    blurb: {
      en: 'Policy gradients, PPO, GRPO, RLHF, unlearning. How a pretrained model is turned into a useful one.',
      zh: '策略梯度、PPO、GRPO、RLHF、反学习。一个预训练模型是怎么被调成一个可用的模型的。',
    },
  },
  {
    id: 'systems',
    accent: '#6cb6ff',
    name: { en: 'Systems', zh: '系统' },
    blurb: {
      en: 'KV-cache, expert routing, pipeline schedules. How the arithmetic actually runs on the hardware.',
      zh: 'KV-cache、专家路由、流水线排布。这些计算最后是怎么在硬件上真正跑起来的。',
    },
  },
  {
    id: 'music',
    accent: '#ffab5e',
    name: { en: 'Music', zh: '音乐' },
    blurb: {
      en: 'Pieces I cannot stop listening to, taken apart on the stave, with the audio in the page.',
      zh: '把反复听的曲子拿到谱面上拆开来看，音频就放在文章里。',
    },
  },
]

export const ARCHIVE = {
  id: 'archive',
  accent: '#8b97a6',
  name: { en: 'Archive', zh: '旧文' },
  blurb: {
    en: '2019 to 2020. C++ internals and Visual SLAM notes. Kept whole and kept indexed, out of the way of everything newer.',
    zh: '2019 年到 2020 年写的东西，C++ 的底层细节和 Visual SLAM 笔记。完整保留并且建好索引，但是不放在显眼的位置。',
  },
}

export const SERIES = [
  {
    id: 'deepseek',
    category: 'post-training',
    name: { en: 'Anatomy of DeepSeek', zh: '解剖 DeepSeek' },
    blurb: {
      en: 'Four passes over one model: what it optimises, how it caches, how it updates, how it schedules.',
      zh: '围绕同一个模型写的四篇：它在优化什么，怎么缓存，怎么更新参数，怎么排布流水线。',
    },
  },
  {
    id: 'grpo-line',
    category: 'post-training',
    name: { en: 'The GRPO lineage', zh: 'GRPO 的谱系' },
    blurb: {
      en: 'From the PPO objective to GRPO, on to DAPO and GSPO, then back to the KL estimator underneath all of them.',
      zh: '从 PPO 的目标函数讲到 GRPO，再到 DAPO 和 GSPO，最后回到支撑它们的那个 KL 估计量。',
    },
  },
  {
    id: 'mantan-cpp',
    category: 'archive',
    name: { en: 'C++ rambles', zh: '漫谈 C++' },
    blurb: { en: 'The parts of C++ that only bite you in production.', zh: 'C++ 里那些只会在生产环境咬你一口的地方。' },
  },
  {
    id: 'constexpr',
    category: 'archive',
    name: { en: 'Compile-time constants to constexpr', zh: '从编译期常量到 constexpr' },
    blurb: { en: 'Four parts, ending at C++17.', zh: '一共四篇，写到 C++17 为止。' },
  },
  {
    id: 'const',
    category: 'archive',
    name: { en: 'The const truth', zh: 'const 真理大讨论' },
    blurb: { en: 'Syntactic const, semantic const, mutable, thread safety.', zh: '语法 const、语义 const、mutable、线程安全。' },
  },
]

/** Titles matching any of these are dropped: competitive-programming practice notes. */
const DROP = [/刷题/, /LeetCode/i, /单调栈/]

/** Deliberately small tag vocabulary. At most two per post; many posts carry none. */
const TAXONOMY = {
  '2026-02-13-vogel-im-kafig': { cat: 'music', langs: ['zh'], tags: ['Sawano'], kind: 'deep-dive' },
  '2025-10-02-ai-infra': { cat: 'systems', langs: ['en'], tags: [], kind: 'essay' },
  '2025-08-08-GRPO_DAPO_GSPO': { cat: 'post-training', langs: ['en', 'zh'], tags: ['GRPO'], series: [['grpo-line', 2]], kind: 'deep-dive' },
  '2025-07-02-KL': { cat: 'post-training', langs: ['en', 'zh'], tags: ['GRPO'], series: [['grpo-line', 3]], kind: 'paper-notes' },
  '2025-04-10-decorators': { cat: 'systems', langs: ['en', 'zh'], tags: ['Python'], kind: 'deep-dive', full: true },
  '2025-03-08-bauklotze': { cat: 'music', langs: ['zh'], tags: ['Sawano'], kind: 'deep-dive' },
  '2025-02-27-dualpipe': { cat: 'systems', langs: ['en', 'zh'], tags: ['Distributed'], series: [['deepseek', 4]], kind: 'deep-dive' },
  '2025-02-11-rlhf': { cat: 'post-training', langs: ['en', 'zh'], tags: ['RLHF'], kind: 'deep-dive' },
  '2025-02-07-grpo': { cat: 'post-training', langs: ['en', 'zh'], tags: ['GRPO', 'RLHF'], series: [['grpo-line', 1], ['deepseek', 3]], kind: 'deep-dive', full: true, featured: true },
  '2025-02-02-mla': { cat: 'systems', langs: ['en', 'zh'], tags: ['KV-Cache'], series: [['deepseek', 2]], kind: 'deep-dive' },
  '2025-01-20-deepseek-r1': { cat: 'post-training', langs: ['en', 'zh'], tags: ['GRPO'], series: [['deepseek', 1]], kind: 'deep-dive' },
  '2025-01-15-moe-load-balancing': { cat: 'systems', langs: ['en', 'zh'], tags: ['MoE'], kind: 'deep-dive' },
  '2024-12-15-unlearning-pitfalls': { cat: 'post-training', langs: ['en', 'zh'], tags: ['Unlearning'], kind: 'deep-dive' },
  '2020-12-31-effective-c++': { cat: 'archive', langs: ['en', 'zh'], tags: ['C++'], kind: 'book-notes' },
  '2020-09-25-Lie-Group': { cat: 'archive', langs: ['zh'], tags: [], kind: 'deep-dive' },
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
    if (item && key) {
      ;(data[key] ||= []).push(item[1].trim())
      continue
    }
    const kv = /^([A-Za-z_][\w-]*):\s*(.*)$/.exec(line)
    if (!kv) continue
    key = kv[1]
    const value = kv[2].trim().replace(/^["'](.*)["']$/, '$1')
    data[key] = value === '' ? [] : value
  }
  return { data, body: m[2] }
}

function excerpt(md, limit = 180) {
  const text = md
    .replace(/^---[\s\S]*?---/, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^#{1,6}\s+.*$/gm, ' ')
    .replace(/[*_`>|]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  return text.length <= limit ? text : `${text.slice(0, limit).replace(/[\s，。、]+\S*$/, '')}…`
}

/**
 * Placeholder read counts. Derived from the slug so the numbers are stable across
 * builds; the real site reads them from the analytics endpoint at runtime.
 */
function placeholderReads(slug, date) {
  let h = 0
  for (const ch of slug) h = (h * 31 + ch.charCodeAt(0)) % 100000
  const ageMonths = Math.max(1, Math.round((Date.parse('2026-08-21') - Date.parse(date)) / 2.63e9))
  return 400 + (h % 900) + ageMonths * (90 + (h % 60))
}

export function loadPosts() {
  const posts = fs
    .readdirSync(path.join(REPO, '_posts'))
    .filter((f) => f.endsWith('.md'))
    .map((file) => {
      const key = file.replace(/\.md$/, '')
      const raw = fs.readFileSync(path.join(REPO, '_posts', file), 'utf8')
      const { data, body } = parseFrontMatter(raw)
      const title = data.title || key
      if (DROP.some((re) => re.test(title))) return null

      const tax = TAXONOMY[key] ?? {}
      const bilingual = data.layout === 'post_lang'
      const langs = tax.langs ?? (bilingual ? ['en', 'zh'] : ['zh'])

      let series = tax.series ?? []
      if (!series.length) {
        for (const [re, id, part] of SERIES_BY_TITLE) if (re.test(title)) series = [[id, part]]
      }
      if (!series.length && /^漫谈C\+\+/.test(title)) series = [['mantan-cpp', null]]

      let summary = excerpt(body)
      let source = body
      if (bilingual) {
        const en = path.join(REPO, '_includes', 'posts', `${key}_en.md`)
        if (fs.existsSync(en)) {
          source = fs.readFileSync(en, 'utf8')
          summary = excerpt(source)
        }
      }
      const eqns = Math.floor((source.match(/\$\$/g) || []).length / 2)

      const slug = key.replace(/^\d{4}-\d{1,2}-\d{1,2}-/, '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
      const date = String(data.date).slice(0, 10)
      return {
        slug,
        key,
        date,
        title,
        subtitle: data.subtitle || '',
        category: tax.cat ?? 'archive',
        kind: tax.kind ?? 'note',
        langs,
        tags: (tax.tags ?? (Array.isArray(data.tags) ? data.tags : [])).slice(0, 2),
        series: series.map(([id, part]) => ({ id, part })),
        excerpt: summary,
        eqns,
        full: Boolean(tax.full),
        featured: Boolean(tax.featured),
        reads: placeholderReads(slug, date),
      }
    })
    .filter(Boolean)

  posts.sort((a, b) => (a.date < b.date ? 1 : -1))
  return posts
}

/** Short posts. Placeholder content: this type does not exist on the site yet. */
export const NOTES = [
  {
    id: 'n12', at: '2026-08-19 23:41', lang: 'en',
    body: 'Rebuilt the site. The old one loaded MathJax from a CDN and typeset every equation twice, once per language, because both translations sat in the same document with one hidden.',
  },
  {
    id: 'n11', at: '2026-08-18 14:07', lang: 'en',
    body: 'Spent the whole morning convinced the $k_3$ estimator was broken. It was fine. The sign convention in my own notes was broken.',
  },
  {
    id: 'n10', at: '2026-08-17 21:12', lang: 'zh',
    body: '把 GSPO 的序列级重要性比值又推了一遍。它和 GRPO 的差别，其实全部落在一个求和符号的位置上。',
  },
  {
    id: 'n09', at: '2026-08-15 10:33', lang: 'en',
    code: { lang: 'python', source: 'def profiling_decorator(fn):\n    @functools.wraps(fn)\n    def wrapper(self, *args, **kwargs):\n        with profiling_context(self, fn.__name__):\n            return fn(self, *args, **kwargs)\n    return wrapper' },
    body: 'The whole trick behind `@profiling_decorator` in trl. Twelve lines, and the only reason anyone can tell where the time goes.',
  },
  {
    id: 'n08', at: '2026-08-14 09:02', lang: 'zh',
    body: '《Vogel im Käfig》里那个小二度，在合唱声部进来之前一共出现了十一次。我数了三遍才敢确定这个数字。',
  },
  {
    id: 'n07', at: '2026-08-12 17:55', lang: 'en',
    body: 'Today I learned that `torch.compile` falls back to eager, silently, if a custom op has a data-dependent output shape. No warning. Two days.',
  },
  {
    id: 'n06', at: '2026-08-11 08:20', lang: 'en',
    body: 'A RoPE table that diverges between the CPU build and the GPU build is the class of bug that eats a week and leaves no trace in the loss curve.',
  },
  {
    id: 'n05', at: '2026-08-09 22:48', lang: 'zh',
    body: '被问到 MoE 的 auxiliary loss 系数该设多大。我的建议是先把 router 的熵画出来看一眼，不要一上来就调那个系数。',
  },
  {
    id: 'n04', at: '2026-08-07 13:14', lang: 'en',
    body: 'The $\\lambda$ in GAE is not a knob you tune once and forget. It is a statement about how much you trust your critic, and that trust changes over training.',
  },
  {
    id: 'n03', at: '2026-08-05 19:30', lang: 'zh',
    body: '重读了一遍自己 2020 年写的 constexpr 系列，发现当时对 C++17 的理解有明显的错误。已经在文末补上更正，原文没有改动。',
  },
  {
    id: 'n02', at: '2026-08-03 11:26', lang: 'en',
    body: 'Three runs from the same step-56 checkpoint. Two diverged at step 62. The third one is the one in the plot.',
  },
  {
    id: 'n01', at: '2026-08-01 16:40', lang: 'zh',
    body: '讲了一次 KL 估计。最有用的一张图是把 $k_1$、$k_2$、$k_3$ 的方差随着真实 KL 的大小画在一起，一眼就能看出该选哪个。',
  },
]

/** Placeholder comment thread, to show the layout a comments widget would occupy. */
export const COMMENTS = [
  {
    who: 'placeholder-reader', at: '3 days ago', reactions: 4,
    body: 'The exam analogy for the critic finally made the baseline click for me. One question: does the same intuition survive when the reward is only available at the end of a long trajectory?',
  },
  {
    who: 'placeholder-reader-2', at: '2 days ago', reactions: 1,
    body: 'Small note on Eq. (4) — worth spelling out that the min is what makes the clip one-sided. I had to stare at it for a while.',
  },
]

export const UI = {
  en: {
    articles: 'Articles', posts: 'Posts', about: 'About', archive: 'Archive', search: 'Search',
    all: 'All', series: 'Series', recent: 'Recent', allArticles: 'All articles',
    article: 'article', articlesN: 'articles', note: 'short post', notesN: 'short posts',
    readingTime: 'min read', words: 'words', equations: 'equations', contents: 'Contents',
    partOf: 'Part {n} of {m}', onlyIn: 'Only in {lang}', copy: 'Copy', copied: 'Copied',
    cite: 'Cite', reads: 'reads', visits: 'visits', comments: 'Comments', reply: 'Reply',
    signIn: 'Sign in with GitHub to comment', noResults: 'Nothing matched that',
    searchPlaceholder: 'Search articles, short posts…', skim: 'Skim', read: 'Read', derive: 'Derive',
    readAs: 'Read as', published: 'Published', older: 'Older', newer: 'Newer', more: 'More',
    langName: { en: 'English', zh: 'Chinese' },
    kind: { 'deep-dive': 'Deep dive', essay: 'Essay', 'paper-notes': 'Paper notes', 'book-notes': 'Book notes', note: 'Note' },
  },
  zh: {
    articles: '文章', posts: '短文', about: '关于', archive: '旧文', search: '搜索',
    all: '全部', series: '系列', recent: '最近', allArticles: '全部文章',
    article: '篇文章', articlesN: '篇文章', note: '条短文', notesN: '条短文',
    readingTime: '分钟', words: '字', equations: '个公式', contents: '目录',
    partOf: '第 {n} 篇 / 共 {m} 篇', onlyIn: '只有{lang}版本', copy: '复制', copied: '已复制',
    cite: '引用', reads: '次阅读', visits: '次访问', comments: '评论', reply: '回复',
    signIn: '用 GitHub 账号登录后可以评论', noResults: '没有找到匹配的内容',
    searchPlaceholder: '搜索文章或者短文…', skim: '略读', read: '通读', derive: '推导',
    readAs: '读法', published: '发布于', older: '更早', newer: '更新', more: '更多',
    langName: { en: '英文', zh: '中文' },
    kind: { 'deep-dive': '深入解析', essay: '随笔', 'paper-notes': '论文笔记', 'book-notes': '读书笔记', note: '短文' },
  },
}

export const SITE = {
  name: { en: 'Yihua Zhang', zh: '张逸骅' },
  handle: 'normaluhr',
  tagline: {
    en: 'Long notes on reinforcement learning, model efficiency, and the occasional piece of music taken apart bar by bar.',
    zh: '关于强化学习和模型效率的长笔记，偶尔也把一首曲子逐小节拆开来讲。',
  },
  links: [
    { label: 'GitHub', href: 'https://github.com/NormalUhr' },
    { label: 'Hugging Face', href: 'https://huggingface.co/NormalUhr' },
    { label: 'Scholar', href: 'https://scholar.google.com' },
    { label: 'X', href: 'https://x.com/zyh2022' },
    { label: '知乎', href: 'https://zhihu.com/people/chi-bo-li-de-xi' },
    { label: 'RSS', href: '#/feed' },
  ],
  since: 2019,
  // Placeholder: the real figure comes from the analytics endpoint at runtime.
  visits: 128407,
}
