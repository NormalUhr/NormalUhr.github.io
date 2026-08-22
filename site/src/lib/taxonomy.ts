import type { Lang } from '../site.config'

export type CategoryId = 'post-training' | 'systems' | 'music' | 'archive'
export type SeriesId = 'deepseek' | 'grpo-line' | 'mantan-cpp' | 'constexpr' | 'const'

type Bi = Record<Lang, string>

export interface Category {
  id: CategoryId
  accent: string
  name: Bi
}

/** Three living topics. Archive is a section, not a topic, and is listed separately. */
export const CATEGORIES: Category[] = [
  {
    id: 'post-training',
    accent: '#5ee39b',
    name: { en: 'Post-training', zh: '后训练' },
  },
  {
    id: 'systems',
    accent: '#6cb6ff',
    name: { en: 'Systems', zh: '系统' },
  },
  {
    id: 'music',
    accent: '#ffab5e',
    name: { en: 'Music', zh: '音乐' },
  },
]

export const ARCHIVE: Category = {
  id: 'archive',
  accent: '#8b97a6',
  name: { en: 'Archive', zh: '旧文' },
}

export const ALL_CATEGORIES = [...CATEGORIES, ARCHIVE]
export const category = (id: string) => ALL_CATEGORIES.find((c) => c.id === id) ?? ARCHIVE

export interface Series {
  id: SeriesId
  category: CategoryId
  name: Bi
}

export const SERIES: Series[] = [
  {
    id: 'deepseek',
    category: 'post-training',
    name: { en: 'Anatomy of DeepSeek', zh: '解剖 DeepSeek' },
  },
  {
    id: 'grpo-line',
    category: 'post-training',
    name: { en: 'The GRPO lineage', zh: 'GRPO 的谱系' },
  },
  {
    id: 'mantan-cpp',
    category: 'archive',
    name: { en: 'C++ rambles', zh: '漫谈 C++' },
  },
  {
    id: 'constexpr',
    category: 'archive',
    name: { en: 'Compile-time constants to constexpr', zh: '从编译期常量到 constexpr' },
  },
  {
    id: 'const',
    category: 'archive',
    name: { en: 'The const truth', zh: 'const 真理大讨论' },
  },
]

export const series = (id: string) => SERIES.find((s) => s.id === id)

export const UI = {
  en: {
    articles: 'Articles', posts: 'Posts', about: 'About', archive: 'Archive', search: 'Search',
    all: 'All', series: 'Series', readingOrder: 'Reading order', contents: 'Contents',
    article: 'article', articles_n: 'articles', post: 'short post', posts_n: 'short posts',
    minRead: 'min read', words: 'words', equations: 'equations', published: 'Published',
    languages: 'languages', partOf: 'Part {n} of {m}', onlyIn: 'Only in {lang}.',
    copy: 'Copy', copied: 'Copied', cite: 'Cite', reads: 'reads', visits: 'visits',
    like: 'Like this article', liked: 'You liked this article', likeOne: 'like', likes: 'likes',
    translated: 'Translated from {lang} by AI',
    readFull: 'Read in full',
    comments: 'Comments', older: 'Older', newer: 'Newer', nothing: 'Nothing matched that',
    searchPlaceholder: 'Search articles and short posts…',
    noPosts: 'Nothing here yet.',
    langName: { en: 'English', zh: 'Chinese' },
    skipToContent: 'Skip to content',
    notFound: 'That page is not here',
  },
  zh: {
    articles: '文章', posts: '随记', about: '关于', archive: '旧文', search: '搜索',
    all: '全部', series: '系列', readingOrder: '阅读顺序', contents: '目录',
    article: '篇文章', articles_n: '篇文章', post: '条随记', posts_n: '条随记',
    minRead: '分钟', words: '字', equations: '个公式', published: '发布于',
    languages: '语言', partOf: '第 {n} 篇 / 共 {m} 篇', onlyIn: '只有{lang}版本。',
    copy: '复制', copied: '已复制', cite: '引用', reads: '次阅读', visits: '次访问',
    like: '喜欢这篇文章', liked: '你已经喜欢过这篇文章', likeOne: '次喜欢', likes: '次喜欢',
    translated: '本文由 AI 译自{lang}',
    readFull: '阅读全文',
    comments: '评论', older: '更早', newer: '更新', nothing: '没有找到匹配的内容',
    searchPlaceholder: '搜索文章和随记…',
    noPosts: '还没有内容。',
    langName: { en: '英文', zh: '中文' },
    skipToContent: '跳到正文',
    notFound: '这个页面不在这里',
  },
} as const

export const t = (lang: Lang) => UI[lang]
export const fill = (s: string, vars: Record<string, string | number>) =>
  Object.entries(vars).reduce((acc, [k, v]) => acc.replace(`{${k}}`, String(v)), s)
