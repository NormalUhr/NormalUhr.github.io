import { defineCollection, z } from 'astro:content'
import { glob } from 'astro/loaders'

const CATEGORY = z.enum(['post-training', 'systems', 'music', 'archive'])
const SERIES = z.enum(['deepseek', 'grpo-line', 'mantan-cpp', 'constexpr', 'const'])

/**
 * One entry per (language, slug). The id is "<lang>/<slug>", so a bilingual article is
 * two entries that share a slug. The schema is what stops a typo in a category or a
 * series name from silently producing an empty page.
 */
const articles = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/articles' }),
  schema: z.object({
    title: z.string(),
    /** The name in the other language, required when the body exists in only one. */
    titleAlt: z.string().optional(),
    subtitle: z.string().optional(),
    date: z.coerce.date(),
    category: CATEGORY,
    series: SERIES.optional(),
    part: z.number().nullable().optional(),
    tags: z.array(z.string()).max(2).default([]),
    draft: z.boolean().default(false),
  }),
})

/**
 * Short posts, as "<lang>/<id>.md". Both languages are required for every id; the
 * build fails otherwise, so a half-translated post cannot reach the site.
 */
const notes = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/notes' }),
  schema: z.object({
    date: z.coerce.date(),
    draft: z.boolean().default(false),
  }),
})

/** Hand-written pages, as "<lang>/<name>.md". Body may be empty. */
const pages = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/pages' }),
  schema: z.object({}).passthrough(),
})

export const collections = { articles, notes, pages }
