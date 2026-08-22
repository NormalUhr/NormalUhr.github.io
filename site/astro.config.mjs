import { defineConfig } from 'astro/config'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import { rehypeDropLeadingH1, rehypeDropCap, rehypeShiki, rehypeNumberEquations, rehypeCollectToc } from './src/lib/markdown.mjs'
import legacyRedirects from './src/generated/legacy-redirects.mjs'

export default defineConfig({
  site: 'https://normaluhr.github.io',
  trailingSlash: 'ignore',
  i18n: {
    locales: ['en', 'zh'],
    defaultLocale: 'en',
    routing: { prefixDefaultLocale: true, redirectToDefaultLocale: true },
  },
  markdown: {
    // Astro's own highlighter would try to treat remark-math's `language-math`
    // blocks as code and eat the equations, so we run our own pass instead.
    syntaxHighlight: false,
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeDropLeadingH1,
      rehypeDropCap,
      rehypeShiki,
      [rehypeKatex, { strict: 'ignore', throwOnError: false, trust: true }],
      rehypeNumberEquations,
      rehypeSlug,
      rehypeCollectToc,
    ],
    gfm: true,
    smartypants: false,
  },
  // Static build, so "/" is a generated meta-refresh page rather than a server redirect.
  redirects: { '/': '/en/', ...legacyRedirects },
  build: { format: 'directory' },
  devToolbar: { enabled: false },
})
