# The new site

Astro, static output, no server. Lives on the `redesign` branch in a git worktree, so
the live site's checkout (`../../website-blog`, branch `master`) is untouched.

## Preview it

```bash
cd site
npm install          # first time only
npm run dev          # http://localhost:4321 — hot reload on save
```

Or preview exactly what would be deployed:

```bash
npm run build
npm run preview      # serves dist/ on http://localhost:4321
```

Both open on the English home page. Useful routes:

| | |
| --- | --- |
| `/en/` `/zh/` | home: article timeline plus short posts |
| `/en/p/grpo` | a long article with 16 equations, the equation index, and the table of contents |
| `/en/p/decorators` | 41 highlighted code blocks |
| `/zh/p/vogel-im-kafig` | the music post, 39 audio and video embeds |
| `/en/p/vogel-im-kafig` | the language fallback: English chrome, Chinese article, a notice saying so |
| `/en/articles` | everything, with topic filters |
| `/zh/archive` | the 2019–2020 material |
| `/en/search` | search over every article and short post |
| `/2025/02/07/grpo` | an old Jekyll permalink, redirecting to the new URL |

Press the `◐` button top right for light mode. Dark is the default and the choice is remembered.

## Check it

```bash
npm run verify       # build, then load all 147 pages in headless Chrome
```

It fails the run on a JavaScript error, a dead internal link, a media file that does
not exist, a page that scrolls sideways, or a page with no title. It also lists any
media loaded from another host. This is the gate to run before pushing anything.

## Layout

```
site/
  src/
    site.config.ts        name, tagline, links, and the two optional services
    lib/taxonomy.ts       topics, series, and every string in both languages
    lib/articles.ts       queries: language fallback, series order, counts
    lib/markdown.mjs      build-time passes: Shiki, equation numbers, table of contents
    content.config.ts     the schema that validates every article's front matter
    content/articles/     en/<slug>.md and zh/<slug>.md
    content/notes/        en/ and zh/, both required per post (currently empty)
    content/pages/        About, one file per language (currently empty)
    layouts/ components/ pages/ styles/
  scripts/
    migrate.mjs           the Jekyll import, re-runnable
    audit.mjs             the check described above
    migration-report.txt  what the import decided, per post
```

## Copy you need to write

The template ships with no prose of its own. Three things are deliberately blank:

| What | Where | If left blank |
| --- | --- | --- |
| The line under your name | `tagline` in `src/site.config.ts` | nothing is shown |
| The About page | `src/content/pages/{en,zh}/about.md` | About is dropped from the nav |
| Topic and series descriptions | not present | topic pages show the name and the list |
| Favicon | `favicon` in `src/site.config.ts` | the page declares it has no icon, so no 404 |

Topic and series pages carry no description at all now. If you want one, add a `blurb`
field to `CATEGORIES` / `SERIES` in `src/lib/taxonomy.ts` and render it.

## Writing

**A new article.** One file per language under `src/content/articles/<lang>/`. A
bilingual article is two files sharing a filename.

```markdown
---
title: "The name, in this file's language"
titleAlt: "The name in the other language"   # required only if the body is single-language
subtitle: "Optional"
date: 2026-09-01
category: post-training      # post-training | systems | music | archive
kind: deep-dive              # deep-dive | essay | paper-notes | book-notes | note
series: grpo-line            # optional
part: 4                      # optional
tags: ["GRPO"]               # at most two
---

Body. `$…$` and `$$…$$` render at build time. Fenced code highlights at build time.
Media paths are absolute from the site root: /img/... and /videos/...
```

A wrong category or series name fails the build rather than producing an empty page.
Every article's name is shown in both languages, so a single-language article needs
`titleAlt`; the build fails without it.

**A short post.** Two files, one per language, sharing a filename:

```
src/content/notes/en/2026-09-01-torch-compile.md
src/content/notes/zh/2026-09-01-torch-compile.md
```

```markdown
---
date: 2026-09-01
---

Short thought. Inline `$k_3$` and `code` both work.
```

Both languages are required. A short post with only one fails the build.

## Turning on the optional services

All three are off. While they are off the page renders no read count, no like button
and no comment section, so nothing on the site is ever a placeholder number.

**Read counts and visit totals — GoatCounter.** Sign up at goatcounter.com, then put
your site code in `src/site.config.ts`:

```ts
goatcounter: 'yourcode',   // the "yourcode" in yourcode.goatcounter.com
```

That adds the page-view script, a read count under each article, and the site total in
the footer, all read from GoatCounter's public counter endpoint. No cookies, no
personal data, free for a personal site.

**Like button — the Worker in `workers/likes/`.** One tap, no account. Two buttons per
article, one under the title and one at the foot, sharing a count between the article's
languages. Deploy it (about five minutes, free tier, see `workers/likes/README.md`) and
set its URL:

```ts
likes: 'https://blog-likes.your-subdomain.workers.dev',
```

**Comments — giscus.** Enable Discussions on the repository, create a category for
comments, run through giscus.app to get the four ids, then:

```ts
giscus: {
  repo: 'NormalUhr/NormalUhr.github.io',
  repoId: '…',
  category: 'Comments',
  categoryId: '…',
},
```

Threads live in that Discussions category. Readers sign in with GitHub. Nothing to host.

## Deploying

`.github/workflows/deploy.yml` builds `site/` and publishes to Pages. It is inert
until this branch is merged **and** Settings → Pages → Build and deployment is switched
to "GitHub Actions". Nothing deploys on its own.

Note the payload: `dist/` is about 477 MB, almost all of it the music post's video
clips (one is 96 MB). It is within the 1 GB Pages limit but the upload is slow, and it
is worth re-encoding those clips before going live.
