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

The postman clock and its town have two checks of their own:

```bash
npm test             # the shared time, the town's calendar and festivals, and what the postman should show
npm run clocks       # after a build: what the clock draws, what the town does, the woodland and the garden
```

`npm run clocks` loads the site in headless Chrome. It compares what the postman clock on
the "not found" page draws with what it should show at 00:00, 00:01, 12:00, 14:04, 14:05,
14:37, 14:59, 15:00, 23:05 and 23:59, while a minute, an hour and midnight pass, and after
the page comes back from the background. It then opens the town in both languages and checks
it against the mail in `dist/<lang>/postman.json`: the postman starts at the given day,
walks to the newest mail when he is clicked, and hands over a paper that carries that day's
articles and short posts, written in the hand for its language. It checks that each Easter
egg opens the town on its day and counts itself once in GoatCounter, and that the letter
leaves the page where the reader had scrolled to. It also checks the printed date, the keys,
the longest trip between two days with mail, that the cats stay on the same roofs while the
postman walks, that the lampposts at a crossing name its two months and move with the
street, reduced motion, and that the town and its paper fit 390 and 320 pixels wide. It then
checks that the clock in the masthead shows the visitor's local time, that pressing it opens
the woodland in the page's language with the ants keeping the same time and counts it once,
that Escape and the close button close it without moving the page, and that phones 320 pixels
wide do not show the clock. It also checks that the woodland's moonlight reaches its
materials. Last, it checks that the years in the footer open the garden in the page's language
and count it once, and that the garden's clock shows the same time as the masthead. Escape must
first end raking and only then close the garden, and a stroke raked just before the garden
closes must still be in the sand when it opens again. The sand dish must show a stone for each
hour of the twelve-hour clock and open the garden with a count of its own. Add `-- --shots` to
also save screenshots to `shots/clocks/`.

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
    eggs/                 the Easter eggs: when each one appears, on every page
    clocks/               the postman clock, his town, and his visits to the pages
    woodland/             the woodland clock, which the clock in the masthead opens
    garden/               the karesansui garden, which the sand dish and the years in the footer open
    layouts/ components/ pages/ styles/
  scripts/
    migrate.mjs           the Jekyll import, re-runnable
    audit.mjs             the page check described above
    clocks.test.mjs       what the clock should show, and the town's calendar
    clocks.mjs            what the clock, the town, the woodland and the garden do, in the browser
    serve.mjs             serves dist/ for the two browser checks
    migration-report.txt  what the import decided, per post
```

## Easter eggs

No page links to the postman's town. A visitor finds it in one of these ways:

- **He walks past.** At the turn of each hour on the visitor's clock, and at a random moment
  on a share of page views, the postman walks along the bottom of the window. Pressing him
  opens the town at today.
- **A date.** Pressing any date in a list or under an article title opens the town on that
  day. Dates show a pointer when the mouse is over them.
- **The page that does not exist.** The "not found" page shows the postman clock described
  below. Pressing it opens the town at today.
- **A letter.** A reader who reaches the end of an article, after spending a share of its
  reading time on it, gets a letter at the right edge of the window. Pressing it opens the
  town on the day the article came out.

No page links to the woodland clock either. The masthead of every page shows the visitor's
local time, such as 01:30, just before the search button, with nothing to say what it is.
Pressing it opens the woodland over the page, where ants lay out the same time in stones.
Phones narrower than 360 pixels do not show the time, because it would push the masthead's
controls onto a row of their own.

The karesansui garden is in the footer. The footer of every page ends with the years the site
has been kept, such as 2019–, and a small dish of raked sand. The dish keeps the hour the way
the garden does, with a stone for each hour of the twelve-hour clock and a groove in the middle
that grows one turn an hour. Pressing the dish or the years opens the garden over the page.

Each egg counts itself in GoatCounter when a visitor opens it, as an event named
`egg-postman-walk`, `egg-postman-date`, `egg-postman-lost`, `egg-postman-letter`,
`egg-woodland`, `egg-garden-dish` or `egg-garden-years` (`src/eggs/found.ts`). The GoatCounter dashboard lists these events with the
pages. GoatCounter counts each event at most once per visit, and the events also add to the
site total in the footer. A preview such as `clock-open` counts nothing, and GoatCounter's
script counts nothing on a local address.

`src/eggs/index.ts` runs on every page. It only waits for these moments, using the triggers
in `src/eggs/triggers.ts`, and loads an egg's code when its moment comes. How often the
walk happens by chance, and how much reading time the letter needs, are set in
`src/eggs/luck.ts`. A new egg adds its moment to `index.ts`, reusing a trigger where one fits.

## The clock

The Time Postman is a clock. Its town has a street for each hour and a house for each
minute, and the postman is at the house whose street is the hour and whose number is the
minute. It stands on the "not found" page.

The clock reads one shared time (`src/clocks/time.ts`) and draws what
`src/clocks/encode.ts` says it should show, so any moment can be drawn directly. It is
drawn only while the page is visible. Its folder, `src/clocks/scenes/postman/`, has the
drawing (`index.ts` and `town.ts`), the styles (`style.css`), its name (`copy.ts`) and the
rest of its words (`phrases.ts`), all in both languages. `src/clocks/host.ts` puts the clock
on the page and opens the town, and `src/clocks/visits.ts` draws the walk and the letter.

## The town

The Easter eggs open the same town at the scale of days, with no words around it.
Each house is a day and each street is a month. A street ends at a crossing, and the next
month's street starts on the other side. A lamppost stands on each corner of a crossing,
and its sign names the street on that side by its month and year. The date of the house the
postman is at is printed in the bottom right corner, the way an old film camera printed the
date on a photo. Days after today are drawn as blueprints.

The mailbox of a day with an article or a short post has its red flag up. At such a day
the postman takes the mail out, and clicking him opens that day's newspaper, The Time
Post, with the article's title, opening lines and a link, or the short post in full.
Clicking him with empty hands sends him to the nearest earlier day with mail. A reader can
also click a house, drag the street, scroll, or use the keys: the arrows walk a day, Shift
with an arrow walks to the next day with mail, Page Up and Page Down walk a month, and Home
walks back to today. A trip of more than 45 houses is shown as a time lapse.

The paper is written by hand on a sheet with a grain, a fold and darker edges. English is
written in Grape Nuts. Chinese is written in Ma Shan Zheng, a brush pen, with Caveat Brush
for the English words and figures in it. The fonts come from Google Fonts like the site's
own, but only once the postman takes out a day's mail, and for Chinese Google sends only
the characters on that day's paper. Without Google Fonts the paper falls back to the
handwriting fonts the system has, such as Bradley Hand and Kaiti.

Each day keeps the same season, weather and festivals every time it is drawn. The date is
the seed for the weather, and the Spring Festival and the Mid-Autumn Festival come from
the browser's Chinese calendar. A dog follows the postman, pigeons sit on the sign where
each street begins, a cat sleeps on the roof of about one house in ten (always the same
houses), someone reads the paper on a bench on days with an article, and now and then
someone rides past on a bicycle.

The mail is built into `/<lang>/postman.json` by `src/pages/[lang]/postman.json.ts` and is
fetched only when the town is first opened. The town's code is in `src/clocks/museum/`:
the calendar (`days.ts`), the mail and its lookup (`mail.ts`), the town and how it moves
(`index.ts`), the season and weather (`sky.ts`), the townsfolk (`cast.ts`), the date in the
corner (`datestamp.ts`), the newspaper
(`paper.ts`), the styles (`style.css`) and the words in both languages (`copy.ts`).

## The woodland clock

The woodland clock is a 3D woodland where worker ants carry stones to lay out the visitor's
local time. A visitor can turn the view, zoom, and drop a stone with a right-click or a long
press. Day and night, the seasons and the weather follow the visitor's clock. At night the
moonlight falls through an opening in the trees onto the clock and the nest. It needs WebGL 2
and WebAssembly, and without them the clock in the masthead stays a plain clock.

It is a page of its own, `/<lang>/woodland/` (`src/pages/[lang]/woodland.astro`), which no page
links to and which asks search engines to leave it out. The woodland takes over its whole window
and has no way to stop itself, so `src/eggs/frame.ts` opens the page in a frame that fills the
window, with a close button in the top right corner. Escape inside the woodland asks the site to
close it. Closing removes the frame, which stops the woodland and frees its graphics memory.

`src/woodland/app/` is the woodland's own program, taken from the source of its single-file
version 5 and built with the versions it was written for, Three.js 0.180.0 and Rapier 0.21.0.
A later single-file build brought the moonlight (`app/moonlight.js`) and a stronger light, but
no source. Those changes were written back into this source, and the source with them builds
into the same program as that file. This copy differs from that source in four ways:

- The six Poly Haven textures are image files in `src/woodland/textures/`, so browsers cache
  them. The source has them as text inside the script.
- Every word the woodland shows is in `app/words.js`, in Chinese as written and in English.
  The language of the page picks one.
- In the site's frame, the woodland leaves the top right corner to the close button, and Escape
  asks the site to close it.
- The markup and styles of its `shell.html` are now in `woodland.astro`.

Its `README.md`, in Chinese, explains how the program works. Its tests, its build script and
the tool that bakes its reflection data stay with the original project and are not copied here.

A visitor who opens the woodland downloads about 5.6 MB the first time, most of it the physics
engine and the textures. The licences of Three.js (MIT) and Rapier (Apache 2.0) and the sources
of the textures (CC0) are in `src/woodland/licenses/` and `src/woodland/textures/`, and the page
carries them in its `third-party-notices` element.

## The karesansui garden

The karesansui garden is a 3D dry garden beside a small house. A spiral raked into the sand
grows by one turn each hour, and stones set around it at the positions of a clock face count
the hours of a twelve-hour clock. A family takes turns coming out of the house to keep the
spiral and to smooth whatever a visitor or a passing animal has marked. A visitor can turn
the view, zoom, rake the sand with a right-click drag or the Rake button, preview the light,
the weather and the seasons, and turn on sound. It needs WebGL 2, and without it the dish and
the years in the footer open nothing.

Like the woodland, it is a page of its own, `/<lang>/garden/` (`src/pages/[lang]/garden.astro`),
that no page links to and that `src/eggs/frame.ts` opens in a frame over the page. The garden
saves itself in the visitor's browser, so it carries on from where it was when the visitor
comes back. Removing the frame would cut short a save that starts at the close, so the garden
asks the site to hold its frame, and the site removes the frame once the garden has saved.
That takes about a tenth of a second.

`src/garden/app/` is the garden's own program, taken from the source of its version 1.6.0 and
built with Three.js 0.180.0, the version it was written for. This copy differs from that source
in three ways:

- Every word the garden shows is in `app/words.js`, in Chinese as written and in English.
  The language of the page picks one.
- In the site's frame, the garden leaves the top right corner to the close button. Escape asks
  the site to close it when there is no raking or open panel for Escape to end, and the garden
  saves before its frame goes.
- The markup and styles of its `shell.html` are now in `garden.astro`, and the page loads the
  font for its Chinese words as a file instead of carrying it inside.

Its `README.md`, in Chinese, explains how the program works. Its tests and its build script stay
with the original project and are not copied here.

A visitor who opens the garden downloads about 1 MB the first time. Half of it is Three.js,
which the woodland uses too, so a visitor who has opened one of them already has it. The
licences of Three.js (MIT) and of the font, a subset of Noto Sans SC (OFL), are in
`src/garden/licenses/` and `src/garden/assets/`, and the page carries them in its
`third-party-notices` element.

## Previews

To see any moment, add these to the address of any page. The clock itself is on `/404.html`.

| | |
| --- | --- |
| `clock-time=14:37` | the clock reads 14:37 today and runs on from there; also `14:37:50` or `2026-11-01T01:59:50` |
| `clock-pause` | stop the time |
| `clock-zone=Asia/Shanghai` | read the time in another zone |
| `clock-open` | open the town straight away |
| `eggs=now` | the postman walks past after a second, and the letter comes as soon as the end of an article is reached |

The town takes today's date from the same time, so `/en/?clock-time=2026-02-17T10:00&clock-open`
opens it on the Spring Festival of 2026, with every later day drawn as a blueprint.

The woodland clock opens on its own at `/en/woodland/` and `/zh/woodland/`, and the garden at
`/en/garden/` and `/zh/garden/`. Adding `?fresh` to the garden's address opens a new garden and
leaves the saved one as it was.

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
origin: zh          # the language it was actually written in
---

Short thought. Inline `$k_3$` and `code` both work.
```

Both files carry the same `origin`. The one whose folder matches it is the piece as
written; the other is a translation, and the page says so underneath it — *Translated
from Chinese by AI* / *本文由 AI 译自英文*. Nothing machine-written is shown without
that line.

The build fails, by name, if a translation is missing or if the two files disagree
about which language is the original.

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
