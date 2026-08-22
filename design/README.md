# Blog redesign — working directory

This lives in a **separate git worktree on the `redesign` branch**, so the live site's
checkout (`../website-blog`, branch `master`) is never modified. Nothing here is
committed yet; `master` has no idea this exists.

```
design/
  src/
    themes/atlas.html      the design itself (CSS + view code, one file)
    content.mjs            taxonomy, short posts, UI strings, placeholder counters
    render.mjs             markdown -> HTML (remark-math + KaTeX + Shiki, all at build time)
    build.mjs              injects content + KaTeX CSS into the template
    serve.mjs              dev server: rebuild on save, auto-reload the open tab
    audit.mjs              walks every route in both languages, fails on JS errors
    shot.mjs              headless screenshots at any route/theme/viewport
    themes-archived/       directions A and C from round one, kept for reference
  demos/atlas.html         the built, self-contained file
  shots/                   screenshots
```

## Debug loop

```bash
cd design/src
npm run dev        # http://localhost:4321/atlas.html — rebuilds on save, tab reloads itself
```

Edit `themes/atlas.html` or `content.mjs`; the browser refreshes in about 200 ms.
To stop it: `pkill -f serve.mjs`.

```bash
npm run check      # build, then walk all 19 routes x 2 languages for JS errors,
                   # unrendered template expressions, and horizontal overflow

npm run shot -- "atlas.html|#/p/grpo|post|dark|1440x900|at:.tally"
                   # file | route | name | scheme | WxH | optional action
                   # actions: skim, derive, zh, at:<css-selector>
```

`npm run check` is the regression gate: it catches the class of bug a screenshot
misses, such as a view that throws only in Chinese or only on a post with no math.

## What is real and what is placeholder

| Real | Placeholder |
| --- | --- |
| All 33 posts, from `_posts/` front matter | The 12 short posts |
| Titles, subtitles, dates, languages, excerpts | Read counts and the site visit total |
| Two articles rendered in full from `_includes/posts/` | The two comment entries |
| Equation counts, counted from `$$` in the source | |

## Not wired up yet

- **Read counts / visits.** The numbers render but come from `content.mjs`. On the real
  site they would be fetched at runtime — see the notes in the handover discussion.
- **Comments.** The block is a layout preview of giscus. It needs a repo with
  Discussions enabled and a `<script>` tag, which the artifact sandbox blocks, so it
  cannot run inside the published preview.
