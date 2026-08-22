# Like counts

A ~100-line Cloudflare Worker backed by D1. It exists because a static site cannot
store a number, and everything else that can is either a login wall or somebody
else's server.

```
GET  /v1/<slug>   -> { key, count }
POST /v1/<slug>   -> { key, count, counted }
```

`<slug>` is the article slug, so `/en/p/grpo` and `/zh/p/grpo` share one count.

**No account, no cookie, no cross-site identifier.** The only per-reader value stored
is a salted SHA-256 of the request IP, truncated to 16 hex characters, used to make a
like count once per article per day. The address itself is never written down.

## Setting it up

Roughly five minutes. Everything below is on Cloudflare's free tier: D1 allows 100,000
row writes a day, which is far past what a personal blog will ever use.

```bash
cd workers/likes
npm install
npx wrangler login

# 1. create the database, then paste the printed database_id into wrangler.jsonc
npx wrangler d1 create blog-likes

# 2. create the tables
npm run schema

# 3. set the salt to any long random string
npx wrangler secret put VOTE_SALT

# 4. deploy
npm run deploy
```

Deploy prints a URL like `https://blog-likes.<your-subdomain>.workers.dev`. Put it in
`site/src/site.config.ts`:

```ts
likes: 'https://blog-likes.your-subdomain.workers.dev',
```

Two like buttons then appear on every article, one under the title and one at the foot,
and they stay in sync. Leave `likes` as `null` and no button is rendered at all.

## Trying it without deploying

```bash
# terminal 1
cd workers/likes
npx wrangler d1 execute blog-likes --local --file=./schema.sql
echo 'VOTE_SALT=local-dev-salt' > .dev.vars
npx wrangler dev --port 8799 --local

# terminal 2: point the site at it, then run the site
#   likes: 'http://localhost:8799',
cd site && npm run dev
```

`ALLOWED_ORIGINS` in `wrangler.jsonc` already includes `http://localhost:4321`.

## What happens when it is unreachable

The count shows `—` and a click rolls itself back: the number returns to what it was
and the remembered "you liked this" flag is cleared, so the page never shows a total
the server did not record.

## Notes

- `ALLOWED_ORIGINS` is a CORS allowlist. Requests from anywhere else get a 403.
- Keys are validated against `^[a-z0-9][a-z0-9-]{0,63}$`, so the endpoint cannot be
  used to create arbitrary rows.
- The increment is a single `INSERT … ON CONFLICT … RETURNING`, so simultaneous clicks
  cannot lose each other.
- Repeat clicks are held off in the browser via `localStorage` and on the server via
  the `voters` table. Neither is proof against someone determined to inflate a number,
  which is the right trade for a like button.
- After changing bindings in `wrangler.jsonc`, run `npm run types` to regenerate `Env`.
  Secrets are declared by hand in `src/env.d.ts`, since wrangler cannot see them.
