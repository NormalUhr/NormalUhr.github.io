/**
 * Anonymous like counts for the blog.
 *
 *   GET  /v1/<key>   -> { key, count }
 *   POST /v1/<key>   -> { key, count, counted }   increments, once per reader per day
 *
 * <key> is an article slug, so the English and Chinese versions of a piece share one
 * count. No account, no cookie, no cross-site identifier: the only thing stored per
 * reader is a salted hash of their IP, truncated, and only for deduplication.
 *
 * Run `wrangler types` after changing bindings to regenerate Env.
 */

const KEY = /^[a-z0-9][a-z0-9-]{0,63}$/

interface Row {
  count: number
}

function corsHeaders(request: Request, env: Env): Record<string, string> {
  const origin = request.headers.get('Origin') ?? ''
  const allowed = env.ALLOWED_ORIGINS.split(',').map((o) => o.trim()).filter(Boolean)
  const headers: Record<string, string> = {
    Vary: 'Origin',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
  }
  if (allowed.includes(origin)) headers['Access-Control-Allow-Origin'] = origin
  return headers
}

function json(body: unknown, status: number, extra: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...extra },
  })
}

/** Salted, truncated hash of the caller's address. Never stores the address itself. */
async function readerId(request: Request, salt: string): Promise<string> {
  const ip = request.headers.get('CF-Connecting-IP') ?? 'unknown'
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`${salt}:${ip}`))
  return [...new Uint8Array(digest)].slice(0, 8).map((b) => b.toString(16).padStart(2, '0')).join('')
}

async function currentCount(env: Env, key: string): Promise<number> {
  const row = await env.DB.prepare('SELECT count FROM likes WHERE key = ?1').bind(key).first<Row>()
  return row?.count ?? 0
}

export default {
  async fetch(request, env): Promise<Response> {
    const cors = corsHeaders(request, env)

    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors })
    if (!cors['Access-Control-Allow-Origin']) {
      return json({ error: 'origin not allowed' }, 403, cors)
    }

    const url = new URL(request.url)
    const match = /^\/v1\/([^/]+)\/?$/.exec(url.pathname)
    if (!match) return json({ error: 'not found' }, 404, cors)

    const key = decodeURIComponent(match[1])
    if (!KEY.test(key)) return json({ error: 'bad key' }, 400, cors)

    try {
      if (request.method === 'GET') {
        return json({ key, count: await currentCount(env, key) }, 200, cors)
      }

      if (request.method === 'POST') {
        const reader = await readerId(request, env.VOTE_SALT)
        const day = new Date().toISOString().slice(0, 10)

        // One like per reader per article per day. INSERT OR IGNORE reports 0 changes
        // when the row already exists, which is how a repeat click is detected.
        const claim = await env.DB
          .prepare('INSERT OR IGNORE INTO voters (key, reader, day) VALUES (?1, ?2, ?3)')
          .bind(key, reader, day)
          .run()

        if (claim.meta.changes === 0) {
          return json({ key, count: await currentCount(env, key), counted: false }, 200, cors)
        }

        // Atomic increment, so simultaneous clicks cannot lose each other.
        const row = await env.DB
          .prepare(
            'INSERT INTO likes (key, count) VALUES (?1, 1) ' +
              'ON CONFLICT(key) DO UPDATE SET count = count + 1 RETURNING count',
          )
          .bind(key)
          .first<Row>()

        return json({ key, count: row?.count ?? 1, counted: true }, 200, cors)
      }

      return json({ error: 'method not allowed' }, 405, cors)
    } catch (error) {
      console.error(JSON.stringify({ msg: 'likes request failed', key, method: request.method, error: String(error) }))
      return json({ error: 'unavailable' }, 500, cors)
    }
  },
} satisfies ExportedHandler<Env>
