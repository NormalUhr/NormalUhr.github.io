// Dev loop: rebuild on any change under src/, serve demos/ over HTTP, and expose a
// tiny long-poll so an open browser tab reloads itself when the build finishes.
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { build } from './build.mjs'

const HERE = path.dirname(new URL(import.meta.url).pathname)
const OUT = path.resolve(HERE, '..', 'demos')
const PORT = Number(process.env.PORT || 4321)

let version = 0
let waiting = []
let building = false
let queued = false

const RELOAD_SNIPPET = `
<script>
// dev-only: hold a request open until the build version changes, then reload
(function poll(v){
  fetch('/__wait?v=' + v).then(r => r.json()).then(d => {
    if (d.version !== v) location.reload(); else poll(v);
  }).catch(() => setTimeout(() => poll(v), 1500));
})(__VERSION__);
</script>`

async function rebuild(reason) {
  if (building) { queued = true; return }
  building = true
  const t = Date.now()
  try {
    await build({ quiet: true })
    version += 1
    console.log(`  rebuilt in ${Date.now() - t}ms  (${reason}) -> v${version}`)
    for (const res of waiting.splice(0)) res.end(JSON.stringify({ version }))
  } catch (err) {
    console.error(`  build failed (${reason}):`, err.message)
  } finally {
    building = false
    if (queued) { queued = false; await rebuild('queued change') }
  }
}

const TYPES = { '.html': 'text/html; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml', '.json': 'application/json' }

http
  .createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost')
    if (url.pathname === '/__wait') {
      const seen = Number(url.searchParams.get('v') || 0)
      res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' })
      if (seen !== version) return res.end(JSON.stringify({ version }))
      waiting.push(res)
      req.on('close', () => { waiting = waiting.filter((r) => r !== res) })
      return
    }

    let name = url.pathname === '/' ? '/atlas.html' : url.pathname
    const file = path.join(OUT, path.normalize(name).replace(/^(\.\.[/\\])+/, ''))
    if (!file.startsWith(OUT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404, { 'content-type': 'text/plain' })
      return res.end('not found\n')
    }
    const ext = path.extname(file)
    res.writeHead(200, { 'content-type': TYPES[ext] || 'application/octet-stream', 'cache-control': 'no-store' })
    if (ext !== '.html') return res.end(fs.readFileSync(file))
    res.end(fs.readFileSync(file, 'utf8') + RELOAD_SNIPPET.replace('__VERSION__', String(version)))
  })
  .listen(PORT, () => {
    console.log(`\n  design preview   http://localhost:${PORT}/atlas.html`)
    console.log(`  watching         ${path.relative(process.cwd(), HERE)}\n`)
  })

await rebuild('startup')

// Debounced watch over the sources that feed the build.
let timer = null
for (const dir of [HERE, path.join(HERE, 'themes')]) {
  fs.watch(dir, (_event, filename) => {
    if (!filename || !/\.(html|mjs)$/.test(filename)) return
    clearTimeout(timer)
    timer = setTimeout(() => rebuild(filename), 120)
  })
}
