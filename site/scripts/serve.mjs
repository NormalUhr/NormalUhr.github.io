// Serves a built site the way GitHub Pages does, for the checks that load it in a browser.
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript',
  '.json': 'application/json', '.xml': 'application/xml', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf',
  '.mp4': 'video/mp4', '.mp3': 'audio/mpeg', '.ico': 'image/x-icon',
}

/** The file a request path maps to inside `dist`, or null. */
export function resolver(dist) {
  return (pathname) => {
    const clean = decodeURIComponent(pathname.split('?')[0].split('#')[0])
    const candidates = [
      path.join(dist, clean),
      path.join(dist, clean, 'index.html'),
      path.join(dist, `${clean}.html`),
    ]
    for (const c of candidates) {
      if (!path.resolve(c).startsWith(dist)) continue
      if (fs.existsSync(c) && fs.statSync(c).isFile()) return c
    }
    return null
  }
}

export async function serve(dist, port) {
  const resolveFile = resolver(dist)
  const server = http.createServer((req, res) => {
    const file = resolveFile(new URL(req.url, 'http://x').pathname)
    if (!file) {
      const lost = resolveFile('/404.html')
      res.writeHead(404, { 'content-type': lost ? TYPES['.html'] : 'text/plain' })
      return lost ? fs.createReadStream(lost).pipe(res) : res.end('404')
    }
    const type = TYPES[path.extname(file)] ?? 'application/octet-stream'
    const size = fs.statSync(file).size
    // <video> and <audio> ask for byte ranges, so a plain 200 is not enough.
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? '')
    if (range) {
      const start = range[1] ? Number(range[1]) : 0
      const end = range[2] ? Number(range[2]) : size - 1
      res.writeHead(206, {
        'content-type': type,
        'accept-ranges': 'bytes',
        'content-range': `bytes ${start}-${end}/${size}`,
        'content-length': end - start + 1,
      })
      return fs.createReadStream(file, { start, end }).pipe(res)
    }
    res.writeHead(200, { 'content-type': type, 'accept-ranges': 'bytes', 'content-length': size })
    fs.createReadStream(file).pipe(res)
  })
  await new Promise((r) => server.listen(port, r))
  return server
}
