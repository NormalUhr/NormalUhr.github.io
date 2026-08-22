// Headless screenshots of the demo files, driven over CDP so colour scheme and
// interactions (palette, depth dial, figure controls) can actually be exercised.
import fs from 'node:fs'
import path from 'node:path'
import puppeteer from 'puppeteer-core'

const HERE = path.dirname(new URL(import.meta.url).pathname)
const OUT = path.resolve(HERE, '..', 'shots')
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

// file | route | name | scheme | WxH | action
const SHOTS = process.argv.slice(2).map((s) => {
  const [file, route, name, scheme = 'dark', size = '1500x1000', action = ''] = s.split('|')
  const [w, h] = size.split('x').map(Number)
  return { file, route, name, scheme, w, h, action }
})

fs.mkdirSync(OUT, { recursive: true })
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--allow-file-access-from-files', '--font-render-hinting=none'] })

for (const s of SHOTS) {
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e.message).slice(0, 160)))
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 160)) })
  await page.setViewport({ width: s.w, height: s.h, deviceScaleFactor: 1 })
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: s.scheme }])
  await page.evaluateOnNewDocument(() => {
    try { sessionStorage.setItem('tty-booted', '1'); sessionStorage.setItem('booted', '1') } catch {}
  })
  const url = `file://${path.resolve(HERE, '..', 'demos', s.file)}${s.route}`
  await page.goto(url, { waitUntil: 'load' })
  await page.evaluate(() => document.fonts.ready)
  await new Promise((r) => setTimeout(r, 700))

  for (const action of s.action.split(',').filter(Boolean)) {
    if (action.startsWith('at:')) {
      await page.evaluate((q) => {
        const el = document.querySelector(q)
        if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 40, behavior: 'instant' })
      }, action.slice(3))
    } else {
      await page.evaluate((a) => {
        const A = {
          palette: () => document.getElementById('palbtn')?.click(),
          derive: () => document.querySelector('button[data-depth="derive"],button[data-mode="derive"]')?.click(),
          skim: () => document.querySelector('button[data-depth="skim"],button[data-mode="skim"]')?.click(),
          list: () => document.querySelector('[data-view-list]')?.click(),
          zh: () => document.querySelector('[data-l="zh"]')?.click(),
          neg: () => {
            const i = document.getElementById('clip-a-i')
            if (!i) return
            i.value = '-14'
            i.dispatchEvent(new Event('input', { bubbles: true }))
          },
        }
        A[a]?.()
      }, action)
    }
    await new Promise((r) => setTimeout(r, 450))
  }

  await page.screenshot({ path: path.join(OUT, `${s.name}.png`) })
  const kb = (fs.statSync(path.join(OUT, `${s.name}.png`)).size / 1024).toFixed(0)
  console.log(`${s.name.padEnd(22)} ${s.scheme.padEnd(5)} ${kb} KB${errors.length ? '  ERRORS: ' + errors.join('; ') : ''}`)
  await page.close()
}
await browser.close()
