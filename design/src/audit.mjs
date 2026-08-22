// Walk every route in every theme, in both languages, and report JS errors or leftover artefacts.
import path from 'node:path'
import puppeteer from 'puppeteer-core'
const HERE = path.dirname(new URL(import.meta.url).pathname)
const ROUTES = ['#/', '#/articles', '#/articles?c=post-training', '#/articles?c=systems', '#/articles?c=music',
                '#/articles?c=archive', '#/articles?t=GRPO', '#/posts', '#/archive', '#/s/grpo-line', '#/s/const',
                '#/p/grpo', '#/p/decorators', '#/p/vogel-im-kafig', '#/p/ai-infra', '#/p/effective-c',
                '#/about', '#/search?q=grpo', '#/p/grpo?at=eq-4']
const b = await puppeteer.launch({ executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true })
let bad = 0
for (const theme of ['atlas.html']) {
  for (const lang of ['en','zh']) {
    const p = await b.newPage()
    await p.setViewport({ width: 1400, height: 900 })
    const errs = []
    p.on('pageerror', e => errs.push(String(e.message).slice(0,120)))
    p.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0,120)) })
    await p.evaluateOnNewDocument(()=>{ try{sessionStorage.setItem('tty-booted','1')}catch{} })
    await p.goto('file://'+path.resolve(HERE,'..','demos',theme), { waitUntil:'load' })
    if (lang === 'zh') { await p.evaluate(()=>document.querySelector('[data-l="zh"]')?.click()); await new Promise(r=>setTimeout(r,150)) }
    for (const r of ROUTES) {
      await p.evaluate(h => { location.hash = h }, r)
      await new Promise(res => setTimeout(res, 90))
      const junk = await p.evaluate(() => {
        const t = document.body.innerText
        const hits = []
        if (/undefined|NaN|\[object Object\]/.test(t)) hits.push('placeholder-text')
        if (/nArticles\(|\bt\.[a-z]+\b(?!\w)|undefined/.test(t)) hits.push('unrendered-expr')
        if (/\b1 articles\b/.test(t)) hits.push('bad-plural')
        if (document.documentElement.scrollWidth > window.innerWidth + 2) hits.push('h-overflow')
        return hits
      })
      if (junk.length) { console.log(`  ${theme} ${lang} ${r}: ${junk.join(', ')}`); bad++ }
    }
    if (errs.length) { console.log(`  ${theme} ${lang} JS: ${[...new Set(errs)].join(' | ')}`); bad++ }
    await p.close()
  }
}
await b.close()
console.log(bad ? `\n${bad} issue group(s)` : '\nclean: no JS errors, no unrendered expressions, no horizontal overflow')
