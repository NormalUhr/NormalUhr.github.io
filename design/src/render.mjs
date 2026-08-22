// Markdown -> HTML pipeline mirroring what the production site would run:
// remark-math + rehype-katex for build-time equations, Shiki for build-time highlighting.
// Nothing here needs a browser, so posts ship as static HTML with no math or highlight JS.
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkRehype from 'remark-rehype'
import rehypeRaw from 'rehype-raw'
import rehypeSlug from 'rehype-slug'
import rehypeKatex from 'rehype-katex'
import rehypeStringify from 'rehype-stringify'
import { visit } from 'unist-util-visit'
import { createHighlighter } from 'shiki'

const highlighter = await createHighlighter({
  themes: ['github-dark-default', 'github-light-default'],
  langs: ['python', 'cpp', 'bash', 'json', 'javascript', 'diff', 'text'],
})

const LANG_ALIAS = { 'c++': 'cpp', pseudo: 'text', perl: 'text', makefile: 'bash', '': 'text' }

const DEEPER_HEADINGS = { en: 'Mathematical Correspondence', zh: '数学对应' }

/**
 * Class list of a hast node. Shiki builds its own hast and stores classes as a `class`
 * string rather than the `className` array rehype uses, so both spellings must be read
 * (and merged on write, or the serialiser emits two class attributes and the browser
 * silently keeps only the first).
 */
function classes(node) {
  const raw = node.properties?.className ?? node.properties?.class
  if (Array.isArray(raw)) return raw
  return raw ? String(raw).split(/\s+/).filter(Boolean) : []
}

function setClasses(node, list) {
  delete node.properties.class
  node.properties.className = list
}

/**
 * Text of a node for the table of contents. KaTeX emits every formula three ways: a MathML
 * tree, a raw-TeX <annotation>, and an aria-hidden HTML rendering. Keeping only the MathML
 * text leaves the Unicode form (\\pi -> π), which is what a TOC entry should say.
 */
function textOf(node) {
  const skip = new Set()
  visit(node, 'element', (el) => {
    const hidden = el.properties?.ariaHidden === 'true' || el.properties?.['aria-hidden'] === 'true'
    if (hidden || el.tagName === 'annotation') visit(el, 'text', (t) => skip.add(t))
  })
  const out = []
  visit(node, 'text', (t) => { if (!skip.has(t)) out.push(t.value) })
  return out.join('').replace(/\s+/g, ' ').trim()
}

/** Replace <pre><code class="language-x"> with Shiki markup carrying both light and dark themes. */
function rehypeShiki() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'pre' || !parent || index === null) return
      const code = node.children.find((c) => c.tagName === 'code')
      if (!code) return
      const codeClasses = classes(code)
      // remark-math emits display math as <pre><code class="language-math math-display">; leave it for KaTeX.
      if (codeClasses.some((c) => c === 'math-display' || c === 'language-math')) return
      const source = code.children.map((c) => (c.type === 'text' ? c.value : '')).join('').replace(/\n$/, '')
      const declared = codeClasses.find((c) => c.startsWith('language-'))
      let lang = declared ? declared.slice(9).toLowerCase() : ''
      lang = LANG_ALIAS[lang] ?? lang
      if (!highlighter.getLoadedLanguages().includes(lang)) lang = 'text'
      const pre = highlighter.codeToHast(source, {
        lang,
        themes: { dark: 'github-dark-default', light: 'github-light-default' },
        defaultColor: false,
        cssVariablePrefix: '--sh-',
      }).children[0]
      setClasses(pre, [...classes(pre), 'code-block'])
      pre.properties['data-lang'] = lang
      pre.properties['data-source'] = source
      parent.children[index] = pre
    })
  }
}

/** Wrap each rendered display equation so themes can hang a "(n)" tag in the gutter. */
function rehypeNumberEquations() {
  return (tree, file) => {
    let n = 0
    visit(tree, 'element', (node, index, parent) => {
      if (!parent || index === null || !classes(node).includes('katex-display')) return
      n += 1
      parent.children[index] = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['eqn'], 'data-eqn': String(n), id: `eq-${n}` },
        children: [
          node,
          {
            type: 'element',
            tagName: 'a',
            properties: { className: ['eqn-tag'], href: `#eq-${n}`, 'aria-label': `Equation ${n}` },
            children: [{ type: 'text', value: `(${n})` }],
          },
        ],
      }
      return 'skip'
    })
    file.data.eqnCount = n
  }
}

function rehypeCollectToc() {
  return (tree, file) => {
    const toc = []
    visit(tree, 'element', (node) => {
      if (!/^h[23]$/.test(node.tagName) || !node.properties?.id) return
      toc.push({ id: node.properties.id, depth: Number(node.tagName[1]), text: textOf(node) })
    })
    file.data.toc = toc
  }
}

/**
 * Author-level transforms applied before parsing.
 * Folding the recurring derivation sections into <details> is what gives the themes
 * something real to drive their progressive-disclosure / depth controls with.
 */
function prepare(md, lang) {
  const summary = DEEPER_HEADINGS[lang] ?? DEEPER_HEADINGS.en
  const lines = md.split('\n')
  const out = []
  let i = 0
  while (i < lines.length) {
    if (!/^###\s+(Mathematical Correspondence|数学对应)\s*$/.test(lines[i])) {
      out.push(lines[i])
      i += 1
      continue
    }
    i += 1
    const body = []
    while (i < lines.length && !/^#{1,4}\s/.test(lines[i]) && !/^---\s*$/.test(lines[i])) {
      body.push(lines[i])
      i += 1
    }
    out.push('<details class="deeper" data-deeper>', `<summary>${summary}</summary>`, '', ...body, '', '</details>', '')
  }
  let joined = out.join('\n')
  const at = joined.indexOf('## 5. Reference Model')
  if (at > -1) joined = `${joined.slice(0, at)}<div data-figure="ppo-clip"></div>\n\n${joined.slice(at)}`
  return joined
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkMath)
  .use(remarkRehype, { allowDangerousHtml: true })
  .use(rehypeRaw)
  .use(rehypeSlug)
  .use(rehypeShiki)
  .use(rehypeKatex, { throwOnError: false, strict: false, trust: true })
  .use(rehypeNumberEquations)
  .use(rehypeCollectToc)
  .use(rehypeStringify, { allowDangerousHtml: true })

/** Rough length metrics that work for both CJK and Latin text. */
function measure(md) {
  const plain = md
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/[#*_>|`\-]/g, ' ')
  const cjk = (plain.match(/[一-鿿]/g) || []).length
  const latin = (plain.match(/[A-Za-z][A-Za-z'’-]*/g) || []).length
  const words = cjk + latin
  return { words, minutes: Math.max(1, Math.round(words / (cjk > latin ? 400 : 220))) }
}

export async function renderMarkdown(md, lang) {
  const file = await processor.process(prepare(md, lang))
  return {
    html: String(file),
    toc: file.data.toc ?? [],
    eqnCount: file.data.eqnCount ?? 0,
    ...measure(md),
  }
}
