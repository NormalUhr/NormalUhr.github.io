// Build-time markdown passes. Nothing here reaches the browser: equations become
// static KaTeX markup and code becomes pre-coloured spans, so no math or highlighting
// JavaScript ships with a page.
import { visit } from 'unist-util-visit'
import { createHighlighter } from 'shiki'

const highlighter = await createHighlighter({
  themes: ['github-dark-default', 'github-light-default'],
  langs: ['python', 'cpp', 'bash', 'json', 'javascript', 'yaml', 'diff', 'text'],
})

/** The fence labels actually used across the archive, mapped to Shiki grammars. */
const LANG_ALIAS = {
  'c++': 'cpp', cc: 'cpp', python3: 'python', py: 'python',
  makefile: 'bash', sh: 'bash', shell: 'bash',
  pseudo: 'text', perl: 'text', '': 'text',
}

function classList(node) {
  const raw = node.properties?.className ?? node.properties?.class
  if (Array.isArray(raw)) return raw
  return raw ? String(raw).split(/\s+/).filter(Boolean) : []
}

function setClasses(node, list) {
  delete node.properties.class
  node.properties.className = list
}

/**
 * Drop the article's opening H1. Every body in the archive begins by repeating its own
 * title, which the page heading already shows, and a second H1 is wrong for a document
 * that already has one. Applied at render time, so the markdown on disk stays a
 * byte-for-byte copy of what the author wrote.
 */
export function rehypeDropLeadingH1() {
  return (tree) => {
    const first = tree.children.findIndex((n) => n.type === 'element')
    if (first === -1 || tree.children[first].tagName !== 'h1') return
    tree.children.splice(first, 1)
  }
}

/**
 * Mark the opening paragraph for a drop cap, and say which script it starts with.
 * A Latin capital fills only the cap height of its em box while a Han character fills
 * the whole square, so the two need different sizes to look the same weight. Deciding
 * per paragraph rather than per article also covers a Chinese piece that opens on an
 * English word.
 *
 * Nothing is marked when the paragraph opens on markup or punctuation, since a floated
 * bracket or a formula reads as a mistake rather than a flourish.
 */
export function rehypeDropCap() {
  return (tree) => {
    const p = tree.children.find((n) => n.type === 'element' && n.tagName === 'p')
    if (!p) return
    const first = p.children[0]
    if (!first || first.type !== 'text') return
    const ch = first.value.trimStart().charAt(0)
    if (!ch) return
    const han = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(ch)
    const latin = /[0-9A-Za-z]/.test(ch)
    if (!han && !latin) return
    setClasses(p, [...classList(p), han ? 'cap-han' : 'cap-latin'])
  }
}

/** Highlight fenced code, leaving remark-math's blocks for KaTeX. */
export function rehypeShiki() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'pre' || !parent || index === null) return
      const code = node.children.find((c) => c.tagName === 'code')
      if (!code) return
      const classes = classList(code)
      if (classes.some((c) => c === 'math-display' || c === 'language-math')) return

      const source = code.children.map((c) => (c.type === 'text' ? c.value : '')).join('').replace(/\n$/, '')
      const declared = classes.find((c) => c.startsWith('language-'))
      let lang = declared ? declared.slice(9).toLowerCase() : ''
      lang = LANG_ALIAS[lang] ?? lang
      if (!highlighter.getLoadedLanguages().includes(lang)) lang = 'text'

      const pre = highlighter.codeToHast(source, {
        lang,
        themes: { dark: 'github-dark-default', light: 'github-light-default' },
        defaultColor: false,
        cssVariablePrefix: '--sh-',
      }).children[0]
      setClasses(pre, [...classList(pre), 'code-block'])
      pre.properties['data-lang'] = lang
      pre.properties['data-source'] = source
      parent.children[index] = pre
    })
  }
}

/**
 * Heading text for the table of contents. KaTeX writes every formula three ways: a
 * MathML tree, a raw-TeX <annotation>, and an aria-hidden HTML rendering. Keeping only
 * the MathML text leaves the Unicode form, so a heading reads "Policy π", not "π\\piπ".
 */
function headingText(node) {
  const skip = new Set()
  visit(node, 'element', (el) => {
    const hidden = el.properties?.ariaHidden === 'true' || el.properties?.['aria-hidden'] === 'true'
    if (hidden || el.tagName === 'annotation') visit(el, 'text', (t) => skip.add(t))
  })
  const out = []
  visit(node, 'text', (t) => { if (!skip.has(t)) out.push(t.value) })
  return out.join('').replace(/\s+/g, ' ').trim()
}

/**
 * Collect the table of contents ourselves rather than using Astro's, whose collector
 * runs after KaTeX and therefore sees the tripled text above.
 */
export function rehypeCollectToc() {
  return (tree, file) => {
    const toc = []
    visit(tree, 'element', (node) => {
      if (!/^h[23]$/.test(node.tagName) || !node.properties?.id) return
      toc.push({ depth: Number(node.tagName[1]), slug: String(node.properties.id), text: headingText(node) })
    })
    if (file?.data?.astro?.frontmatter) file.data.astro.frontmatter.toc = toc
  }
}

/**
 * Number each display equation and give it an anchor, the way a paper does. This is
 * what the equation index in the sidebar and any "Eq. (n)" reference link to.
 */
export function rehypeNumberEquations() {
  return (tree, file) => {
    let count = 0
    visit(tree, 'element', (node, index, parent) => {
      if (!parent || index === null || !classList(node).includes('katex-display')) return
      count += 1
      parent.children[index] = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['eqn'], 'data-eqn': String(count), id: `eq-${count}` },
        children: [
          node,
          {
            type: 'element',
            tagName: 'a',
            properties: { className: ['eqn-tag'], href: `#eq-${count}`, 'aria-label': `Equation ${count}` },
            children: [{ type: 'text', value: `(${count})` }],
          },
        ],
      }
      return 'skip'
    })
    if (file?.data?.astro?.frontmatter) file.data.astro.frontmatter.eqnCount = count
  }
}
