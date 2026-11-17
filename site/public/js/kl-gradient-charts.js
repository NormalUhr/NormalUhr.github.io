/* Figures of the article on the gradient of KL estimators.
 *
 * The conventions follow the figures of the Score Function Centering articles.
 * Every colour is one of the page's custom properties, so the light and dark
 * themes need no redraw. The SVG carries no viewBox and is resized in pixels,
 * and every piece of text is an HTML element beside it, so labels keep their
 * size on a 390px screen. Labels are English on both language pages.
 *
 * Every curve is computed here from its formula. The fixed points of the two
 * examples are the one exception: they come from iterating the expected logit
 * update until it stops moving, and they are embedded in FIXED below.
 */
(() => {
  /* Limits of the expected logit update in the two examples. qp is
   * KL(generator || theta), k4 is half the variance of log r under the generator,
   * and pq is KL(theta || generator). A 0 is a token whose probability goes to
   * zero; k4 is in closed form, so its small values are kept as they are. */
  const FIXED = [{"q":[0.6,0.3,0.1],"A":[0.5,0,-0.5],"rows":[{"beta":0.01,"qp":[1,0,0],"k4":[1,9.64e-23,6.2e-45],"pq":[1,0,0]},{"beta":0.015,"qp":[1,0,0],"k4":[1,1.67e-15,1.86e-30],"pq":[1,0,0]},{"beta":0.02,"qp":[1,0,0],"k4":[1,6.94e-12,3.21e-23],"pq":[1,0,0]},{"beta":0.03,"qp":[1,0,0],"k4":[1,2.89e-8,5.56e-16],"pq":[1,0,0]},{"beta":0.05,"qp":[1,0,0],"k4":[1,0.0000227,3.44e-10],"pq":[1,0,0]},{"beta":0.07,"qp":[1,0,0],"k4":[1,0.000395,1.04e-7],"pq":[1,0,0]},{"beta":0.1,"qp":[1,0,0],"k4":[0.997,0.00336,0.00000754],"pq":[1,0,0]},{"beta":0.15,"qp":[1,0,0],"k4":[0.982,0.0175,0.000208],"pq":[1,0,0]},{"beta":0.2,"qp":[1,0,0],"k4":[0.96,0.0394,0.00108],"pq":[1,0,0]},{"beta":0.3,"qp":[1,0,0],"k4":[0.909,0.0858,0.0054],"pq":[1,0,0]},{"beta":0.5,"qp":[0.875,0.125,0],"k4":[0.829,0.152,0.0187],"pq":[1,0,0]},{"beta":0.7,"qp":[0.8107,0.1893,0],"k4":[0.778,0.191,0.0311],"pq":[1,0,0]},{"beta":1,"qp":[0.75,0.225,0.025],"k4":[0.733,0.222,0.0449],"pq":[0.8028,0.1972,0]},{"beta":1.5,"qp":[0.7,0.25,0.05],"k4":[0.693,0.248,0.0593],"pq":[0.7487,0.2513,0]},{"beta":2,"qp":[0.675,0.2625,0.0625],"k4":[0.671,0.261,0.0678],"pq":[0.6884,0.2684,0.04314]},{"beta":3,"qp":[0.65,0.275,0.075],"k4":[0.648,0.274,0.0774],"pq":[0.6529,0.2762,0.07082]},{"beta":5,"qp":[0.63,0.285,0.085],"k4":[0.629,0.285,0.0859],"pq":[0.6308,0.2854,0.08383]},{"beta":7,"qp":[0.6214,0.2893,0.08929],"k4":[0.621,0.289,0.0897],"pq":[0.6218,0.2895,0.08874]},{"beta":10,"qp":[0.615,0.2925,0.0925],"k4":[0.615,0.292,0.0927],"pq":[0.6152,0.2926,0.09225]}]},{"q":[0.9,0.09,0.01],"A":[0,0,0.5],"rows":[{"beta":0.01,"qp":[0.45,0.045,0.505],"k4":[1.74e-20,1.74e-21,1],"pq":[0.733,0.0733,0.1937]},{"beta":0.015,"qp":[0.6,0.06,0.34],"k4":[3e-13,3e-14,1],"pq":[0.7832,0.07832,0.1384]},{"beta":0.02,"qp":[0.675,0.0675,0.2575],"k4":[1.25e-9,1.25e-10,1],"pq":[0.8084,0.08084,0.1108]},{"beta":0.03,"qp":[0.75,0.075,0.175],"k4":[0.0000052,5.2e-7,1],"pq":[0.8342,0.08342,0.08233]},{"beta":0.05,"qp":[0.81,0.081,0.109],"k4":[0.00407,0.000407,0.996],"pq":[0.8563,0.08563,0.05809]},{"beta":0.07,"qp":[0.8357,0.08357,0.08071],"k4":[0.066,0.0066,0.927],"pq":[0.8665,0.08665,0.04687]},{"beta":0.1,"qp":[0.855,0.0855,0.0595],"k4":[0.364,0.0364,0.6],"pq":[0.8747,0.08747,0.03784]},{"beta":0.15,"qp":[0.87,0.087,0.043],"k4":[0.708,0.0708,0.221],"pq":[0.8816,0.08816,0.03021]},{"beta":0.2,"qp":[0.8775,0.08775,0.03475],"k4":[0.809,0.0809,0.11],"pq":[0.8854,0.08854,0.02607]},{"beta":0.3,"qp":[0.885,0.0885,0.0265],"k4":[0.863,0.0863,0.0508],"pq":[0.8895,0.08895,0.02159]},{"beta":0.5,"qp":[0.891,0.0891,0.0199],"k4":[0.885,0.0885,0.0267],"pq":[0.8931,0.08931,0.0176]},{"beta":0.7,"qp":[0.8936,0.08936,0.01707],"k4":[0.891,0.0891,0.0202],"pq":[0.8948,0.08948,0.01571]},{"beta":1,"qp":[0.8955,0.08955,0.01495],"k4":[0.894,0.0894,0.0164],"pq":[0.8962,0.08962,0.01419]},{"beta":1.5,"qp":[0.897,0.0897,0.0133],"k4":[0.896,0.0896,0.0139],"pq":[0.8973,0.08973,0.01292]},{"beta":2,"qp":[0.8978,0.08978,0.01247],"k4":[0.897,0.0897,0.0128],"pq":[0.898,0.0898,0.01224]},{"beta":3,"qp":[0.8985,0.08985,0.01165],"k4":[0.898,0.0898,0.0118],"pq":[0.8986,0.08986,0.01154]},{"beta":5,"qp":[0.8991,0.08991,0.01099],"k4":[0.899,0.0899,0.011],"pq":[0.8991,0.08991,0.01095]},{"beta":7,"qp":[0.8994,0.08994,0.01071],"k4":[0.899,0.0899,0.0107],"pq":[0.8994,0.08994,0.01068]},{"beta":10,"qp":[0.8995,0.08996,0.01049],"k4":[0.9,0.09,0.0105],"pq":[0.8996,0.08996,0.01048]}]}]

  const INK = {
    qp: 'var(--accent)', k4: 'var(--moss)', pq: 'var(--vermilion)',
    t1: 'var(--minute-bridge)', t2: 'var(--stone)', t3: 'var(--hour-bridge)',
  }
  const OPTION = {
    qp: 'KL(π<sub>generator</sub> ‖ π<sub>θ</sub>)',
    k4: 'k<sub>4</sub>',
    pq: 'KL(π<sub>θ</sub> ‖ π<sub>generator</sub>)',
  }

  const CSS = `
.kfig { margin: 1.9rem 0; }
.kfig-title { font: 600 .95rem/1.35 var(--sans); color: var(--ink-strong); margin: 0 0 .5rem; }
.kfig-title span { font-weight: 400; font-size: .8rem; color: var(--ink-faint); }
.kfig-head { display: flex; flex-direction: column; gap: .3rem; margin-bottom: .6rem; }
.kfig ul.kfig-legend { display: flex; flex-wrap: wrap; gap: .3rem .95rem; list-style: none; margin: 0; padding: 0; }
.kfig-legend li { display: flex; align-items: center; gap: .45em; cursor: pointer;
  font: 500 .78rem/1.4 var(--sans); color: var(--ink-soft); -webkit-user-select: none; user-select: none; }
.kfig-legend li[aria-disabled="true"] { cursor: default; }
.kfig-legend li[aria-pressed="false"] { opacity: .35; }
.kfig-legend li:focus-visible { outline: 1px solid var(--accent); outline-offset: 2px; }
.kfig-sw { width: 1.6em; height: 0; flex: none; border-top: 2px solid var(--sw, var(--ink-faint)); }
.kfig-sw.dash { border-top-style: dashed; }
.kfig-sw.thick { border-top-width: 6px; opacity: .4; }
.kfig-sw.box { width: 1.1em; height: .7em; border: 0; border-radius: 1px; background: var(--sw); }
/* The readout keeps one fixed line, two on a phone, whatever it holds: a
 * readout that grew on hover would push the figure away from the cursor,
 * end the hover and shrink again, over and over. */
.kfig-readout { font: .74rem/1.4 var(--mono); color: var(--ink-soft); height: 1.4em;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.kfig-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: .45rem 1.3rem; margin: 0 0 .7rem;
  font: .78rem/1.4 var(--sans); color: var(--ink-soft); }
.kfig-ctl label { display: flex; align-items: center; gap: .55em; }
.kfig-ctl input[type=range] { width: 10.5em; accent-color: var(--accent); }
.kfig-ctl output { font: .76rem/1 var(--mono); color: var(--ink); min-width: 3.4em; }
.kfig-seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 3px; overflow: hidden; }
.kfig-seg button { font: 500 .76rem/1.3 var(--sans); color: var(--ink-soft); background: none; border: 0;
  padding: .3em .75em; cursor: pointer; }
.kfig-seg button + button { border-left: 1px solid var(--rule); }
.kfig-seg button[aria-pressed="true"] { background: var(--accent-soft); color: var(--ink-strong); }
.kfig-seg button:focus-visible { outline: 1px solid var(--accent); outline-offset: -2px; }
.kfig-pane { position: relative; height: 250px; padding: 6px 8px 1.5em 3.3em; }
.kfig-pane svg { position: absolute; overflow: visible; }
.kfig-pane svg .grid { stroke: var(--rule-soft); stroke-width: 1; shape-rendering: crispEdges; }
.kfig-pane svg .axis { stroke: var(--rule); stroke-width: 1; shape-rendering: crispEdges; }
.kfig-pane svg .ln { fill: none; stroke-linejoin: round; stroke-linecap: round; }
.kfig-pane svg .ref { fill: none; stroke-width: 1.2; stroke-dasharray: 5 4; }
.kfig-pane svg .mk { stroke-width: 1.6; fill: var(--paper); }
.kfig-pane svg .guide { stroke: var(--ink-faint); stroke-width: 1; stroke-dasharray: 2 3; opacity: 0; }
.kfig-pane svg .hit { fill: transparent; }
.kfig-pane svg .dot { opacity: 0; }
.kfig-pane.live svg .guide, .kfig-pane.live svg .dot { opacity: 1; }
.kfig-ty, .kfig-tx { position: absolute; font: .68rem/1 var(--mono); color: var(--ink-faint); white-space: nowrap; }
.kfig-ty { transform: translateY(-50%); text-align: right; }
.kfig-tx { bottom: .15em; transform: translateX(-50%); }
.kfig-ty sup, .kfig-tx sup { font-size: .8em; }
.kfig-reflab { position: absolute; font: .68rem/1.2 var(--sans); color: var(--ink-soft); white-space: nowrap;
  background: var(--paper); padding: 0 .2em; pointer-events: none; }
.kfig-axlabel { font: .72rem/1.4 var(--sans); color: var(--ink-faint); text-align: center; margin-top: .2rem; }
.kfig-status { font: .76rem/1.5 var(--sans); color: var(--ink-soft); margin-top: .45rem; min-height: 3em; }
.kfig-status b { font-weight: 600; color: var(--ink); }

.kfig-plot { position: relative; }
.kfig-plot.bars { padding: 0 3.6em; }
.kfig-group + .kfig-group { margin-top: .85rem; }
.kfig-gname { font: 600 .79rem/1.4 var(--sans); color: var(--ink); margin-bottom: .25rem; }
.kfig-gname span { font-weight: 400; color: var(--ink-faint); }
.kfig-row { position: relative; height: 15px; margin: 3px 0; transition: opacity .12s; }
.kfig-row.off { display: none; }
.kfig-plot:hover .kfig-row { opacity: .5; }
.kfig-plot .kfig-row:hover { opacity: 1; }
.kfig-zero { position: absolute; top: -3px; bottom: -3px; width: 1px; background: var(--rule); pointer-events: none; }
.kfig-gridv { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--rule-soft); pointer-events: none; }
.kfig-bar { position: absolute; top: 0; bottom: 0; min-width: 1px; background: var(--sw); }
.kfig-val { position: absolute; top: 50%; transform: translateY(-50%); z-index: 2; padding: 0 3px;
  font: .72rem/1 var(--mono); color: var(--ink-soft); white-space: nowrap; }
.kfig-axis { position: relative; height: 1.4em; margin-top: .3rem; }
.kfig-axis span { position: absolute; top: .25em; font: .7rem/1 var(--mono); color: var(--ink-faint);
  transform: translateX(-50%); }
.kfig-axis b { position: absolute; left: 0; right: 0; top: 0; height: 1px; background: var(--rule); }

.kfig-stack + .kfig-stack { margin-top: .75rem; }
.kfig-shead { display: flex; justify-content: space-between; align-items: baseline; gap: .8em;
  font: 600 .79rem/1.4 var(--sans); color: var(--ink); margin-bottom: .2rem; }
.kfig-shead b { font-weight: 600; }
.kfig-shead span { font: .72rem/1.4 var(--mono); font-weight: 400; color: var(--ink-soft); white-space: nowrap; }
.kfig-sbar { display: flex; height: 15px; border-radius: 1px; overflow: hidden; background: var(--rule-soft); }
.kfig-sbar i { display: block; height: 100%; background: var(--sw); transition: width .25s ease; }
.kfig-sbar i + i { box-shadow: inset 1px 0 0 var(--paper); }
.kfig-sbar i:hover { filter: brightness(1.15); }
.kfig-note { font: .74rem/1.45 var(--mono); color: var(--ink-soft); margin: -.25rem 0 .7rem; }
@media (prefers-reduced-motion: reduce) { .kfig-sbar i { transition: none; } }
@media (max-width: 640px) {
  .kfig-readout { height: 2.8em; white-space: normal; display: -webkit-box;
    -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
  .kfig-pane { height: 220px; }
  .kfig-status { min-height: 4.5em; }
}`

  const svgEl = (n, a) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', n)
    for (const k in a) e.setAttribute(k, a[k])
    return e
  }
  const el = (tag, cls, html) => {
    const e = document.createElement(tag)
    if (cls) e.className = cls
    if (html != null) e.innerHTML = html
    return e
  }
  const plain = (html) => html.replace(/<[^>]+>/g, '')
  const sci = (v, d = 1) => {
    if (v === 0) return '0'
    const [m, e] = v.toExponential(d).split('e')
    return m + 'e' + (e[0] === '+' ? e.slice(1) : e)
  }
  const pow10 = (k) => '10<sup>' + (k < 0 ? '−' + -k : k) + '</sup>'
  const minus = (s) => s.replace(/^-/, '−')
  const fx = (v, d) => minus(v.toFixed(d))
  let uid = 0

  function style() {
    if (document.getElementById('kfig-css')) return
    const s = el('style')
    s.id = 'kfig-css'
    s.textContent = CSS
    document.head.appendChild(s)
  }

  function title(fig, html, note) {
    const t = el('div', 'kfig-title', html)
    if (note) t.appendChild(el('span', null, '  ' + note))
    fig.appendChild(t)
  }

  /* A legend whose entries show or hide a series. */
  function legendOf(items, hidden, onToggle) {
    const ul = el('ul', 'kfig-legend')
    ul.setAttribute('aria-label', 'Show or hide a curve')
    const keys = items.filter((s) => !s.ref).map((s) => s.key)
    for (const s of items) {
      const li = el('li')
      const sw = el('span', 'kfig-sw' + (s.dash ? ' dash' : '') + (s.thick ? ' thick' : '') + (s.box ? ' box' : ''))
      sw.style.setProperty('--sw', s.ink)
      li.append(sw, el('span', null, s.name))
      if (s.ref) {
        li.setAttribute('aria-disabled', 'true')
      } else {
        li.tabIndex = 0
        li.setAttribute('role', 'button')
        li.setAttribute('aria-pressed', 'true')
        const flip = () => {
          // Hiding the last visible series would leave an empty figure.
          if (!hidden.has(s.key) && hidden.size >= keys.length - 1) return
          if (hidden.has(s.key)) hidden.delete(s.key); else hidden.add(s.key)
          li.setAttribute('aria-pressed', hidden.has(s.key) ? 'false' : 'true')
          onToggle()
        }
        li.addEventListener('click', flip)
        li.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip() }
        })
      }
      ul.appendChild(li)
    }
    return ul
  }

  /* Sliders and segmented buttons that write into the figure's state. */
  function controls(list, st, onChange) {
    const row = el('div', 'kfig-ctl')
    for (const c of list) {
      if (c.type === 'segment') {
        const seg = el('div', 'kfig-seg')
        seg.setAttribute('role', 'group')
        seg.setAttribute('aria-label', plain(c.label))
        const buttons = c.options.map((o, i) => {
          const b = el('button', null, o)
          b.type = 'button'
          b.setAttribute('aria-pressed', st[c.key] === i ? 'true' : 'false')
          b.addEventListener('click', () => {
            st[c.key] = i
            buttons.forEach((x, j) => x.setAttribute('aria-pressed', j === i ? 'true' : 'false'))
            onChange()
          })
          seg.appendChild(b)
          return b
        })
        row.appendChild(seg)
        continue
      }
      const lab = el('label')
      lab.appendChild(el('span', null, c.label))
      const input = el('input')
      input.type = 'range'
      const values = c.values
      if (values) {
        input.min = 0
        input.max = values.length - 1
        input.step = 1
        input.value = values.indexOf(st[c.key])
      } else {
        input.min = c.min
        input.max = c.max
        input.step = c.step
        input.value = st[c.key]
      }
      const out = el('output')
      const show = () => { out.innerHTML = c.fmt(st[c.key]) }
      input.addEventListener('input', () => {
        st[c.key] = values ? values[+input.value] : +input.value
        show()
        onChange()
      })
      show()
      lab.append(input, out)
      row.appendChild(lab)
    }
    return row
  }

  /* ---- line charts ---------------------------------------------------- */

  function lineChart(fig, spec) {
    const st = Object.assign({}, spec.state)
    const hidden = new Set()
    const clip = 'kclip' + ++uid
    fig.textContent = ''
    fig.classList.add('kfig')
    title(fig, spec.title, spec.note)

    const head = el('div', 'kfig-head')
    const readout = el('div', 'kfig-readout')
    readout.setAttribute('aria-live', 'polite')
    const REST = spec.rest || 'hover the plot to read values'
    readout.textContent = REST
    head.append(legendOf(spec.series.concat(spec.keys || []), hidden, () => draw()), readout)
    fig.appendChild(head)
    if (spec.controls) fig.appendChild(controls(spec.controls, st, () => draw()))

    const pane = el('div', 'kfig-pane')
    const svg = svgEl('svg')
    pane.appendChild(svg)
    fig.appendChild(pane)
    if (spec.xlabel) fig.appendChild(el('div', 'kfig-axlabel', spec.xlabel))
    const status = spec.status ? el('div', 'kfig-status') : null
    if (status) fig.appendChild(status)

    function draw() {
      const cs = getComputedStyle(pane)
      const pl = parseFloat(cs.paddingLeft), pr = parseFloat(cs.paddingRight)
      const pt = parseFloat(cs.paddingTop), pb = parseFloat(cs.paddingBottom)
      const w = pane.clientWidth - pl - pr
      const h = pane.clientHeight - pt - pb
      if (w <= 0 || h <= 0) return
      svg.setAttribute('width', w)
      svg.setAttribute('height', h)
      svg.style.left = pl + 'px'
      svg.style.top = pt + 'px'
      svg.textContent = ''
      pane.querySelectorAll('.kfig-ty, .kfig-tx, .kfig-reflab').forEach((n) => n.remove())
      pane.classList.remove('live')

      const xa = spec.x
      const ya = typeof spec.y === 'function' ? spec.y(st) : spec.y
      const tx = xa.log ? Math.log10 : (v) => v
      const ty = ya.log ? Math.log10 : (v) => v
      const X = (v) => ((tx(v) - tx(xa.min)) / (tx(xa.max) - tx(xa.min))) * w
      const Y = (v) => h - ((ty(v) - ty(ya.min)) / (ty(ya.max) - ty(ya.min))) * h
      const atPx = (px) => {
        const u = tx(xa.min) + (px / w) * (tx(xa.max) - tx(xa.min))
        return xa.log ? Math.pow(10, u) : u
      }

      const defs = svgEl('defs')
      const cp = svgEl('clipPath', { id: clip })
      cp.appendChild(svgEl('rect', { x: 0, y: -2, width: w, height: h + 4 }))
      defs.appendChild(cp)
      svg.appendChild(defs)

      for (const t of ya.ticks) {
        const v = typeof t === 'object' ? t.v : t
        const y = Y(v)
        const zero = !ya.log && v === 0
        svg.appendChild(svgEl('line', { class: zero ? 'axis' : 'grid', x1: 0, x2: w, y1: y, y2: y }))
        const lab = el('div', 'kfig-ty', typeof t === 'object' ? t.l : (ya.fmt ? ya.fmt(v) : String(v)))
        lab.style.top = pt + y + 'px'
        lab.style.right = 'calc(100% - ' + (pl - 5) + 'px)'
        pane.appendChild(lab)
      }
      // On a narrow pane every other tick is enough, and labels at either end
      // are aligned inward so they never hang off the pane.
      const every = w < 300 && xa.ticks.length > 5 ? 2 : 1
      xa.ticks.forEach((t, i) => {
        if (i % every) return
        const v = typeof t === 'object' ? t.v : t
        const x = X(v)
        if (xa.grid) svg.appendChild(svgEl('line', { class: 'grid', x1: x, x2: x, y1: 0, y2: h }))
        const lab = el('div', 'kfig-tx', typeof t === 'object' ? t.l : (xa.fmt ? xa.fmt(v) : String(v)))
        lab.style.left = pl + x + 'px'
        if (x < 8) lab.style.transform = 'none'
        else if (x > w - 8) lab.style.transform = 'translateX(-100%)'
        pane.appendChild(lab)
      })
      svg.appendChild(svgEl('line', { class: 'axis', x1: 0, x2: w, y1: h, y2: h }))
      svg.appendChild(svgEl('line', { class: 'axis', x1: 0, x2: 0, y1: 0, y2: h }))

      for (const r of spec.refs ? spec.refs(st) : []) {
        const line = r.h != null
          ? svgEl('line', { class: 'ref', x1: 0, x2: w, y1: Y(r.h), y2: Y(r.h), 'clip-path': 'url(#' + clip + ')' })
          : svgEl('line', { class: 'ref', x1: X(r.v), x2: X(r.v), y1: 0, y2: h, 'clip-path': 'url(#' + clip + ')' })
        line.style.stroke = r.ink || 'var(--ink-soft)'
        if (r.solid) line.style.strokeDasharray = 'none'
        svg.appendChild(line)
        if (r.label) {
          const lab = el('div', 'kfig-reflab', r.label)
          if (r.h != null) {
            lab.style.top = pt + Y(r.h) - 16 + 'px'
            lab.style.right = pr + 2 + 'px'
          } else {
            lab.style.top = pt + 2 + 'px'
            lab.style.left = pl + X(r.v) + 4 + 'px'
          }
          pane.appendChild(lab)
          // A label that would run past the right edge goes on the other side of its line.
          if (r.h == null && pl + X(r.v) + 4 + lab.offsetWidth > pane.clientWidth) {
            lab.style.left = pl + X(r.v) - 4 - lab.offsetWidth + 'px'
          }
        }
      }

      const n = spec.samples || 400
      const live = []
      for (const s of spec.series) {
        if (hidden.has(s.key)) continue
        let d = ''
        let pen = false
        for (let i = 0; i <= n; i++) {
          const x = atPx((i / n) * w)
          const y = s.fn(x, st)
          if (!Number.isFinite(y) || (ya.log && y <= 0)) { pen = false; continue }
          const yy = Math.max(-h, Math.min(2 * h, Y(y)))
          d += (pen ? 'L' : 'M') + ((i / n) * w).toFixed(1) + ' ' + yy.toFixed(1)
          pen = true
        }
        const path = svgEl('path', { class: 'ln', d, 'clip-path': 'url(#' + clip + ')' })
        path.style.stroke = s.ink
        path.style.strokeWidth = s.thick ? 6 : 1.7
        if (s.thick) path.style.opacity = 0.4
        if (s.dash) path.style.strokeDasharray = '6 5'
        svg.appendChild(path)
        live.push(s)
      }

      for (const m of spec.marks ? spec.marks(st) : []) {
        if (hidden.has(m.key)) continue
        if (!Number.isFinite(m.y) || m.y < ya.min || m.y > ya.max) continue
        const c = svgEl('circle', { class: 'mk', cx: X(m.x), cy: Y(m.y), r: 3.6 })
        c.style.stroke = m.ink
        if (m.fill) c.style.fill = m.ink
        svg.appendChild(c)
      }

      const guide = svgEl('line', { class: 'guide', y1: 0, y2: h, x1: 0, x2: 0 })
      svg.appendChild(guide)
      const dots = live.map((s) => {
        const c = svgEl('circle', { class: 'mk dot', r: 3, cx: 0, cy: 0 })
        c.style.stroke = s.ink
        svg.appendChild(c)
        return c
      })
      const hit = svgEl('rect', { class: 'hit', x: 0, y: 0, width: w, height: h })
      svg.appendChild(hit)
      const move = (ev) => {
        const r = svg.getBoundingClientRect()
        const px = Math.max(0, Math.min(w, ev.clientX - r.left))
        const x = atPx(px)
        guide.setAttribute('x1', px)
        guide.setAttribute('x2', px)
        pane.classList.add('live')
        const vals = {}
        live.forEach((s, i) => {
          const y = s.fn(x, st)
          vals[s.key] = y
          const ok = Number.isFinite(y) && !(ya.log && y <= 0) && y >= ya.min && y <= ya.max
          dots[i].style.display = ok ? '' : 'none'
          if (ok) { dots[i].setAttribute('cx', px); dots[i].setAttribute('cy', Y(y)) }
        })
        readout.innerHTML = spec.readout(x, st, vals, live.map((s) => s.key))
      }
      hit.addEventListener('pointermove', move)
      hit.addEventListener('pointerdown', move)
      hit.addEventListener('pointerleave', () => {
        pane.classList.remove('live')
        readout.textContent = REST
      })
      if (status) status.innerHTML = spec.status(st)
    }

    draw()
    if (window.ResizeObserver) new ResizeObserver(() => draw()).observe(fig)
    else window.addEventListener('resize', draw)
  }

  /* ---- the two-token example ------------------------------------------ */

  const logit = (p) => Math.log(p / (1 - p))

  /* Expected gradient on z_A of each estimator used as a loss, averaged over
   * which token is sampled from pi_theta. d log pi(A)/d z_A = 1 - p and
   * d log pi(B)/d z_A = -p. fwd is the exact gradient of KL(pi_theta || pi_ref)
   * and rev that of KL(pi_ref || pi_theta), named as in the blog: forward when
   * pi_theta comes first. */
  const twoToken = {
    fwd: (p, m) => p * (1 - p) * (logit(p) - logit(m)),
    rev: (p, m) => p - m,
    k1: (p) => p * (1 - p) + (1 - p) * -p,
    k2: (p, m) => p * Math.log(p / m) * (1 - p) + (1 - p) * Math.log((1 - p) / (1 - m)) * -p,
    k3: (p, m) => p * (1 - m / p) * (1 - p) + (1 - p) * (1 - (1 - m) / (1 - p)) * -p,
  }

  function niceBound(v) {
    const step = v > 0.5 ? 0.2 : 0.1
    return Math.ceil(v / step - 1e-9) * step
  }

  const FIGS = {}

  FIGS['kl-two-token'] = (fig) => lineChart(fig, {
    title: 'Expected gradient on the logit of A',
    note: 'vocabulary {A, B}, samples from π<sub>θ</sub>',
    state: { m: 0.5 },
    controls: [{ key: 'm', label: 'π<sub>ref</sub>(A)', min: 0.1, max: 0.9, step: 0.05, fmt: (v) => v.toFixed(2) }],
    x: { min: 0.02, max: 0.98, ticks: [0.2, 0.4, 0.6, 0.8], fmt: (v) => v.toFixed(1), grid: true },
    y: (st) => {
      let lo = 0, hi = 0
      for (let i = 0; i <= 96; i++) {
        const p = 0.02 + i * 0.01
        for (const f of [twoToken.fwd, twoToken.rev]) { const v = f(p, st.m); lo = Math.min(lo, v); hi = Math.max(hi, v) }
      }
      const top = niceBound(hi), bot = -niceBound(-lo)
      const step = top - bot > 1 ? 0.4 : 0.2
      const ticks = []
      for (let t = Math.ceil(bot / step - 1e-9) * step; t <= top + 1e-9; t += step) ticks.push(Math.round(t * 10) / 10)
      return { min: bot, max: top, ticks, fmt: (v) => fx(v, 1) }
    },
    series: [
      { key: 'fwd', name: 'forward KL, exact', ink: 'var(--accent)', thick: true, fn: (p, st) => twoToken.fwd(p, st.m) },
      { key: 'k2', name: 'k<sub>2</sub> as a loss', ink: 'var(--accent)', dash: true, fn: (p, st) => twoToken.k2(p, st.m) },
      { key: 'rev', name: 'reverse KL, exact', ink: 'var(--vermilion)', thick: true, fn: (p, st) => twoToken.rev(p, st.m) },
      { key: 'k3', name: 'k<sub>3</sub> as a loss', ink: 'var(--vermilion)', dash: true, fn: (p, st) => twoToken.k3(p, st.m) },
      { key: 'k1', name: 'k<sub>1</sub> as a loss', ink: 'var(--ink-soft)', fn: (p) => twoToken.k1(p) },
    ],
    refs: (st) => [{ v: 0.7, label: st.m === 0.5 ? 'the example in the text' : 'π<sub>θ</sub>(A) = 0.7' }],
    marks: (st) => [
      { key: 'fwd', x: 0.7, y: twoToken.fwd(0.7, st.m), ink: 'var(--accent)' },
      { key: 'rev', x: 0.7, y: twoToken.rev(0.7, st.m), ink: 'var(--vermilion)' },
      { key: 'k1', x: 0.7, y: 0, ink: 'var(--ink-soft)' },
    ],
    xlabel: 'π<sub>θ</sub>(A), the probability the current model gives to A',
    rest: 'hover the plot to read values, or drag the slider',
    readout: (p, st, v, keys) => 'π<sub>θ</sub>(A) ' + p.toFixed(3) + keys.map((k) =>
      ' · ' + { fwd: 'forward KL', k2: 'k2', rev: 'reverse KL', k3: 'k3', k1: 'k1' }[k] + ' ' + fx(v[k], 3)).join(''),
  })

  FIGS['kl-coef'] = (fig) => lineChart(fig, {
    title: 'Coefficient on the score of one sample',
    note: 'each estimator used as a loss',
    x: { min: -3, max: 3, ticks: [-3, -2, -1, 0, 1, 2, 3], fmt: (v) => minus(String(v)), grid: true },
    y: { min: -7, max: 3, ticks: [-6, -4, -2, 0, 2], fmt: (v) => minus(String(v)) },
    series: [
      { key: 'k1', name: 'k<sub>1</sub>: 1', ink: 'var(--ink-soft)', fn: () => 1 },
      { key: 'k2', name: 'k<sub>2</sub>: ρ', ink: 'var(--accent)', fn: (r) => r },
      { key: 'k3', name: 'k<sub>3</sub>: 1 − e<sup>−ρ</sup>', ink: 'var(--vermilion)', fn: (r) => 1 - Math.exp(-r) },
    ],
    xlabel: 'ρ = log π<sub>θ</sub>(y) − log π<sub>ref</sub>(y)',
    readout: (r, st, v, keys) => 'ρ ' + fx(r, 2) + keys.map((k) => ' · ' + k + ' ' + fx(v[k], 3)).join(''),
  })

  /* ---- the three-token example ---------------------------------------- */

  FIGS['kl-three-token'] = (fig) => {
    const q = [0.6, 0.3, 0.1], p = [0.5, 0.3, 0.2]
    const lr = p.map((pi, i) => Math.log(pi / q[i]))
    const eq = q.reduce((s, qi, i) => s + qi * lr[i], 0)
    const K = p.reduce((s, pi, i) => s + pi * lr[i], 0)
    const grad = {
      qp: p.map((pi, i) => pi - q[i]),
      k4: q.map((qi, i) => qi * (lr[i] - eq)),
      pq: p.map((pi, i) => pi * (lr[i] - K)),
    }
    const keys = ['qp', 'k4', 'pq']
    const SPAN = 0.15
    const hidden = new Set()
    fig.textContent = ''
    fig.classList.add('kfig')
    title(fig, 'Expected gradient on each logit', 'π<sub>generator</sub> = (0.6, 0.3, 0.1), π<sub>θ</sub> = (0.5, 0.3, 0.2)')
    const head = el('div', 'kfig-head')
    const readout = el('div', 'kfig-readout')
    readout.setAttribute('aria-live', 'polite')
    const REST = 'hover a bar to read it; a positive gradient lowers that logit under gradient descent'
    readout.textContent = REST
    const rows = []
    const sync = () => { for (const r of rows) r.node.classList.toggle('off', hidden.has(r.key)) }
    head.append(legendOf(keys.map((k) => ({ key: k, name: OPTION[k], ink: INK[k], box: true })), hidden, sync), readout)
    const plot = el('div', 'kfig-plot bars')
    const at = (v) => ((v + SPAN) / (2 * SPAN)) * 100
    for (let t = 0; t < 3; t++) {
      const g = el('div', 'kfig-group')
      g.appendChild(el('div', 'kfig-gname', 'token ' + (t + 1) + ' <span>r = ' + (p[t] / q[t]).toFixed(3) + '</span>'))
      const body = el('div')
      body.style.position = 'relative'
      for (const v of [-0.1, -0.05, 0.05, 0.1]) {
        const i = el('i', 'kfig-gridv')
        i.style.left = at(v) + '%'
        body.appendChild(i)
      }
      const zero = el('i', 'kfig-zero')
      zero.style.left = at(0) + '%'
      body.appendChild(zero)
      for (const k of keys) {
        const v = grad[k][t]
        const row = el('div', 'kfig-row')
        const bar = el('div', 'kfig-bar')
        bar.style.setProperty('--sw', INK[k])
        const a = at(Math.min(0, v)), b = at(Math.max(0, v))
        bar.style.left = a + '%'
        bar.style.width = Math.max(0.25, b - a) + '%'
        const val = el('div', 'kfig-val', (v > 0 ? '+' : '') + fx(v, 3))
        if (v >= 0) val.style.left = 'calc(' + b + '% + 2px)'
        else val.style.right = 'calc(' + (100 - a) + '% + 2px)'
        row.append(bar, val)
        row.addEventListener('pointerenter', () => {
          readout.innerHTML = OPTION[k] + ' · token ' + (t + 1) + ' · ' + (v > 0 ? '+' : '') + fx(v, 4) +
            (Math.abs(v) < 5e-4 ? ' · this logit does not move' : v > 0 ? ' · gradient descent lowers it' : ' · gradient descent raises it')
        })
        row.addEventListener('pointerleave', () => { readout.textContent = REST })
        body.appendChild(row)
        rows.push({ key: k, node: row })
      }
      g.appendChild(body)
      plot.appendChild(g)
    }
    const axis = el('div', 'kfig-axis')
    axis.appendChild(el('b'))
    for (const v of [-0.1, 0, 0.1]) {
      const s = el('span', null, fx(v, 1))
      s.style.left = at(v) + '%'
      axis.appendChild(s)
    }
    plot.appendChild(axis)
    fig.append(head, plot)
  }

  /* ---- the pull on a token that is being pushed down ------------------ */

  /* Where -r log r reaches s on the branch r in (1/e, 1), by bisection. */
  function upperBranch(s) {
    let lo = 1 / Math.E, hi = 1
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2
      if (-mid * Math.log(mid) > s) lo = mid; else hi = mid
    }
    return (lo + hi) / 2
  }

  FIGS['kl-pull'] = (fig) => lineChart(fig, {
    title: 'Pull back on a token whose probability is pushed down',
    note: 'in units of β · π<sub>generator</sub>(v)',
    state: { s: 1.5 },
    controls: [{ key: 's', label: 'push |ΔA<sub>v</sub>| / β', min: 0.1, max: 4, step: 0.05, fmt: (v) => v.toFixed(2) }],
    x: { min: 0.001, max: 1, log: true, grid: true,
      ticks: [{ v: 0.001, l: '0.001' }, { v: 0.01, l: '0.01' }, { v: 0.1, l: '0.1' }, { v: 1, l: '1' }] },
    y: { min: 0, max: 5, ticks: [0, 1, 2, 3, 4, 5] },
    series: [
      { key: 'qp', name: OPTION.qp + ': 1 − r', ink: INK.qp, fn: (r) => 1 - r },
      { key: 'k4', name: OPTION.k4 + ': −log r', ink: INK.k4, fn: (r) => -Math.log(r) },
      { key: 'pq', name: OPTION.pq + ': −r log r', ink: INK.pq, fn: (r) => -r * Math.log(r) },
    ],
    keys: [{ key: 'push', name: 'push', ink: 'var(--ink-soft)', dash: true, ref: true }],
    refs: (st) => [{ h: st.s, ink: 'var(--ink-soft)' }],
    marks: (st) => {
      const out = []
      if (st.s < 1) out.push({ key: 'qp', x: 1 - st.s, y: st.s, ink: INK.qp, fill: true })
      out.push({ key: 'k4', x: Math.exp(-st.s), y: st.s, ink: INK.k4, fill: true })
      if (st.s <= 1 / Math.E) out.push({ key: 'pq', x: upperBranch(st.s), y: st.s, ink: INK.pq, fill: true })
      return out
    },
    xlabel: 'r = π<sub>θ</sub>(v) / π<sub>generator</sub>(v); the token moves left as it is pushed down',
    rest: 'hover the plot to read values; drag the slider to change the push',
    readout: (r, st, v, keys) => 'r ' + (r < 0.01 ? r.toFixed(4) : r.toFixed(3)) +
      keys.map((k) => ' · ' + OPTION[k] + ' ' + v[k].toFixed(3)).join('') + ' · push ' + st.s.toFixed(2),
    status: (st) => {
      const s = st.s
      const qp = s < 1 ? 'stops at r = ' + (1 - s).toFixed(2) : 'cannot hold it, the token goes to 0'
      const k4 = 'stops at r = ' + (Math.exp(-s) < 0.01 ? Math.exp(-s).toExponential(1) : Math.exp(-s).toFixed(2))
      const pq = s <= 1 / Math.E ? 'stops at r = ' + upperBranch(s).toFixed(2) : 'cannot hold it, the token goes to 0'
      return '<b>' + OPTION.qp + '</b> ' + qp + ' · <b>' + OPTION.k4 + '</b> ' + k4 + ' · <b>' + OPTION.pq + '</b> ' + pq
    },
  })

  /* ---- fixed points of the two examples ------------------------------- */

  FIGS['kl-fixed'] = (fig) => {
    const BETAS = FIXED[0].rows.map((r) => r.beta)
    const st = { ex: 0, beta: 1 }
    fig.textContent = ''
    fig.classList.add('kfig')
    title(fig, 'Where the expected updates stop', 'starting from π<sub>θ</sub> = π<sub>generator</sub>')

    const head = el('div', 'kfig-head')
    const legend = el('ul', 'kfig-legend')
    const readout = el('div', 'kfig-readout')
    readout.setAttribute('aria-live', 'polite')
    const REST = 'hover a segment to read it'
    readout.textContent = REST
    head.append(legend, readout)

    const ctl = controls([
      { type: 'segment', key: 'ex', label: 'example', options: ['example 1', 'example 2'] },
      { key: 'beta', label: 'β', values: BETAS, fmt: (v) => String(v) },
    ], st, () => {
      // Each example opens at the β the text discusses.
      if (st.ex !== lastEx) {
        lastEx = st.ex
        st.beta = st.ex === 0 ? 1 : 0.05
        slider.value = BETAS.indexOf(st.beta)
        slider.dispatchEvent(new Event('input'))
        return
      }
      draw()
    })
    let lastEx = st.ex
    const slider = ctl.querySelector('input[type=range]')
    const note = el('div', 'kfig-note')
    const plot = el('div', 'kfig-plot')
    fig.append(head, ctl, note, plot)

    const rowsOf = (ex, row) => [
      { key: 'start', name: 'π<sub>generator</sub>, the starting point', p: ex.q },
      { key: 'qp', name: OPTION.qp, p: row.qp },
      { key: 'k4', name: OPTION.k4, p: row.k4 },
      { key: 'pq', name: OPTION.pq, p: row.pq },
    ]
    const num = (v) => (v === 0 ? '0' : v >= 0.001 ? v.toFixed(3) : sci(v, 1))

    function draw() {
      const ex = FIXED[st.ex]
      const row = ex.rows.find((r) => r.beta === st.beta)
      const eq = ex.q.reduce((s, qi, i) => s + qi * ex.A[i], 0)
      legend.textContent = ''
      ex.A.forEach((a, i) => {
        const li = el('li')
        li.setAttribute('aria-disabled', 'true')
        const sw = el('span', 'kfig-sw box')
        sw.style.setProperty('--sw', INK['t' + (i + 1)])
        li.append(sw, el('span', null, 'token ' + (i + 1) + ', A = ' + minus(String(a))))
        legend.appendChild(li)
      })
      note.innerHTML = 'A = (' + ex.A.map((a) => minus(String(a))).join(', ') + '), ΔA = A − E<sub>generator</sub>[A] = (' +
        ex.A.map((a) => fx(a - eq, 3)).join(', ') + ')'
      plot.textContent = ''
      for (const r of rowsOf(ex, row)) {
        const box = el('div', 'kfig-stack')
        const h = el('div', 'kfig-shead')
        h.append(el('b', null, r.name), el('span', null, r.p.map(num).join(' · ')))
        const bar = el('div', 'kfig-sbar')
        r.p.forEach((v, i) => {
          const seg = el('i')
          seg.style.setProperty('--sw', INK['t' + (i + 1)])
          seg.style.width = v * 100 + '%'
          seg.addEventListener('pointerenter', () => {
            const ratio = v / ex.q[i]
            readout.innerHTML = r.name.replace(', the starting point', '') + ' · token ' + (i + 1) + ' · ' + num(v) +
              (r.key === 'start' ? '' : ' · ' + (ratio === 0 ? 'pushed to 0' : (ratio >= 10 ? ratio.toFixed(1) : ratio.toFixed(2)) + ' times its generator probability'))
          })
          seg.addEventListener('pointerleave', () => { readout.textContent = REST })
          bar.appendChild(seg)
        })
        box.append(h, bar)
        plot.appendChild(box)
      }
    }
    draw()
  }

  /* ---- finite precision ----------------------------------------------- */

  const f32 = new Float32Array(1)
  const u32 = new Uint32Array(f32.buffer)
  /* Round to the nearest BF16 value, ties to even. */
  function bf16(x) {
    f32[0] = x
    const b = u32[0]
    u32[0] = (b + 0x7fff + ((b >>> 16) & 1)) & 0xffff0000
    return f32[0]
  }
  const fr = Math.fround
  const k3True = (d) => Math.expm1(d) - d
  const k3Naive = (d) => { const x = fr(d); return fr(fr(fr(Math.exp(x)) - 1) - x) }
  const k3Expm1 = (d) => { const x = fr(d); return fr(fr(Math.expm1(x)) - x) }
  const k3Series = (d) => { const x = fr(d); return fr(fr(fr(x * x) * 0.5) * fr(1 + fr(x / 3))) }

  FIGS['kl-fp32'] = (fig) => lineChart(fig, {
    title: 'k<sub>3</sub> = e<sup>ρ</sup> − 1 − ρ computed in FP32',
    note: 'size of the true value and of each error, log scales',
    x: { min: 1e-8, max: 1e-1, log: true, grid: true,
      ticks: [-8, -7, -6, -5, -4, -3, -2, -1].map((k) => ({ v: Math.pow(10, k), l: pow10(k) })) },
    y: { min: 1e-18, max: 1e-1, log: true,
      ticks: [-18, -15, -12, -9, -6, -3].map((k) => ({ v: Math.pow(10, k), l: pow10(k) })) },
    samples: 700,
    series: [
      { key: 'true', name: 'true value', ink: 'var(--ink-soft)', fn: (d) => k3True(d) },
      { key: 'naive', name: 'error of exp(ρ) − 1 − ρ', ink: INK.pq, fn: (d) => Math.abs(k3Naive(d) - k3True(d)) },
      { key: 'expm1', name: 'error of expm1(ρ) − ρ', ink: INK.qp, fn: (d) => Math.abs(k3Expm1(d) - k3True(d)) },
      { key: 'series', name: 'error of ρ<sup>2</sup>/2 + ρ<sup>3</sup>/6', ink: INK.k4, fn: (d) => Math.abs(k3Series(d) - k3True(d)) },
    ],
    xlabel: 'ρ, the difference of two log probabilities',
    rest: 'hover the plot to read the computed values',
    readout: (d) => 'ρ ' + sci(d, 2) + ' · true ' + sci(k3True(d), 2) + ' · exp(ρ) − 1 − ρ gives ' +
      minus(sci(k3Naive(d), 2)) + ' · expm1(ρ) − ρ gives ' + minus(sci(k3Expm1(d), 2)) + ' · the series gives ' + sci(k3Series(d), 2),
  })

  FIGS['kl-bf16'] = (fig) => lineChart(fig, {
    title: 'The coefficient r − 1 when r is stored in BF16',
    x: { min: -0.012, max: 0.012, grid: true,
      ticks: [-0.01, -0.005, 0, 0.005, 0.01], fmt: (v) => (v === 0 ? '0' : fx(v, 3)) },
    y: { min: -0.014, max: 0.018, ticks: [-0.01, -0.005, 0, 0.005, 0.01, 0.015], fmt: (v) => (v === 0 ? '0' : fx(v, 3)) },
    samples: 1600,
    series: [
      { key: 'exact', name: 'exact, e<sup>ρ</sup> − 1', ink: 'var(--ink-soft)', thick: true, fn: (d) => Math.expm1(d) },
      { key: 'naive', name: 'r stored in BF16, then minus 1', ink: INK.pq, fn: (d) => bf16(Math.exp(d)) - 1 },
      { key: 'fixed', name: 'expm1(ρ) in FP32, then stored in BF16', ink: INK.qp, dash: true, fn: (d) => bf16(fr(Math.expm1(fr(d)))) },
    ],
    refs: () => [{ v: 0.0039, label: '0.0039' }, { v: -0.00195, label: '−0.002' }],
    xlabel: 'ρ = log π<sub>θ</sub>(y) − log π<sub>generator</sub>(y)',
    readout: (d) => 'ρ ' + fx(d, 4) + ' · exact ' + fx(Math.expm1(d), 5) + ' · BF16 r, then −1 ' +
      fx(bf16(Math.exp(d)) - 1, 5) + ' · expm1 then BF16 ' + fx(bf16(fr(Math.expm1(fr(d)))), 5),
  })

  const init = () => {
    const mounts = document.querySelectorAll('figure[data-chart]')
    let styled = false
    for (const f of mounts) {
      const fn = FIGS[f.dataset.chart]
      if (!fn) continue
      if (!styled) { style(); styled = true }
      fn(f)
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init)
  else init()
})()
