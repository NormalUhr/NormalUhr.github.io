/* The methods figure of the Score Function Centering KL article.
 *
 * The bars are HTML elements rather than SVG. An SVG scaled down to a 390px
 * screen shrinks its own labels with it, and every colour here is one of the
 * page's custom properties, so the figure follows the light and dark themes
 * with no redraw at all. Without JavaScript the <noscript> next to the mount
 * point keeps the PNG.
 */
(() => {
  const SCALE = 0.9 // the axis runs to 0.9, a little above the largest bar

  const METHODS = [
    { key: 'kl', label: 'SFC+KL(β<sub>t</sub>)', ink: 'var(--accent)' },
    { key: 'sc', label: 'SFC', ink: 'var(--ink-faint)' },
    { key: 'tis', label: 'SFC+TIS', ink: 'var(--moss)' },
    { key: 'mis', label: 'SFC+MIS', ink: 'var(--vermilion)' },
  ]

  /* Best held-out evaluation reward of each run, and whether it collapsed.
   * `null` is an arm that was not run at that setting. `start` is the mean
   * evaluation of the same runs before training, the floor of a best
   * evaluation. */
  const SETTINGS = [
    { name: 'Countdown, 4-prompt slices', stale: '46 to 53', start: 0.095,
      runs: { kl: [0.725, false], sc: [0.080, true], tis: [0.389, false], mis: [0.646, true] } },
    { name: 'Countdown, 2-prompt slices', stale: '92 to 116', start: 0.091,
      runs: { kl: [0.670, false], sc: [0.094, true], tis: [0.111, false], mis: [0.068, true] } },
    { name: 'AdamW, 4-prompt slices', stale: '50 to 53', start: 0.082,
      runs: { kl: [0.738, false], sc: [0.715, false], tis: [0.725, true], mis: [0.652, true] } },
    { name: 'GSM8K, 4-prompt slices', stale: '47 to 55', start: 0.389,
      runs: { kl: [0.803, false], sc: [0.393, true], tis: [0.785, false], mis: [0.668, true] } },
    { name: 'Qwen3-4B, 1-prompt slices', stale: '184 to 367', start: 0.414,
      runs: { kl: [0.744, false], sc: [0.412, true], tis: [0.412, true], mis: [0.871, false] } },
    { name: 'Qwen3-30B-A3B, 1-prompt slices', stale: '92 to 137', start: 0.678,
      runs: { kl: [0.877, false], sc: null, tis: [0.680, false], mis: [0.850, false] } },
    { name: 'GLM-4.5-Air, 1-prompt slices', stale: '123 to 225', start: 0.650,
      runs: { kl: [0.750, false], sc: null, tis: [0.641, true], mis: [0.777, false] } },
  ]

  const CSS = `
.mchart { margin: 1.7rem 0; }
.mchart-head { display: flex; flex-direction: column; gap: .3rem; margin-bottom: .75rem; }
.mchart-legend { display: flex; flex-wrap: wrap; gap: .3rem .95rem; list-style: none; margin: 0; padding: 0; }
.mchart-legend li { display: flex; align-items: center; gap: .4em; cursor: pointer;
  font: 500 .78rem/1.4 var(--sans); color: var(--ink-soft); -webkit-user-select: none; user-select: none; }
.mchart-legend li[aria-pressed="false"] { opacity: .35; }
.mchart-legend li:focus-visible { outline: 1px solid var(--accent); outline-offset: 2px; }
.mchart-sw { width: 1.5em; height: .6em; flex: none; border-radius: 1px; background: var(--sw, var(--ink-faint)); }
.mchart-sw.hatch { background-image: repeating-linear-gradient(45deg,
  transparent 0 3px, var(--paper) 3px 5px); }
.mchart-sw.start { position: relative; height: .9em; background: none; border-radius: 0; }
.mchart-sw.start::after { content: ''; position: absolute; left: 50%; top: 0; bottom: 0;
  border-left: 1.5px dashed var(--sw); }
/* The readout keeps one fixed line, two on a phone, whatever it holds: a
 * readout that grew on hover would push the bars away from the cursor, end
 * the hover and shrink again, over and over. */
.mchart-readout { font: .74rem/1.4 var(--mono); color: var(--ink-soft); height: 1.4em;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.mchart-plot { position: relative; padding-right: 3.5em; }
.mchart-rows { position: relative; }
.mchart-grid { position: absolute; inset: 0; pointer-events: none; }
.mchart-grid i { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--rule-soft); }
.mchart-start { position: absolute; top: -3px; bottom: -3px; z-index: 1; width: 0;
  border-left: 1.5px dashed var(--ink-soft); pointer-events: none; }
.mchart-group + .mchart-group { margin-top: .85rem; }
.mchart-name { font: 600 .79rem/1.4 var(--sans); color: var(--ink); margin-bottom: .25rem; }
.mchart-name span { font-weight: 400; color: var(--ink-faint); }
.mchart-row { position: relative; height: 15px; margin: 3px 0; transition: opacity .12s; }
.mchart-row.off { display: none; }
.mchart-plot:hover .mchart-row { opacity: .5; }
.mchart-plot .mchart-row:hover { opacity: 1; }
.mchart-bar { position: absolute; left: 0; top: 0; bottom: 0; min-width: 1px;
  background-color: var(--sw); border-radius: 0 1px 1px 0; }
.mchart-bar.hatch { background-image: repeating-linear-gradient(45deg,
  transparent 0 3px, var(--paper) 3px 5px); }
.mchart-val { position: absolute; top: 50%; transform: translateY(-50%); z-index: 2;
  padding: 2px; background: var(--paper);
  font: .72rem/1 var(--mono); color: var(--ink-soft); white-space: nowrap; }
.mchart-none { position: absolute; top: 50%; left: 2px; transform: translateY(-50%);
  font: italic .72rem/1 var(--sans); color: var(--ink-faint); }
.mchart-axis { position: relative; height: 1.4em; }
.mchart-axis span { position: absolute; top: .25em; font: .7rem/1 var(--mono); color: var(--ink-faint);
  transform: translateX(-50%); }
.mchart-axis b { position: absolute; left: 0; right: 0; top: 0; height: 1px; background: var(--rule); }
.mchart-title { font: 600 .95rem/1.35 var(--sans); color: var(--ink-strong); margin: 0 0 .55rem; }
@media (max-width: 520px) {
  .mchart-readout { height: 2.8em; white-space: normal; display: -webkit-box;
    -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
  .mchart-plot { padding-right: 3.2em; }
}`

  function style() {
    if (document.getElementById('mchart-css')) return
    const el = document.createElement('style')
    el.id = 'mchart-css'
    el.textContent = CSS
    document.head.appendChild(el)
  }

  function build(fig) {
    style()
    const hidden = new Set()
    fig.textContent = ''
    fig.classList.add('mchart')

    const head = document.createElement('div')
    head.className = 'mchart-head'
    const legend = document.createElement('ul')
    legend.className = 'mchart-legend'
    legend.setAttribute('aria-label', 'Show or hide a method')
    const readout = document.createElement('div')
    readout.className = 'mchart-readout'
    readout.setAttribute('aria-live', 'polite')
    const REST = 'hover a bar for its run'
    readout.textContent = REST
    head.append(legend, readout)

    const plot = document.createElement('div')
    plot.className = 'mchart-plot'

    const gridFor = (parent) => {
      const grid = document.createElement('div')
      grid.className = 'mchart-grid'
      for (const at of [0.2, 0.4, 0.6, 0.8]) {
        const line = document.createElement('i')
        line.style.left = (at / SCALE) * 100 + '%'
        grid.appendChild(line)
      }
      parent.appendChild(grid)
    }

    const rows = []
    for (const setting of SETTINGS) {
      const group = document.createElement('div')
      group.className = 'mchart-group'
      const name = document.createElement('div')
      name.className = 'mchart-name'
      name.innerHTML = setting.name + ' <span>staleness ' + setting.stale + '</span>'
      group.appendChild(name)
      const bars = document.createElement('div')
      bars.className = 'mchart-rows'
      gridFor(bars)
      group.appendChild(bars)

      for (const method of METHODS) {
        const row = document.createElement('div')
        row.className = 'mchart-row'
        row.dataset.method = method.key
        row.style.setProperty('--sw', method.ink)
        const run = setting.runs[method.key]
        if (!run) {
          const none = document.createElement('span')
          none.className = 'mchart-none'
          none.textContent = 'not run'
          row.appendChild(none)
        } else {
          const value = run[0]
          const collapsed = run[1]
          const width = (value / SCALE) * 100
          const bar = document.createElement('span')
          bar.className = 'mchart-bar' + (collapsed ? ' hatch' : '')
          bar.style.width = width + '%'
          const label = document.createElement('span')
          label.className = 'mchart-val'
          label.textContent = value.toFixed(3)
          label.style.left = 'calc(' + width + '% + 6px)'
          row.append(bar, label)
          row.addEventListener('pointerenter', () => {
            readout.textContent = method.label.replace(/<[^>]+>/g, '') + ' · ' + setting.name +
              ' · ' + value.toFixed(3) + (collapsed ? ' · collapsed' : '') +
              ' · before training ' + setting.start.toFixed(2)
          })
        }
        rows.push(row)
        bars.appendChild(row)
      }
      const start = document.createElement('i')
      start.className = 'mchart-start'
      start.style.left = (setting.start / SCALE) * 100 + '%'
      bars.appendChild(start)
      plot.appendChild(group)
    }
    plot.addEventListener('pointerleave', () => { readout.textContent = REST })

    const axis = document.createElement('div')
    axis.className = 'mchart-axis'
    axis.appendChild(document.createElement('b'))
    for (const at of [0, 0.2, 0.4, 0.6, 0.8]) {
      const tick = document.createElement('span')
      tick.textContent = at.toFixed(1)
      tick.style.left = (at / SCALE) * 100 + '%'
      if (at === 0) tick.style.transform = 'none'
      axis.appendChild(tick)
    }
    plot.appendChild(axis)

    const title = document.createElement('div')
    title.className = 'mchart-title'
    title.textContent = 'Best held-out evaluation reward'

    for (const method of METHODS) {
      const item = document.createElement('li')
      item.dataset.method = method.key
      item.setAttribute('role', 'button')
      item.setAttribute('tabindex', '0')
      item.setAttribute('aria-pressed', 'true')
      item.innerHTML = '<span class="mchart-sw" style="--sw:' + method.ink + '"></span><span>' +
        method.label + '</span>'
      const sync = () => {
        const off = hidden.has(method.key)
        item.setAttribute('aria-pressed', off ? 'false' : 'true')
        for (const row of rows) {
          if (row.dataset.method === method.key) row.classList.toggle('off', off)
        }
      }
      const toggle = () => {
        // Hiding the last visible method would leave an empty figure.
        if (!hidden.has(method.key) && hidden.size >= METHODS.length - 1) return
        if (hidden.has(method.key)) hidden.delete(method.key)
        else hidden.add(method.key)
        sync()
      }
      item.addEventListener('click', toggle)
      item.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return
        event.preventDefault()
        toggle()
      })
      legend.appendChild(item)
    }
    const collapsed = document.createElement('li')
    collapsed.setAttribute('aria-disabled', 'true')
    collapsed.style.cursor = 'default'
    collapsed.innerHTML = '<span class="mchart-sw hatch" style="--sw:var(--ink-faint)"></span><span>collapsed</span>'
    legend.appendChild(collapsed)
    const before = document.createElement('li')
    before.setAttribute('aria-disabled', 'true')
    before.style.cursor = 'default'
    before.innerHTML = '<span class="mchart-sw start" style="--sw:var(--ink-soft)"></span><span>before training</span>'
    legend.appendChild(before)

    fig.append(title, head, plot)
  }

  document.querySelectorAll('[data-chart="sfc-methods"]').forEach(build)
})()
