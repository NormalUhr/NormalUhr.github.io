const root = document.getElementById('consistency-diagnostic-map');

if (root) {
  const buttons = Array.from(root.querySelectorAll('[data-map-select]'));
  const panels = Array.from(root.querySelectorAll('[data-map-panel]'));
  const reset = root.querySelector('[data-map-reset]');
  const returnToMap = root.querySelector('[data-map-return]');
  let selectedButton = null;
  const endpoints = {
    inputs: ['train', 'prefill', 'decode'],
    trainer: ['train'],
    prefill: ['prefill'],
    phase: ['prefill', 'decode'],
    state: ['decode'],
    sampling: ['decode'],
    engines: ['train', 'prefill'],
    total: ['train', 'decode'],
    sync: [],
    monitor: ['train', 'prefill', 'decode'],
    overview: [],
  };

  function select(key, showDetail = false) {
    const selected = panels.find(panel => panel.dataset.mapPanel === key);
    if (!selected) return;
    for (const panel of panels) panel.hidden = panel !== selected;
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button.dataset.mapSelect === key));
    }
    for (const endpoint of root.querySelectorAll('[data-map-endpoint]')) {
      endpoint.classList.toggle('cm-highlight', endpoints[key].includes(endpoint.dataset.mapEndpoint));
    }
    for (const system of root.querySelectorAll('[data-map-system]')) {
      system.classList.toggle('cm-highlight', key === 'sync');
    }
    reset.hidden = key === 'overview';
    returnToMap.hidden = key === 'overview';
    root.dataset.mapSelection = key;
    if (showDetail) {
      selected.tabIndex = -1;
      selected.focus({ preventScroll: true });
      selected.scrollIntoView({ block: 'nearest', behavior: 'auto' });
    }
  }

  for (const button of buttons) {
    button.addEventListener('click', () => {
      selectedButton = button;
      select(button.dataset.mapSelect, true);
    });
  }
  returnToMap.addEventListener('click', () => {
    selectedButton.focus({ preventScroll: true });
    selectedButton.scrollIntoView({ block: 'center', behavior: 'auto' });
  });
  reset.addEventListener('click', () => {
    select('overview');
    buttons[0].focus({ preventScroll: true });
  });
  select('overview');
}
