import { initThreeJsViewer, initUnifiedBrainViewer } from './viewer3d.js';

function initAll() {
  initThreeJsViewer();
  initUnifiedBrainViewer();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAll);
} else {
  initAll();
}
