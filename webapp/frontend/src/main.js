import { initThreeJsViewer } from './viewer3d.js';

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initThreeJsViewer);
} else {
  initThreeJsViewer();
}
