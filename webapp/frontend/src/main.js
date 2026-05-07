import { initThreeJsViewer, initUnifiedBrainViewer } from './viewer3d.js';

function initAll() {
  // Check if each section is open before initializing
  var threeCt = document.getElementById('viewer3d-three-ct');
  var threeSeg = document.getElementById('viewer3d-three-seg');
  var unified = document.getElementById('viewer3d-unified');
  var threeSection = threeCt ? threeCt.closest('.collapsible-section') : null;
  var unifiedSection = unified ? unified.closest('.collapsible-section') : null;

  // Only init if sections are open or don't have collapsible wrapper
  if (!threeSection || threeSection.classList.contains('open')) {
    initThreeJsViewer();
  }
  if (!unifiedSection || unifiedSection.classList.contains('open')) {
    initUnifiedBrainViewer();
  }

  // Setup lazy init on section expand
  document.querySelectorAll('.collapsible-header').forEach(function(header) {
    header.addEventListener('click', function() {
      var targetId = this.getAttribute('data-target');
      var content = document.getElementById(targetId);
      if (!content) return;
      setTimeout(function() {
        var section = content.closest('.collapsible-section');
        if (!section || !section.classList.contains('open')) return;
        if (targetId === 'content-threejs-mesh' && !window._threejsMeshInit) {
          window._threejsMeshInit = true;
          initThreeJsViewer();
        }
        if (targetId === 'content-brain-unified' && !window._brainUnifiedInit) {
          window._brainUnifiedInit = true;
          initUnifiedBrainViewer();
        }
      }, 300);
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAll);
} else {
  initAll();
}
