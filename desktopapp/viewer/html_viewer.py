"""Generate HTML pages for 3D visualization in QWebEngineView.

Reuses the same JavaScript libraries (Papaya, Three.js, Plotly)
as the webapp for consistent 3D visualization.

Key fix: all static assets (JS/CSS) are copied into the same temp directory
as the HTML file so that QWebEngineView's file:// security policy allows
loading them (same-origin / same-directory).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Static assets to copy into temp dir alongside the HTML
# ---------------------------------------------------------------------------
_STATIC_JS = ["three.min.js", "OrbitControls.js", "papaya.js", "plotly.min.js"]
_STATIC_CSS = ["papaya.css"]


def _copy_static_assets(temp_dir: Path) -> None:
    """Copy JS/CSS assets from desktopapp/static into temp_dir."""
    static_dir = Path(__file__).parent.parent / "static"
    js_src = static_dir / "js"
    css_src = static_dir / "css"
    for fname in _STATIC_JS:
        src = js_src / fname
        if src.exists():
            shutil.copy2(src, temp_dir / fname)
    for fname in _STATIC_CSS:
        src = css_src / fname
        if src.exists():
            shutil.copy2(src, temp_dir / fname)


# ---------------------------------------------------------------------------
# HTML template — uses [[PLACEHOLDER]] to avoid clash with JS curly braces
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stroke Segmentation Result — [[RUN_ID]]</title>
<!-- All assets are in the same directory as this HTML file -->
<link rel="stylesheet" href="papaya.css">
<style>
  * { box-sizing: border-box; margin:0; padding:0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .container { max-width: 1400px; margin: 0 auto; padding: 12px; }
  .card { background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  h2 { color: #e94560; margin-bottom: 10px; font-size: 1.1em; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
  .metric { background: #0f3460; padding: 8px 12px; border-radius: 6px; }
  .metric strong { display: block; font-size: 0.8em; color: #a0a0a0; }
  .metric span { font-size: 1.1em; }
  .viewer { width: 100%; height: 450px; background: #0a0a1a; border-radius: 6px;
            position: relative; overflow: hidden; }
  .viewer-error { display: flex; align-items: center; justify-content: center;
                  height: 100%; color: #e94560; font-size: 0.9em; }
  .vtk-compare { display: flex; gap: 10px; }
  .vtk-compare .vtk-panel { flex: 1; min-width: 0; }
  .vtk-panel-title { text-align: center; padding: 6px; color: #e94560; font-weight: bold; }
  .collapsible-section { }
  .collapsible-header { cursor: pointer; user-select: none; display: flex;
                        justify-content: space-between; align-items: center; }
  .collapsible-header:hover { background: #1a2a4e; border-radius: 6px; padding: 4px; }
  .collapse-icon { transition: transform 0.3s; }
  .collapsible-section.open .collapse-icon { transform: rotate(180deg); }
  .collapsible-content { max-height: 0; overflow: hidden; transition: max-height 0.4s ease; }
  .collapsible-section.open .collapsible-content { max-height: 2000px; }
  button { background: #e94560; color: white; border: none; padding: 8px 16px;
           border-radius: 6px; cursor: pointer; margin: 4px; }
  button:hover { background: #c73e54; }
  .papaya-wrap { width: 100%; height: 500px; background: #000; border-radius: 6px; overflow: hidden; }
</style>
</head>
<body>
<div class="container">

  <!-- Metrics card -->
  <div class="card">
    <h2>Prediction Result: [[RUN_ID]]</h2>
    <div class="metrics-grid">
      <div class="metric"><strong>Slices</strong><span>[[SLICES]]</span></div>
      <div class="metric"><strong>Resolution</strong><span>[[RES_HW]]</span></div>
      <div class="metric"><strong>Spacing (mm)</strong><span>[[SPACING]]</span></div>
      <div class="metric"><strong>Lesion Voxels</strong><span>[[LESION_VOXELS]]</span></div>
      <div class="metric"><strong>Lesion Volume</strong><span>[[LESION_VOLUME]]</span></div>
    </div>
  </div>

  [[THREEJS_SECTION]]

  [[PAPAYA_SECTION]]

  <div class="card" style="text-align:right;">
    <a href="javascript:void(0)" onclick="window.history.back()" style="color:#e94560;">&#8592; Back</a>
  </div>

</div>

<!-- Mesh data embedded as JSON -->
<script id="mesh-data" type="application/json">[[MESH_JSON]]</script>

<!-- Local JS assets (same directory as HTML) -->
<script src="three.min.js"></script>
<script src="OrbitControls.js"></script>
<script src="papaya.js"></script>

<script>
// ── Collapsible sections ──────────────────────────────────────────────────
document.querySelectorAll('.collapsible-header').forEach(function(header) {
  header.addEventListener('click', function() {
    this.closest('.collapsible-section').classList.toggle('open');
  });
});

// ── Three.js mesh viewer ──────────────────────────────────────────────────
(function() {
  var meshDataEl = document.getElementById('mesh-data');
  if (!meshDataEl) return;
  var meshData;
  try { meshData = JSON.parse(meshDataEl.textContent); }
  catch(e) { console.error('mesh-data parse error:', e); return; }

  function showError(containerId, msg) {
    var el = document.getElementById(containerId);
    if (el) el.innerHTML = '<div class="viewer-error">' + msg + '</div>';
  }

  function createMeshScene(containerId, mesh, colorHex, opacity) {
    if (typeof THREE === 'undefined') {
      showError(containerId, 'Three.js failed to load');
      console.error('Three.js (THREE) is not defined!');
      return;
    }
    var container = document.getElementById(containerId);
    if (!container) return;
    if (!mesh || !mesh.x || mesh.x.length === 0) {
      showError(containerId, 'No mesh data');
      return;
    }

    var w = container.clientWidth || 400;
    var h = container.clientHeight || 450;

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);

    var camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 5000);
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h);
    container.appendChild(renderer.domElement);

    // OrbitControls — attached as THREE.OrbitControls by OrbitControls.js
    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    var dir1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dir1.position.set(1, 2, 3);
    scene.add(dir1);
    var dir2 = new THREE.DirectionalLight(0x8888ff, 0.3);
    dir2.position.set(-1, -1, -1);
    scene.add(dir2);

    // Build geometry
    var geometry = new THREE.BufferGeometry();
    var positions = new Float32Array(mesh.x.length * 3);
    for (var i = 0; i < mesh.x.length; i++) {
      positions[i * 3]     = mesh.x[i];
      positions[i * 3 + 1] = mesh.y[i];
      positions[i * 3 + 2] = mesh.z[i];
    }
    var indices = new Uint32Array(mesh.i.length * 3);
    for (var i = 0; i < mesh.i.length; i++) {
      indices[i * 3]     = mesh.i[i];
      indices[i * 3 + 1] = mesh.j[i];
      indices[i * 3 + 2] = mesh.k[i];
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    // Center geometry
    geometry.computeBoundingBox();
    var center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    var material = new THREE.MeshPhongMaterial({
      color: colorHex,
      transparent: opacity < 1.0,
      opacity: opacity,
      side: THREE.DoubleSide,
      shininess: 60,
    });

    var mesh3 = new THREE.Mesh(geometry, material);
    scene.add(mesh3);

    // Position camera based on bounding sphere
    geometry.computeBoundingSphere();
    var radius = geometry.boundingSphere ? geometry.boundingSphere.radius : 50;
    camera.position.set(0, 0, radius * 2.5);
    camera.near = radius * 0.01;
    camera.far = radius * 20;
    camera.updateProjectionMatrix();
    controls.update();

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    // Responsive resize
    var ro = new ResizeObserver(function() {
      var nw = container.clientWidth;
      var nh = container.clientHeight;
      if (nw > 0 && nh > 0) {
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
        renderer.setSize(nw, nh);
      }
    });
    ro.observe(container);
  }

  if (meshData.hu_mesh) {
    createMeshScene('three-ct', meshData.hu_mesh, 0xbcc8da, 0.75);
  }
  if (meshData.lesion_mesh) {
    createMeshScene('three-lesion', meshData.lesion_mesh, 0xea580c, 0.92);
  }
})();

// ── Papaya NIfTI viewer ───────────────────────────────────────────────────
(function() {
  var ctUrl = "[[NII_CT]]";
  var maskUrl = "[[NII_MASK]]";
  if (!ctUrl || !maskUrl) return;

  var params = [];
  params["worldSpace"] = false;
  params["showOrientation"] = true;
  params["smoothDisplay"] = false;
  params["interpolation"] = "none";
  params["images"] = [ctUrl, maskUrl];
  params[ctUrl] = { min: 0, max: 255 };
  params[maskUrl] = { min: 0.5, max: 255, alpha: 0.68, lut: "Overlay (Positives)" };
  window.papayaParams = params;

  function startPapaya() {
    var host = document.getElementById("papaya-host");
    if (host && window.papaya && window.papaya.Container) {
      host.innerHTML = '<div class="papaya" data-params="papayaParams"></div>';
      window.papaya.Container.startPapaya();
    } else {
      setTimeout(startPapaya, 400);
    }
  }
  startPapaya();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_threejs_section(result: dict) -> str:
    """Build the Three.js 3D mesh section. Shows CT surface + lesion panels."""
    if not result.get("enable_3d") or not result.get("hu_mesh"):
        return ""
    has_lesion = bool(result.get("lesion_mesh"))
    lesion_panel = ""
    if has_lesion:
        lesion_panel = """
      <div class="vtk-panel">
        <div class="vtk-panel-title">LESION MESH</div>
        <div class="viewer" id="three-lesion"></div>
      </div>"""
    return (
        '<section class="card collapsible-section open">\n'
        '  <div class="collapsible-header">\n'
        '    <h2>3D Mesh Visualization (Three.js)</h2>\n'
        '    <span class="collapse-icon">&#9660;</span>\n'
        '  </div>\n'
        '  <div class="collapsible-content">\n'
        '    <p style="margin-bottom:8px;font-size:0.85em;color:#a0a0a0;">'
        'CT brain surface (blue-gray) and ischemic lesion (orange). '
        'Drag to rotate &bull; scroll to zoom &bull; right-drag to pan.</p>\n'
        '    <div class="vtk-compare">\n'
        '      <div class="vtk-panel">\n'
        '        <div class="vtk-panel-title">CT BRAIN SURFACE</div>\n'
        '        <div class="viewer" id="three-ct"></div>\n'
        '      </div>'
        + lesion_panel +
        '\n    </div>\n'
        '  </div>\n'
        '</section>'
    )


def _build_papaya_section(result: dict, run_dir: Path, temp_dir: Path) -> str:
    """Build the Papaya NIfTI viewer section and copy NIfTI files to temp_dir."""
    if not result.get("enable_3d"):
        return ""

    nii_ct_name = result.get("ct_view_nii", "")
    nii_mask_name = result.get("mask_view_nii", "")
    nii_ct = run_dir / nii_ct_name if nii_ct_name else None
    nii_mask = run_dir / nii_mask_name if nii_mask_name else None

    if nii_ct and nii_ct.exists():
        shutil.copy2(nii_ct, temp_dir / nii_ct.name)
    if nii_mask and nii_mask.exists():
        shutil.copy2(nii_mask, temp_dir / nii_mask.name)

    return """<section class="card collapsible-section open">
  <div class="collapsible-header">
    <h2>Papaya Viewer (NIfTI CT + Mask Overlay)</h2>
    <span class="collapse-icon">&#9660;</span>
  </div>
  <div class="collapsible-content">
    <p style="margin-bottom:8px;font-size:0.85em;color:#a0a0a0;">
      Axial / coronal / sagittal slices with lesion mask overlay (orange).
    </p>
    <div class="papaya-wrap" id="papaya-host"></div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Main entry: build HTML and save to temp file
# ---------------------------------------------------------------------------

def build_result_html(run_dir: Path, result: dict, temp_dir: Path) -> Path:
    """Generate the full HTML visualization page and save it to temp_dir.

    All static assets (JS/CSS) are copied into temp_dir so that
    QWebEngineView's file:// same-origin policy allows loading them.

    Returns the Path to the saved HTML file.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy static assets into temp dir (same-origin as the HTML)
    _copy_static_assets(temp_dir)

    # Also copy NIfTI files for Papaya
    nii_ct_name = result.get("ct_view_nii", "")
    nii_mask_name = result.get("mask_view_nii", "")

    run_id = result.get("run_id", "unknown")
    slices = result.get("slices", "-")
    shape_hw = result.get("shape_hw", [0, 0])
    res_hw = f"{shape_hw[0]} × {shape_hw[1]}" if shape_hw and shape_hw != [0, 0] else "-"
    spacing = result.get("spacing", [])
    spacing_str = (
        f"{spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f}"
        if spacing and len(spacing) == 3 else "-"
    )
    lesion_voxels = result.get("lesion_voxels", 0)
    lesion_volume = (
        f"{result.get('lesion_volume_mm3', 0):.2f} mm³ "
        f"({result.get('lesion_volume_ml', 0):.4f} mL)"
    )

    # Mesh JSON for Three.js
    mesh_data = {
        "hu_mesh": result.get("hu_mesh"),
        "lesion_mesh": result.get("lesion_mesh"),
    }
    mesh_json = json.dumps(mesh_data)

    # Build sections
    threejs_section = _build_threejs_section(result)
    papaya_section = _build_papaya_section(result, run_dir, temp_dir)

    # NIfTI URLs — relative paths since files are in same temp_dir
    nii_ct_url = nii_ct_name if nii_ct_name else ""
    nii_mask_url = nii_mask_name if nii_mask_name else ""

    # Fill template
    html = _HTML_TEMPLATE
    html = html.replace("[[RUN_ID]]", str(run_id))
    html = html.replace("[[SLICES]]", str(slices))
    html = html.replace("[[RES_HW]]", res_hw)
    html = html.replace("[[SPACING]]", spacing_str)
    html = html.replace("[[LESION_VOXELS]]", str(lesion_voxels))
    html = html.replace("[[LESION_VOLUME]]", lesion_volume)
    html = html.replace("[[MESH_JSON]]", mesh_json)
    html = html.replace("[[THREEJS_SECTION]]", threejs_section)
    html = html.replace("[[PAPAYA_SECTION]]", papaya_section)
    html = html.replace("[[NII_CT]]", nii_ct_url)
    html = html.replace("[[NII_MASK]]", nii_mask_url)

    html_path = temp_dir / "result_viewer.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
