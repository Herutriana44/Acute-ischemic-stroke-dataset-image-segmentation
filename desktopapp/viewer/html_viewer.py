"""Generate HTML pages for 3D visualization in QWebEngineView.

Reuses the same JavaScript libraries (vtk.js, Papaya, Three.js, Plotly)
as the webapp for consistent 3D visualization.
Uses simple placeholder replacement to avoid Python format-string conflicts.
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# HTML template — uses {{PLACEHOLDER}} to avoid clash with JS curly braces
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stroke Segmentation Result — [[RUN_ID]]</title>
<style>
  * { box-sizing: border-box; margin:0; padding:0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; }
  .container { max-width: 1400px; margin: 0 auto; padding: 12px; }
  .card { background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  h2 { color: #e94560; margin-bottom: 10px; font-size: 1.1em; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
  .metric { background: #0f3460; padding: 8px 12px; border-radius: 6px; }
  .metric strong { display: block; font-size: 0.8em; color: #a0a0a0; }
  .metric span { font-size: 1.1em; }
  .viewer { width: 100%; height: 450px; background: #0a0a1a; border-radius: 6px; position: relative; }
  .vtk-compare { display: flex; gap: 10px; }
  .vtk-compare .vtk-panel { flex: 1; }
  .vtk-panel-title { text-align: center; padding: 6px; color: #e94560; font-weight: bold; }
  .collapsible-header { cursor: pointer; user-select: none; display: flex; justify-content: space-between; align-items: center; }
  .collapsible-header:hover { background: #1a2a4e; border-radius: 6px; padding: 4px; }
  .collapse-icon { transition: transform 0.3s; }
  .open > .collapse-icon { transform: rotate(180deg); }
  .collapsible-content { max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }
  .open > .collapsible-content { max-height: 2000px; }
  button { background: #e94560; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin: 4px; }
  button:hover { background: #c73e54; }
  .papaya-wrap { width: 100%; height: 500px; background: #000; border-radius: 6px; overflow: hidden; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h2>Prediction Result: [[RUN_ID]]</h2>
    <div class="metrics-grid">
      <div class="metric"><strong>Slices</strong><span>[[SLICES]]</span></div>
      <div class="metric"><strong>Resolution</strong><span>[[RES_HW]]</span></div>
      <div class="metric"><strong>Spacing</strong><span>[[SPACING]]</span></div>
      <div class="metric"><strong>Lesion Voxels</strong><span>[[LESION_VOXELS]]</span></div>
      <div class="metric"><strong>Lesion Volume</strong><span>[[LESION_VOLUME]]</span></div>
    </div>
  </div>

  [[THREEJS_SECTION]]

  [[VTK_SECTION]]

  [[PAPAYA_SECTION]]

  <div class="card">
    <a href="javascript:window.close()" style="color:#e94560;">Close Viewer</a>
  </div>
</div>

<script id="mesh-data" type="application/json">[[MESH_JSON]]</script>

<!-- Local Assets -->
<link rel="stylesheet" href="file://[[STATIC_DIR]]/css/papaya.css">
<script src="file://[[STATIC_DIR]]/js/vtk.js"></script>
<script src="file://[[STATIC_DIR]]/js/papaya.js"></script>
<script src="file://[[STATIC_DIR]]/js/plotly.min.js"></script>
<script src="file://[[STATIC_DIR]]/js/three.min.js"></script>
<script src="file://[[STATIC_DIR]]/js/OrbitControls.js"></script>

<script>
// Collapsible sections
(function() {
  var sections = document.querySelectorAll('.collapsible-header');
  sections.forEach(function(header) {
    header.addEventListener('click', function() {
      this.parentElement.classList.toggle('open');
    });
  });
})();

// Three.js mesh viewer
(function() {
  var meshData = JSON.parse(document.getElementById('mesh-data').textContent);

  function createMeshScene(containerId, mesh, color, opacity) {
    if (typeof THREE === 'undefined') {
        console.error("Three.js (THREE) is not defined!");
        return;
    }
    
    var container = document.getElementById(containerId);
    if (!container || !mesh || !mesh.x || mesh.x.length === 0) return;

    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);

    var camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(1, 1, 1);
    scene.add(dirLight);

    var geometry = new THREE.BufferGeometry();
    var positions = [];
    var indices = [];
    for (var i = 0; i < mesh.x.length; i++) {
      positions.push(mesh.x[i], mesh.y[i], mesh.z[i]);
    }
    for (var i = 0; i < mesh.i.length; i++) {
      indices.push(mesh.i[i], mesh.j[i], mesh.k[i]);
    }
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    var material = new THREE.MeshPhongMaterial({
      color: new THREE.Color(color.r/255, color.g/255, color.b/255),
      transparent: true,
      opacity: opacity,
      side: THREE.DoubleSide,
    });

    var threeMesh = new THREE.Mesh(geometry, material);
    scene.add(threeMesh);
    camera.position.z = 50;
    controls.update();

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    new ResizeObserver(function() {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }).observe(container);
  }

  if (meshData.hu_mesh) {
    createMeshScene('three-ct', meshData.hu_mesh, {r:188, g:200, b:218}, 0.8);
  }
  if (meshData.lesion_mesh) {
    createMeshScene('three-lesion', meshData.lesion_mesh, {r:234, g:88, b:12}, 0.9);
  }
})();

// Papaya viewer
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
  var host = document.getElementById("papaya-host");
  
  function start() {
    if (host && window.papaya && window.papaya.Container) {
      host.innerHTML = '<div class="papaya" data-params="papayaParams"></div>';
      window.papaya.Container.startPapaya();
    } else {
      setTimeout(start, 500);
    }
  }
  start();
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_threejs_section(result: dict) -> str:
    if not result.get("enable_3d") or not result.get("hu_mesh"):
        return ""
    return """<section class="card collapsible-section open">
  <div class="collapsible-header">
    <h2>3D Mesh Visualization (Three.js)</h2>
    <span class="collapse-icon">&#9660;</span>
  </div>
  <div class="collapsible-content">
    <p>CT surface (blue-gray) and lesion (orange) rendered with Three.js. Drag to rotate, scroll to zoom.</p>
    <div class="vtk-compare">
      <div class="vtk-panel">
        <div class="vtk-panel-title">CT SURFACE</div>
        <div class="viewer" id="three-ct"></div>
      </div>
      <div class="vtk-panel">
        <div class="vtk-panel-title">LESION</div>
        <div class="viewer" id="three-lesion"></div>
      </div>
    </div>
  </div>
</section>"""


def _build_vtk_section() -> str:
    return """<section class="card collapsible-section">
  <div class="collapsible-header">
    <h2>3D Volume (VTK.js)</h2>
    <span class="collapse-icon">&#9660;</span>
  </div>
  <div class="collapsible-content">
    <p>Volume rendering via VTK.js. Uses hu_volume.npy and mask_pred.npy binary data.</p>
    <div class="vtk-compare">
      <div class="vtk-panel">
        <div class="vtk-panel-title">CT VOLUME</div>
        <div class="viewer" id="vtk-ct"></div>
      </div>
      <div class="vtk-panel">
        <div class="vtk-panel-title">CT + MASK</div>
        <div class="viewer" id="vtk-seg"></div>
      </div>
    </div>
  </div>
</section>"""


def _build_papaya_section(result: dict, run_dir: Path, temp_dir: Path) -> str:
    if not result.get("enable_3d"):
        return ""

    import shutil
    nii_ct = run_dir / result.get("ct_view_nii", "")
    nii_mask = run_dir / result.get("mask_view_nii", "")

    if nii_ct.exists():
        shutil.copy2(nii_ct, temp_dir / nii_ct.name)
    if nii_mask.exists():
        shutil.copy2(nii_mask, temp_dir / nii_mask.name)

    return """<section class="card collapsible-section open">
    <div class="collapsible-header">
    <h2>Papaya Viewer (NIfTI)</h2>
    <span class="collapse-icon">&#9660;</span>
    </div>
    <div class="collapsible-content">
    <p>CT + mask overlay using Papaya DICOM/NIfTI viewer.</p>
    <div class="papaya-wrap" id="papaya-host"></div>
    </div>
    </section>"""


# ---------------------------------------------------------------------------
# Main entry: build HTML and save to temp file
# ---------------------------------------------------------------------------

def build_result_html(run_dir: Path, result: dict, temp_dir: Path) -> Path:
    """Generate the full HTML visualization page and save it to a temp file.

    Returns the Path to the saved HTML file.
    """
    run_id = result.get("run_id", "unknown")
    slices = result.get("slices", "-")
    shape_hw = result.get("shape_hw", [0, 0])
    res_hw = f"{shape_hw[0]} x {shape_hw[1]}" if shape_hw != [0, 0] else "-"
    spacing = result.get("spacing", [0, 0, 0])
    spacing_str = f"({spacing[0]}, {spacing[1]}, {spacing[2]})" if spacing else "-"
    lesion_voxels = result.get("lesion_voxels", 0)
    lesion_volume = f"{result.get('lesion_volume_mm3', 0)} mm³ ({result.get('lesion_volume_ml', 0)} mL)"

    # Mesh JSON for Three.js
    mesh_data = {
        "hu_mesh": result.get("hu_mesh"),
        "lesion_mesh": result.get("lesion_mesh"),
    }
    mesh_json = json.dumps(mesh_data)

    # Build sections
    threejs_section = _build_threejs_section(result)
    vtk_section = _build_vtk_section()
    papaya_section = _build_papaya_section(result, run_dir, temp_dir)

    # File URLs for Papaya
    nii_ct_name = Path(result.get("ct_view_nii", "")).name if result.get("ct_view_nii") else ""
    nii_mask_name = Path(result.get("mask_view_nii", "")).name if result.get("mask_view_nii") else ""
    nii_ct_url = f"file://{temp_dir / nii_ct_name}" if nii_ct_name else ""
    nii_mask_url = f"file://{temp_dir / nii_mask_name}" if nii_mask_name else ""

    # Static dir path for local assets
    static_dir = Path(__file__).parent.parent / "static"
    static_dir_str = static_dir.as_posix()

    # Fill template using simple replacement
    html = _HTML_TEMPLATE
    html = html.replace("[[STATIC_DIR]]", static_dir_str)
    html = html.replace("[[RUN_ID]]", str(run_id))
    html = html.replace("[[SLICES]]", str(slices))
    html = html.replace("[[RES_HW]]", res_hw)
    html = html.replace("[[SPACING]]", spacing_str)
    html = html.replace("[[LESION_VOXELS]]", str(lesion_voxels))
    html = html.replace("[[LESION_VOLUME]]", lesion_volume)
    html = html.replace("[[MESH_JSON]]", mesh_json)
    html = html.replace("[[THREEJS_SECTION]]", threejs_section)
    html = html.replace("[[VTK_SECTION]]", vtk_section)
    html = html.replace("[[PAPAYA_SECTION]]", papaya_section)
    html = html.replace("[[NII_CT]]", nii_ct_url)
    html = html.replace("[[NII_MASK]]", nii_mask_url)

    html_path = temp_dir / "result_viewer.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
