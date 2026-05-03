import "@kitware/vtk.js/Rendering/Profiles/Volume.js";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction.js";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow.js";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume.js";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper.js";

const VTK_MAX = 192;

function el(tag, text, className) {
  const n = document.createElement(tag);
  if (text) n.textContent = text;
  if (className) n.className = className;
  return n;
}

function buildHuVolume(imageData, huArray, nx, ny, nz, spacing) {
  const da = vtkDataArray.newInstance({
    name: "HU",
    numberOfComponents: 1,
    values: huArray,
  });
  imageData.setDimensions(nx, ny, nz);
  imageData.setSpacing(spacing[0], spacing[1], spacing[2]);
  imageData.getPointData().setScalars(da);
}

function buildMaskVolume(imageData, maskArray, nx, ny, nz, spacing) {
  const da = vtkDataArray.newInstance({
    name: "mask",
    numberOfComponents: 1,
    values: maskArray,
  });
  imageData.setDimensions(nx, ny, nz);
  imageData.setSpacing(spacing[0], spacing[1], spacing[2]);
  imageData.getPointData().setScalars(da);
}

function ctVolumeActor(imageData) {
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setBlendModeToComposite();

  const actor = vtkVolume.newInstance();
  actor.setMapper(mapper);

  const ct = vtkColorTransferFunction.newInstance();
  const op = vtkPiecewiseFunction.newInstance();
  const range = imageData.getPointData().getScalars().getRange();
  const lo = range[0];
  const hi = range[1];
  const width = Math.max(hi - lo, 1);
  const c = lo + width * 0.35;
  const w = width * 0.22;
  const wmin = c - w;
  const wmax = c + w;

  ct.addRGBPoint(lo, 0.02, 0.02, 0.05);
  ct.addRGBPoint(wmin, 0.15, 0.18, 0.22);
  ct.addRGBPoint(c, 0.75, 0.78, 0.82);
  ct.addRGBPoint(wmax, 0.95, 0.95, 0.97);
  ct.addRGBPoint(hi, 1.0, 1.0, 0.92);

  op.addPoint(lo, 0.0);
  op.addPoint(wmin, 0.0);
  op.addPoint(c, 0.18);
  op.addPoint(wmax, 0.55);
  op.addPoint(hi, 0.72);

  const prop = actor.getProperty();
  prop.setRGBTransferFunction(0, ct);
  prop.setScalarOpacity(0, op);
  prop.setInterpolationTypeToLinear();
  prop.setShade(true);
  prop.setAmbient(0.35);
  prop.setDiffuse(0.65);
  prop.setSpecular(0.12);
  return actor;
}

function maskVolumeActor(imageData) {
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setBlendModeToComposite();

  const actor = vtkVolume.newInstance();
  actor.setMapper(mapper);

  const ct = vtkColorTransferFunction.newInstance();
  const op = vtkPiecewiseFunction.newInstance();
  ct.addRGBPoint(0, 0, 0, 0);
  ct.addRGBPoint(0.5, 0.92, 0.35, 0.08);
  ct.addRGBPoint(1, 0.96, 0.38, 0.1);

  op.addPoint(0, 0.0);
  op.addPoint(0.01, 0.0);
  op.addPoint(1, 0.52);

  const prop = actor.getProperty();
  prop.setRGBTransferFunction(0, ct);
  prop.setScalarOpacity(0, op);
  prop.setInterpolationTypeToNearest();
  prop.setShade(false);
  return actor;
}

async function fetchBinary(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.arrayBuffer();
}

function mountViewer(container, genericRw) {
  genericRw.setContainer(container);
  genericRw.resize();
  const rw = genericRw.getRenderWindow();
  const renderer = genericRw.getRenderer();
  renderer.setBackground(0.04, 0.05, 0.08);
  rw.render();

  const ro = new ResizeObserver(() => {
    genericRw.resize();
    rw.render();
  });
  ro.observe(container);
}

async function startVtkDicomViewer(opts) {
  const { container, metaUrl, huUrl, maskUrl } = opts;
  if (!container) return;

  container.innerHTML = "";
  const status = el("p", "Memuat volume DICOM + mask untuk VTK…", "vtk-status muted-note");
  container.appendChild(status);

  let genericRw;
  try {
    const metaR = await fetch(metaUrl, { cache: "no-store" });
    if (!metaR.ok) throw new Error(`Meta HTTP ${metaR.status}`);
    const meta = await metaR.json();
    const [nx, ny, nz] = meta.dims_xyz;
    const sp = meta.spacing_xyz_mm;

    const [huBuf, maskBuf] = await Promise.all([fetchBinary(huUrl), fetchBinary(maskUrl)]);
    const hu = new Float32Array(huBuf);
    const mask = new Uint8Array(maskBuf);
    const expected = nx * ny * nz;
    if (hu.length !== expected || mask.length !== expected) {
      throw new Error(`Ukuran buffer tidak cocok: ${hu.length}, ${mask.length} vs ${expected}`);
    }

    container.innerHTML = "";
    const wrap = el("div", "", "vtk-wrap");
    wrap.style.width = "100%";
    wrap.style.height = "100%";
    wrap.style.minHeight = "650px";
    container.appendChild(wrap);

    const huImage = vtkImageData.newInstance();
    buildHuVolume(huImage, hu, nx, ny, nz, sp);

    const maskImage = vtkImageData.newInstance();
    buildMaskVolume(maskImage, mask, nx, ny, nz, sp);

    genericRw = vtkGenericRenderWindow.newInstance({
      background: [0.04, 0.05, 0.08],
      listenWindowResize: true,
    });
    mountViewer(wrap, genericRw);

    const renderer = genericRw.getRenderer();
    const ctActor = ctVolumeActor(huImage);
    const mkActor = maskVolumeActor(maskImage);
    renderer.addVolume(ctActor);
    renderer.addVolume(mkActor);
    renderer.resetCamera();
    genericRw.getRenderWindow().render();

    const hint = el(
      "p",
      "VTK.js: volume rendering HU (grid DICOM) + overlay mask lesi. Putar dengan drag; scroll untuk zoom.",
      "muted-note"
    );
    hint.style.marginTop = "8px";
    container.appendChild(hint);
  } catch (e) {
    console.error("[VTK viewer]", e);
    container.innerHTML = "";
    const msg =
      e && e.message
        ? `Gagal memuat visualisasi VTK: ${e.message}`
        : "Gagal memuat visualisasi VTK. Pastikan inferensi telah menghasilkan hu_volume.npy dan mask_pred.npy.";
    container.appendChild(el("p", msg, "muted-note"));
  }
}

const payload = document.getElementById("mesh-data");
const vtkContainer = document.getElementById("viewer3d-vtk");
if (payload && vtkContainer) {
  const result = JSON.parse(payload.textContent);
  const runId = result.run_id;
  const q = `max=${VTK_MAX}`;
  const base = new URL(window.location.href);
  const metaUrl = new URL(`/runs/${encodeURIComponent(runId)}/vtk_meta?${q}`, base).href;
  const huUrl = new URL(`/runs/${encodeURIComponent(runId)}/vtk_hu.bin?${q}`, base).href;
  const maskUrl = new URL(`/runs/${encodeURIComponent(runId)}/vtk_mask.bin?${q}`, base).href;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      startVtkDicomViewer({ container: vtkContainer, metaUrl, huUrl, maskUrl });
    });
  } else {
    startVtkDicomViewer({ container: vtkContainer, metaUrl, huUrl, maskUrl });
  }
}
