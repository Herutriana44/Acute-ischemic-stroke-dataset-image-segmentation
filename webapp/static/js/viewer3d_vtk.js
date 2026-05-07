// Use absolute CDN imports to avoid importmap incompatibilities (some browsers / embedded viewers).
// NOTE: static `import` requires a string literal (no template strings).
import "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Rendering/Profiles/Volume.js";
import vtkColorTransferFunction from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Rendering/Core/ColorTransferFunction.js";
import vtkDataArray from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Common/Core/DataArray.js";
import vtkImageData from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Common/DataModel/ImageData.js";
import vtkPiecewiseFunction from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Common/DataModel/PiecewiseFunction.js";
import vtkGenericRenderWindow from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Rendering/Misc/GenericRenderWindow.js";
import vtkVolume from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Rendering/Core/Volume.js";
import vtkVolumeMapper from "https://cdn.jsdelivr.net/npm/@kitware/vtk.js@28.12.4/Rendering/Core/VolumeMapper.js";

const VTK_MAX = 192;

function isWebGLSupported() {
  try {
    const canvas = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")));
  } catch (e) {
    return false;
  }
}

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
  imageData.setOrigin(0, 0, 0);
  imageData.setSpacing(spacing[0], spacing[1], spacing[2]);
  imageData.setExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
  imageData.getPointData().setScalars(da);
}

function buildMaskVolume(imageData, maskArray, nx, ny, nz, spacing) {
  const da = vtkDataArray.newInstance({
    name: "mask",
    numberOfComponents: 1,
    values: maskArray,
  });
  imageData.setOrigin(0, 0, 0);
  imageData.setSpacing(spacing[0], spacing[1], spacing[2]);
  imageData.setExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
  imageData.getPointData().setScalars(da);
}

function ctVolumeActor(imageData) {
  return ctVolumeActorWithOpacityScale(imageData, 1.0);
}

function ctVolumeActorWithOpacityScale(imageData, opacityScale) {
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setBlendModeToComposite();
  const sp = imageData.getSpacing();
  const minSp = Math.min(sp[0] || 1, sp[1] || 1, sp[2] || 1);
  mapper.setSampleDistance(Math.max(minSp * 0.4, 0.15));
  mapper.setAutoAdjustSampleDistances(true);

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
  op.addPoint(c, 0.18 * opacityScale);
  op.addPoint(wmax, 0.55 * opacityScale);
  op.addPoint(hi, 0.72 * opacityScale);

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
  const sp = imageData.getSpacing();
  const minSp = Math.min(sp[0] || 1, sp[1] || 1, sp[2] || 1);
  mapper.setSampleDistance(Math.max(minSp * 0.35, 0.12));
  mapper.setAutoAdjustSampleDistances(true);

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
  const rw = genericRw.getRenderWindow();
  const renderer = genericRw.getRenderer();
  renderer.setBackground(0.04, 0.05, 0.08);

  const renderFrame = () => {
    genericRw.resize();
    rw.render();
  };
  requestAnimationFrame(() => requestAnimationFrame(renderFrame));

  const ro = new ResizeObserver(() => {
    genericRw.resize();
    rw.render();
  });
  ro.observe(container);
}

function prepareVtkContainer(container) {
  container.innerHTML = "";
  const wrap = el("div", "", "vtk-wrap");
  wrap.style.width = "100%";
  const h = Math.max(container.clientHeight || 0, 650);
  wrap.style.height = `${h}px`;
  wrap.style.minHeight = "650px";
  container.appendChild(wrap);
  return wrap;
}

async function loadVtkBuffers(metaUrl, huUrl, maskUrl) {
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

  const huImage = vtkImageData.newInstance();
  buildHuVolume(huImage, hu, nx, ny, nz, sp);

  const maskImage = vtkImageData.newInstance();
  buildMaskVolume(maskImage, mask, nx, ny, nz, sp);

  return { meta, dims: [nx, ny, nz], spacing: sp, huImage, maskImage };
}

function createGenericRw(container) {
  const genericRw = vtkGenericRenderWindow.newInstance({
    background: [0.16, 0.17, 0.19],
    listenWindowResize: true,
  });
  mountViewer(container, genericRw);
  return genericRw;
}

function renderVolumes(container, volumes) {
  const wrap = prepareVtkContainer(container);
  const genericRw = createGenericRw(wrap);
  const renderer = genericRw.getRenderer();
  volumes.forEach((v) => renderer.addVolume(v));
  renderer.resetCamera();
  renderer.resetCameraClippingRange();
  genericRw.resize();
  genericRw.getRenderWindow().render();
  return genericRw;
}

function summarizeVolumes(huImage, maskImage) {
  const huRange = huImage.getPointData().getScalars().getRange();
  const mkRange = maskImage.getPointData().getScalars().getRange();
  const mkVals = maskImage.getPointData().getScalars().getData();
  let mkCount = 0;
  for (let i = 0; i < mkVals.length; i++) if (mkVals[i] > 0) mkCount++;
  const pct = mkVals.length ? (100 * mkCount) / mkVals.length : 0;
  return `HU range: ${huRange[0].toFixed(1)}..${huRange[1].toFixed(1)} | mask: ${mkCount} vox (${pct.toFixed(2)}%) | mask range: ${mkRange[0]}..${mkRange[1]}`;
}

async function startVtkDicomViewer(opts) {
  const { container, metaUrl, huUrl, maskUrl } = opts;
  if (!container) return;

  container.innerHTML = "";
  const status = el("p", "Memuat volume DICOM + mask untuk VTK...", "vtk-status muted-note");
  container.appendChild(status);

  try {
    const { huImage, maskImage } = await loadVtkBuffers(metaUrl, huUrl, maskUrl);
    container.innerHTML = "";
    renderVolumes(container, [ctVolumeActor(huImage), maskVolumeActor(maskImage)]);
    const info = el("p", summarizeVolumes(huImage, maskImage), "muted-note");
    info.style.marginTop = "8px";
    container.appendChild(info);
  } catch (e) {
    console.error("[VTK viewer]", e);
    container.innerHTML = "";
    const msg = e && e.message
      ? `Gagal memuat visualisasi VTK: ${e.message}`
      : "Gagal memuat visualisasi VTK. Pastikan inferensi telah menghasilkan hu_volume.npy dan mask_pred.npy.";
    container.appendChild(el("p", msg, "muted-note"));
  }
}

function initVtkViewer() {
  const payload = document.getElementById("mesh-data");
  const vtkContainerLegacy = document.getElementById("viewer3d-vtk");
  const vtkCt = document.getElementById("viewer3d-vtk-ct");
  const vtkSeg = document.getElementById("viewer3d-vtk-seg");
  if (!payload || (!vtkContainerLegacy && !(vtkCt && vtkSeg))) return;

  if (!isWebGLSupported()) {
    const msg = "Browser Anda tidak mendukung WebGL. Gunakan browser modern (Chrome, Edge, Safari) untuk melihat visualisasi 3D.";
    [vtkContainerLegacy, vtkCt, vtkSeg].forEach((c) => {
      if (c) {
        c.innerHTML = "";
        c.appendChild(el("p", msg, "muted-note"));
      }
    });
    return;
  }

  try {
    const result = JSON.parse(payload.textContent || "{}");
    const runId = result && result.run_id;
    if (!runId) throw new Error("run_id tidak ditemukan di halaman hasil.");
    const q = `max=${VTK_MAX}`;
    const base = new URL(window.location.href);
    const metaUrl = new URL(`/runs/${encodeURIComponent(runId)}/vtk_meta?${q}`, base).href;
    const huUrl = new URL(`/runs/${encodeURIComponent(runId)}/vtk_hu.bin?${q}`, base).href;
    const maskUrl = new URL(`/runs/${encodeURIComponent(runId)}/vtk_mask.bin?${q}`, base).href;

    const go = async () => {
      if (vtkCt && vtkSeg) {
        vtkCt.innerHTML = "";
        vtkSeg.innerHTML = "";
        const note = el("p", "Memuat data VTK (HU + mask)...", "muted-note");
        note.style.margin = "0 0 10px";
        vtkCt.appendChild(note);
        try {
          const { huImage, maskImage } = await loadVtkBuffers(metaUrl, huUrl, maskUrl);
          vtkCt.innerHTML = "";
          vtkSeg.innerHTML = "";

          renderVolumes(vtkCt, [ctVolumeActor(huImage)]);
          renderVolumes(vtkSeg, [ctVolumeActorWithOpacityScale(huImage, 0.35), maskVolumeActor(maskImage)]);

          const infoCt = el("p", summarizeVolumes(huImage, maskImage), "muted-note");
          infoCt.style.marginTop = "8px";
          vtkCt.appendChild(infoCt);

          const infoSeg = el("p", "Tips: putar (drag), zoom (scroll). Mask = overlay.", "muted-note");
          infoSeg.style.marginTop = "8px";
          vtkSeg.appendChild(infoSeg);
          return;
        } catch (e) {
          console.error("[VTK compare viewer]", e);
          const msg = e && e.message ? `Gagal memuat VTK compare: ${e.message}` : "Gagal memuat VTK compare.";
          vtkCt.innerHTML = "";
          vtkSeg.innerHTML = "";
          vtkCt.appendChild(el("p", msg, "muted-note"));
          vtkSeg.appendChild(el("p", msg, "muted-note"));
          return;
        }
      }

      if (vtkContainerLegacy) {
        startVtkDicomViewer({ container: vtkContainerLegacy, metaUrl, huUrl, maskUrl });
      }
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", go);
    } else {
      go();
    }
  } catch (e) {
    console.error("[VTK viewer] inisialisasi", e);
    const msg = e && e.message ? `Visualisasi VTK tidak bisa dimulai: ${e.message}` : "Visualisasi VTK tidak bisa dimulai.";
    if (vtkCt && vtkSeg) {
      vtkCt.innerHTML = "";
      vtkSeg.innerHTML = "";
      vtkCt.appendChild(el("p", msg, "muted-note"));
      vtkSeg.appendChild(el("p", msg, "muted-note"));
    } else if (vtkContainerLegacy) {
      vtkContainerLegacy.innerHTML = "";
      vtkContainerLegacy.appendChild(el("p", msg, "muted-note"));
    }
  }
}

// Lazy init: only run when called or when section is expanded.
window.initVtkViewer = initVtkViewer;
