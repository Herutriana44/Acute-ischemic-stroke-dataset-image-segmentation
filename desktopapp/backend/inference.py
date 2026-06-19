"""Backend inference module for the desktop app.

Wraps the webapp's inference service functions so they can be called
directly from the desktop app without running a Flask server.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple

from unet_segmentation.dicom_pipeline import (
    dicom_affine_from_slices,
    load_dicom_series,
    postprocess_mask2d,
    window_hu,
)
from webapp.services.archive_service import extract_archive, find_dicom_series_dir

# Re-use inference helpers from webapp
from infer_dicom_unet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_unet_from_checkpoint,
    resize_if_needed,
)
import numpy as np
import nibabel as nib
import torch
from skimage.measure import marching_cubes
from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def find_model_path() -> Path:
    """Find best_unet.pt in executable directory, CWD, or project root."""
    # 1. Check directory of the running executable/script
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        path = exe_dir / "best_unet.pt"
        if path.exists():
            return path
    
    # 2. Check current working directory
    path = Path.cwd() / "best_unet.pt"
    if path.exists():
        return path
    
    # 3. Fallback to PROJECT_ROOT
    return PROJECT_ROOT / "best_unet.pt"

MODEL_PATH = find_model_path()
RUNS_BASE = PROJECT_ROOT / "desktopapp" / "runs"


def _ensure_runs_dir() -> Path:
    RUNS_BASE.mkdir(parents=True, exist_ok=True)
    return RUNS_BASE


def run_dicom_inference(
    archive_path: Path,
    run_id: Optional[str] = None,
    progress_cb=None,
) -> Tuple[Path, dict]:
    """Run full DICOM series inference. Returns (run_dir, result_dict).

    Parameters
    ----------
    archive_path : Path
        Path to the ZIP/RAR/TAR archive containing a DICOM series.
    run_id : str, optional
        Unique run identifier; generated automatically if not provided.
    progress_cb : callable(str), optional
        Optional callback to report progress messages.
    """
    def _progress(msg: str) -> None:
        print(msg)
        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass

    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    runs_dir = _ensure_runs_dir()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _progress(f"Extracting archive: {archive_path.name}")

    # Copy archive to a temp file so we can safely extract it
    from tempfile import NamedTemporaryFile
    with open(archive_path, "rb") as f:
        with NamedTemporaryFile(suffix=archive_path.suffix, delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = Path(tmp.name)

    try:
        # Extract
        extracted_dir = run_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(tmp_path, extracted_dir)

        # Find DICOM series directory (robust — no modality restriction)
        _progress("Searching for DICOM series…")
        series_dir = _find_dicom_series(extracted_dir)
        dcm_count = len(list(series_dir.glob("*.dcm")))
        _progress(f"Found DICOM series in: {series_dir.name!r} ({dcm_count} files)")

        # Run inference
        result = _run_inference_core(series_dir, run_id, run_dir, progress_cb=_progress)
        return run_dir, result
    finally:
        tmp_path.unlink(missing_ok=True)


def run_image_inference(
    image_path: Path, run_id: Optional[str] = None
) -> Tuple[Path, dict]:
    """Run single image or single DICOM inference. Returns (run_dir, result_dict)."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    runs_dir = _ensure_runs_dir()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Single .dcm file → route through DICOM pipeline (produces overlay.png too)
    if image_path.suffix.lower() == ".dcm":
        result = _run_single_dicom_inference(image_path, run_id, run_dir)
    else:
        result = _run_image_inference_core(image_path, run_id, run_dir)

    return run_dir, result


def _extract_archive(archive_path: Path, out_dir: Path) -> None:
    """Extract archive file to output directory."""
    import shutil

    suffixes = "".join(archive_path.suffixes).lower()
    try:
        if suffixes.endswith(".zip"):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="zip")
            return
        if suffixes.endswith((".tar.gz", ".tgz")):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="gztar")
            return
        if suffixes.endswith((".tar.bz2", ".tbz")):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="bztar")
            return
        if suffixes.endswith(".tar"):
            shutil.unpack_archive(str(archive_path), str(out_dir), format="tar")
            return
        import patoolib
        patoolib.extract_archive(str(archive_path), outdir=str(out_dir), verbosity=-1)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract archive: {exc}") from exc


def _find_dicom_series(extracted_dir: Path) -> Path:
    """Find DICOM series directory (CT series with most slices).

    Search strategy (in order of priority):
    1. Directories whose DICOM files report Modality == 'CT'.
    2. Directories whose path contains a component named 'CT' (case-insensitive).
    3. Directories that contain readable DICOM pixel data (any modality).
    4. The directory with the most .dcm files (last-resort fallback).
    """
    import pydicom

    # ── collect every directory that contains at least one .dcm file ──
    candidate_dirs: dict[Path, int] = {}
    for dicom_path in extracted_dir.rglob("*.dcm"):
        parent = dicom_path.parent
        candidate_dirs[parent] = candidate_dirs.get(parent, 0) + 1

    # Also handle files without .dcm extension that are still DICOM
    if not candidate_dirs:
        for p in extracted_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() not in {
                ".zip", ".tar", ".gz", ".json", ".txt", ".xml", ".jpg",
                ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".nii",
            }:
                try:
                    pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                    parent = p.parent
                    candidate_dirs[parent] = candidate_dirs.get(parent, 0) + 1
                except Exception:
                    pass

    if not candidate_dirs:
        raise RuntimeError(
            "No DICOM files found in archive. "
            "Make sure the ZIP contains .dcm files (e.g. 000.dcm, 001.dcm …)."
        )

    # ── pass 1: explicit CT modality ─────────────────────────────────
    ct_dirs: list[tuple[Path, int]] = []
    any_dicom_dirs: list[tuple[Path, int]] = []
    for d, n in candidate_dirs.items():
        sample_files = list(d.glob("*.dcm")) or [
            p for p in d.iterdir() if p.is_file()
        ]
        if not sample_files:
            continue
        try:
            ds = pydicom.dcmread(
                str(sample_files[0]), stop_before_pixels=True, force=True
            )
            modality = getattr(ds, "Modality", "").strip().upper()
            if modality == "CT":
                ct_dirs.append((d, n))
            # Track all dirs with readable DICOM (for pass 3)
            any_dicom_dirs.append((d, n))
        except Exception:
            # Unreadable header — still count it for last-resort fallback
            any_dicom_dirs.append((d, n))

    if ct_dirs:
        return max(ct_dirs, key=lambda x: x[1])[0]

    # ── pass 2: 'CT' in path components ──────────────────────────────
    path_ct_dirs: list[tuple[Path, int]] = []
    for d, n in candidate_dirs.items():
        if any(part.upper() == "CT" for part in d.parts):
            path_ct_dirs.append((d, n))

    if path_ct_dirs:
        return max(path_ct_dirs, key=lambda x: x[1])[0]

    # ── pass 3: any readable DICOM (modality-agnostic) ───────────────
    if any_dicom_dirs:
        return max(any_dicom_dirs, key=lambda x: x[1])[0]

    # ── pass 4: most .dcm files (absolute last resort) ────────────────
    return max(candidate_dirs.items(), key=lambda x: x[1])[0]


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write mesh to Wavefront OBJ format."""
    with path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def _write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write mesh to binary STL format using vtkmodules.

    verts : (N, 3) float64 — vertex coordinates (x, y, z)
    faces : (M, 3) int64   — triangle indices into verts
    """
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkTriangle
    from vtkmodules.vtkIOGeometry import vtkSTLWriter

    # Build vtkPoints
    vtk_points = vtkPoints()
    vtk_points.SetNumberOfPoints(len(verts))
    for i, (x, y, z) in enumerate(verts):
        vtk_points.SetPoint(i, float(x), float(y), float(z))

    # Build vtkCellArray of triangles
    vtk_cells = vtkCellArray()
    for tri in faces:
        triangle = vtkTriangle()
        triangle.GetPointIds().SetId(0, int(tri[0]))
        triangle.GetPointIds().SetId(1, int(tri[1]))
        triangle.GetPointIds().SetId(2, int(tri[2]))
        vtk_cells.InsertNextCell(triangle)

    poly = vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(vtk_cells)

    writer = vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()


def _run_inference_core(
    dicom_dir: Path, run_id: str, run_dir: Path, progress_cb=None
) -> dict:
    """Core DICOM inference logic (adapted from webapp/services/inference_service.py)."""
    def _progress(msg: str) -> None:
        print(msg)
        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    # Copy DICOM series
    dicom_series_dir = run_dir / "dicom_series"
    dicom_series_dir.mkdir(parents=True, exist_ok=True)
    dicom_files = sorted(
        [p for p in Path(dicom_dir).glob("*.dcm") if p.is_file()],
        key=lambda p: _extract_numeric(p.stem)
    )
    if not dicom_files:
        dicom_files = sorted(
            [p for p in Path(dicom_dir).iterdir() if p.is_file()],
            key=lambda p: _extract_numeric(p.stem)
        )
    for fp in dicom_files:
        shutil.copy2(fp, dicom_series_dir / fp.name)

    # Load DICOM series
    _progress(f"Loading {len(dicom_files)} DICOM slices…")
    slices = load_dicom_series(dicom_dir)
    enable_3d = len(slices) > 1
    _progress(f"Loaded {len(slices)} slices — enable_3d={enable_3d}")
    model, _ = build_unet_from_checkpoint(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Process slices
    _, spacing = dicom_affine_from_slices(slices)
    ps_row, ps_col, ps_z = spacing
    hu_vol = np.stack([s.hu for s in slices], axis=0).astype(np.float32)
    vol01 = window_hu(hu_vol, center=40.0, width=80.0).astype(np.float32)

    masks = np.zeros_like(vol01, dtype=np.uint8)
    with torch.no_grad():
        for i in range(vol01.shape[0]):
            img = resize_if_needed(vol01[i], None)
            rgb = np.stack([img, img, img], axis=-1)
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            logits = model(tensor_img)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            mask = (prob > 0.5).astype(np.uint8)
            mask = postprocess_mask2d(mask, min_area=64, closing_radius=2)
            masks[i] = mask

    # Metrics
    voxel_mm3 = float(ps_row * ps_col * ps_z)
    lesion_vox = int(masks.sum())
    lesion_mm3 = float(lesion_vox * voxel_mm3)
    lesion_ml = lesion_mm3 / 1000.0

    # Save NIfTI files
    affine, _ = dicom_affine_from_slices(slices)
    ct_hu_nii_path = run_dir / "ct_hu.nii.gz"
    ct_nii_path = run_dir / "ct_window_u8.nii.gz"
    mask_nii_path = run_dir / "mask_pred.nii.gz"
    nib.save(nib.Nifti1Image(hu_vol.astype(np.float32), affine), str(ct_hu_nii_path))
    ct_u8 = np.clip(vol01 * 255.0, 0, 255).round().astype(np.uint8)
    nib.save(nib.Nifti1Image(ct_u8, affine), str(ct_nii_path))
    nib.save(nib.Nifti1Image(masks.astype(np.uint8), affine), str(mask_nii_path))

    # NIfTI for Papaya viewer (H,W,Z layout)
    ct_view_nii_path = run_dir / "ct_view_u8_hwz.nii.gz"
    mask_view_nii_path = run_dir / "mask_view_hwz.nii.gz"
    ct_hwz = ct_u8.transpose(1, 2, 0)
    mask_hwz = masks.transpose(1, 2, 0)
    affine_view = np.eye(4, dtype=np.float64)
    affine_view[0, 0] = float(ps_col)
    affine_view[1, 1] = float(ps_row)
    affine_view[2, 2] = float(ps_z)
    nib.save(nib.Nifti1Image(ct_hwz.astype(np.uint8), affine_view), str(ct_view_nii_path))
    nib.save(nib.Nifti1Image((mask_hwz.astype(np.uint8) * 255), affine_view), str(mask_view_nii_path))

    # Overlay slices
    overlay_dir = run_dir / "overlay_slices"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    alpha = 0.35
    for z in range(ct_u8.shape[0]):
        gray = ct_u8[z]
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        m = masks[z].astype(bool)
        if m.any():
            overlay = np.zeros_like(rgb)
            overlay[..., 0] = 255.0
            overlay[..., 1] = 70.0
            overlay[..., 2] = 70.0
            rgb[m] = (1.0 - alpha) * rgb[m] + alpha * overlay[m]
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(overlay_dir / f"{z:04d}.png")

    # 3D meshes
    hu_mesh = None
    lesion_mesh = None
    mesh_ply_name = ""
    if enable_3d:
        # ── per-axis downsampling ──────────────────────────────────────
        # Downsample each axis independently to a max of mesh_max_dim voxels.
        # This preserves the relative proportions of the volume instead of
        # squashing the Z axis (which usually has far fewer slices than XY).
        mesh_max_dim = 128
        Z, Y, X = hu_vol.shape
        sz_s = max(1, int(round(Z / mesh_max_dim)))  # stride along Z
        sy_s = max(1, int(round(Y / mesh_max_dim)))  # stride along Y
        sx_s = max(1, int(round(X / mesh_max_dim)))  # stride along X

        hu_volume_for_mesh = hu_vol[::sz_s, ::sy_s, ::sx_s]
        mask_for_mesh = masks.astype(np.float32)[::sz_s, ::sy_s, ::sx_s]

        # Physical voxel size after downsampling (Z, Y, X order)
        mesh_spacing = (
            sz_s * ps_z,   # physical step along Z (slice direction)
            sy_s * ps_row, # physical step along Y (row direction)
            sx_s * ps_col, # physical step along X (col direction)
        )

        hu_level = float(np.percentile(hu_volume_for_mesh, 60))
        hu_surf = _marching_surface(hu_volume_for_mesh, mesh_spacing, hu_level)
        lesion_surf = _marching_surface(mask_for_mesh, mesh_spacing, 0.5) if lesion_vox > 0 else None

        if hu_surf:
            hu_mesh = _mesh_to_json(*hu_surf)
            _write_obj(run_dir / "brain.obj", *hu_surf)
            _write_stl(run_dir / "brain.stl", *hu_surf)
        if lesion_surf:
            lesion_mesh = _mesh_to_json(*lesion_surf)
            _write_obj(run_dir / "lesion.obj", *lesion_surf)
            _write_stl(run_dir / "lesion.stl", *lesion_surf)

        ply_path = run_dir / "mesh_ct_lesion_colored.ply"
        _write_colored_ply(ply_path, hu_surf, lesion_surf)
        mesh_ply_name = ply_path.name if ply_path.exists() else ""

        if lesion_surf:
            _write_colored_ply(run_dir / "lesion_mesh.ply", None, lesion_surf)

        # Save numpy volumes for VTK viewer
        np.save(run_dir / "hu_volume.npy", hu_vol)
        np.save(run_dir / "mask_pred.npy", masks)

    # Build result dict
    result = {
        "run_id": run_id,
        "dicom_dir": str(dicom_dir),
        "out_dir": str(run_dir),
        "ct_nii": ct_nii_path.name,
        "ct_hu_nii": ct_hu_nii_path.name,
        "mask_nii": mask_nii_path.name,
        "ct_view_nii": ct_view_nii_path.name,
        "mask_view_nii": mask_view_nii_path.name,
        "overlay_slices_dir": overlay_dir.name,
        "dicom_series_dir": dicom_series_dir.name,
        "mesh_ply_colored": mesh_ply_name,
        "lesion_voxels": lesion_vox,
        "lesion_volume_mm3": round(lesion_mm3, 2),
        "lesion_volume_ml": round(lesion_ml, 4),
        "spacing": [ps_row, ps_col, ps_z],
        "slices": masks.shape[0],
        "shape_hw": [masks.shape[1], masks.shape[2]],
        "hu_mesh": hu_mesh,
        "lesion_mesh": lesion_mesh,
        "enable_3d": enable_3d,
    }

    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result



def _run_single_dicom_inference(
    dcm_path: Path, run_id: str, run_dir: Path
) -> dict:
    """Inference on a single .dcm file — extracts the pixel array and runs 2D segmentation."""
    import pydicom

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    ds = pydicom.dcmread(str(dcm_path))

    # Convert pixel array to HU if possible, otherwise use raw values
    arr = ds.pixel_array.astype(np.float32)
    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        # Window to brain CT range [0,1]
        vol01 = window_hu(arr, center=40.0, width=80.0).astype(np.float32)
    except Exception:
        # Fallback: normalise to [0,1]
        arr_min, arr_max = arr.min(), arr.max()
        vol01 = (arr - arr_min) / max(arr_max - arr_min, 1.0)

    arr_u8 = np.clip(vol01 * 255.0, 0, 255).astype(np.uint8)

    model, _ = build_unet_from_checkpoint(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        img_resized = resize_if_needed(vol01, None)
        rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        logits = model(tensor_img)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8)
        mask = postprocess_mask2d(mask, min_area=64, closing_radius=2)

    # Resize mask back to original image size if needed
    if mask.shape != arr_u8.shape:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize(
            (arr_u8.shape[1], arr_u8.shape[0]), resample=Image.NEAREST
        )
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    lesion_px = int(mask.sum())

    original_png = "input.png"
    mask_png     = "mask_pred.png"
    overlay_png  = "overlay.png"

    Image.fromarray(arr_u8).save(run_dir / original_png, optimize=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(run_dir / mask_png, optimize=True)

    rgb_arr = np.stack([arr_u8, arr_u8, arr_u8], axis=-1).astype(np.float32)
    alpha = 0.35
    m = mask.astype(bool)
    if m.any():
        overlay_color = np.zeros_like(rgb_arr)
        overlay_color[..., 0] = 255.0
        overlay_color[..., 1] = 70.0
        overlay_color[..., 2] = 70.0
        rgb_arr[m] = (1.0 - alpha) * rgb_arr[m] + alpha * overlay_color[m]
    Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8)).save(
        run_dir / overlay_png, optimize=True
    )

    result = {
        "run_id": run_id,
        "input_name": dcm_path.name,
        "original_png": original_png,
        "mask_png": mask_png,
        "overlay_png": overlay_png,
        "lesion_pixels": lesion_px,
        "shape_hw": [arr_u8.shape[0], arr_u8.shape[1]],
        "enable_3d": False,
    }

    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def _run_image_inference_core(
    image_path: Path, run_id: str, run_dir: Path
) -> dict:
    """Core single image inference logic."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    img = Image.open(image_path).convert("L")
    arr_u8 = np.array(img, dtype=np.uint8)
    vol01 = (arr_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    model, _ = build_unet_from_checkpoint(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        img_resized = resize_if_needed(vol01, None)
        rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        logits = model(tensor_img)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8)
        mask = postprocess_mask2d(mask, min_area=64, closing_radius=2)

    if mask.shape != arr_u8.shape:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((arr_u8.shape[1], arr_u8.shape[0]), resample=Image.NEAREST)
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    lesion_px = int(mask.sum())

    original_png = "input.png"
    mask_png = "mask_pred.png"
    overlay_png = "overlay.png"

    Image.fromarray(arr_u8).save(run_dir / original_png, optimize=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(run_dir / mask_png, optimize=True)

    rgb = np.stack([arr_u8, arr_u8, arr_u8], axis=-1).astype(np.float32)
    alpha = 0.35
    m = mask.astype(bool)
    if m.any():
        overlay = np.zeros_like(rgb)
        overlay[..., 0] = 255.0
        overlay[..., 1] = 70.0
        overlay[..., 2] = 70.0
        rgb[m] = (1.0 - alpha) * rgb[m] + alpha * overlay[m]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(run_dir / overlay_png, optimize=True)

    result = {
        "run_id": run_id,
        "input_name": image_path.name,
        "original_png": original_png,
        "mask_png": mask_png,
        "overlay_png": overlay_png,
        "lesion_pixels": lesion_px,
        "shape_hw": [arr_u8.shape[0], arr_u8.shape[1]],
        "enable_3d": False,
    }

    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Helpers (adapted from webapp/services/inference_service.py)
# ---------------------------------------------------------------------------

def _extract_numeric(stem: str) -> tuple:
    """Extract numeric parts from a filename stem for natural sorting.

    Handles formats: 1, 01, 001, 0001, IMG-0001, CT_Slice_1, etc.
    Returns a tuple alternating strings and ints for correct ordering.
    """
    import re
    parts = re.split(r'(\d+)', stem)
    result = []
    for p in parts:
        if p.isdigit():
            result.append(int(p))
        else:
            result.append(p.lower())
    return tuple(result)


def _downsample_volume(volume: np.ndarray, max_dim: int = 128) -> tuple[np.ndarray, int]:
    """Uniform downsample (same stride on all axes). Returns (downsampled, stride).

    NOTE: Prefer per-axis downsampling for anisotropic volumes (e.g. CT with
    thick slices) to avoid squashing the Z axis.
    """
    if max(volume.shape) <= max_dim:
        return volume, 1
    stride = max(1, int(round(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride], stride


def _marching_surface(
    volume: np.ndarray,
    mesh_spacing: tuple[float, float, float],
    level: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run marching cubes on a (Z, Y, X) volume.

    Parameters
    ----------
    volume : ndarray (Z, Y, X)
    mesh_spacing : (sz, sy, sx) physical voxel size in mm
    level : iso-surface level
    """
    if np.nanmax(volume) <= level:
        return None
    from skimage.measure import marching_cubes

    sz, sy, sx = mesh_spacing  # Z, Y, X physical spacing

    # skimage marching_cubes expects volume in (row0, row1, row2) order.
    # Our volume is (Z, Y, X), so spacing must also be (sz, sy, sx).
    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=level,
        spacing=(sz, sy, sx),   # matches (Z, Y, X) axis order
    )
    if len(verts) == 0 or len(faces) == 0:
        return None

    # marching_cubes returns verts in the same axis order as the volume:
    # verts[:, 0] = Z-axis coords, verts[:, 1] = Y-axis, verts[:, 2] = X-axis.
    # Remap to (X, Y, Z) for standard 3-D rendering.
    xyz = np.column_stack([
        verts[:, 2],   # X
        verts[:, 1],   # Y
        verts[:, 0],   # Z
    ]).astype(np.float64)
    return xyz, faces.astype(np.int64)


def _mesh_to_json(xyz: np.ndarray, faces: np.ndarray) -> dict:
    return {
        "x": xyz[:, 0].round(4).tolist(),
        "y": xyz[:, 1].round(4).tolist(),
        "z": xyz[:, 2].round(4).tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


def _write_colored_ply(
    path: Path,
    hu_surf: tuple | None,
    lesion_surf: tuple | None,
) -> None:
    """Write a single PLY with CT (gray-blue) and lesion (orange) meshes."""
    parts = []
    if hu_surf:
        verts, faces = hu_surf
        parts.append((verts, faces, (188, 200, 218)))
    if lesion_surf:
        verts, faces = lesion_surf
        parts.append((verts, faces, (234, 88, 12)))

    if not parts:
        return

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    offset = 0
    for xyz, faces, rgb in parts:
        n = len(xyz)
        if n == 0:
            continue
        all_v.append(xyz)
        all_f.append(faces + offset)
        r, g, b = rgb
        all_rgb.append(np.array([[r, g, b]] * n, dtype=np.uint8))
        offset += n

    if not all_v:
        return

    verts = np.vstack(all_v)
    faces = np.vstack(all_f)
    colors = np.vstack(all_rgb)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(len(verts)):
            f.write(f"{verts[i][0]:.6f} {verts[i][1]:.6f} {verts[i][2]:.6f} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
