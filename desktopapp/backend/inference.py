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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "best_unet.pt"
RUNS_BASE = PROJECT_ROOT / "desktopapp" / "runs"


def _ensure_runs_dir() -> Path:
    RUNS_BASE.mkdir(parents=True, exist_ok=True)
    return RUNS_BASE


def run_dicom_inference(
    archive_path: Path, run_id: Optional[str] = None
) -> Tuple[Path, dict]:
    """Run full DICOM series inference. Returns (run_dir, result_dict)."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    runs_dir = _ensure_runs_dir()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Extract archive
    from werkzeug.datastructures import FileStorage
    from io import BytesIO

    # We need to simulate a FileStorage object for extract_archive
    with open(archive_path, "rb") as f:
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=archive_path.suffix, delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = Path(tmp.name)

    try:
        # Use archive_service to extract
        extracted_dir = run_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        # Extract manually based on suffix
        _extract_archive(tmp_path, extracted_dir)

        # Find DICOM series
        series_dir = _find_dicom_series(extracted_dir)

        # Run inference
        result = _run_inference_core(series_dir, run_id, run_dir)
        return run_dir, result
    finally:
        tmp_path.unlink(missing_ok=True)


def run_image_inference(
    image_path: Path, run_id: Optional[str] = None
) -> Tuple[Path, dict]:
    """Run single image inference. Returns (run_dir, result_dict)."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]

    runs_dir = _ensure_runs_dir()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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
    """Find DICOM series directory (CT series with most slices)."""
    import pydicom

    REQUIRED_MODALITY = "CT"
    candidate_dirs: dict[Path, int] = {}
    for dicom_path in extracted_dir.rglob("*.dcm"):
        parent = dicom_path.parent
        candidate_dirs[parent] = candidate_dirs.get(parent, 0) + 1

    if not candidate_dirs:
        raise RuntimeError("No .dcm files found in archive.")

    # Check for CT modality
    ct_dirs: list[tuple[Path, int]] = []
    for d, n in candidate_dirs.items():
        try:
            fp = next(d.glob("*.dcm"))
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", "").strip().upper() == REQUIRED_MODALITY:
                ct_dirs.append((d, n))
        except Exception:
            continue

    if ct_dirs:
        return max(ct_dirs, key=lambda x: x[1])[0]

    # Fallback: look for CT in path
    for d, n in candidate_dirs.items():
        if any(p.upper() == "CT" for p in d.parts):
            ct_dirs.append((d, n))

    if ct_dirs:
        return max(ct_dirs, key=lambda x: x[1])[0]

    raise RuntimeError("No DICOM CT series found in archive.")


def _run_inference_core(
    dicom_dir: Path, run_id: str, run_dir: Path
) -> dict:
    """Core DICOM inference logic (adapted from webapp/services/inference_service.py)."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    # Copy DICOM series
    dicom_series_dir = run_dir / "dicom_series"
    dicom_series_dir.mkdir(parents=True, exist_ok=True)
    dicom_files = sorted([p for p in Path(dicom_dir).glob("*.dcm") if p.is_file()])
    if not dicom_files:
        dicom_files = sorted([p for p in Path(dicom_dir).iterdir() if p.is_file()])
    for fp in dicom_files:
        shutil.copy2(fp, dicom_series_dir / fp.name)

    # Load DICOM series
    slices = load_dicom_series(dicom_dir)
    enable_3d = len(slices) > 1

    # Build model
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
        mesh_max_dim = 100
        hu_volume_for_mesh, stride = _downsample_volume(hu_vol, max_dim=mesh_max_dim)
        mask_for_mesh = masks.astype(np.float32)[::stride, ::stride, ::stride]
        mesh_spacing = (stride * ps_row, stride * ps_col, stride * ps_z)
        hu_level = float(np.percentile(hu_volume_for_mesh, 60))
        hu_surf = _marching_surface(hu_volume_for_mesh, mesh_spacing, hu_level)
        lesion_surf = _marching_surface(mask_for_mesh, mesh_spacing, 0.5) if lesion_vox > 0 else None

        if hu_surf:
            hu_mesh = _mesh_to_json(*hu_surf)
        if lesion_surf:
            lesion_mesh = _mesh_to_json(*lesion_surf)

        ply_path = run_dir / "mesh_ct_lesion_colored.ply"
        _write_colored_ply(ply_path, hu_surf, lesion_surf)
        mesh_ply_name = ply_path.name if ply_path.exists() else ""

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

def _downsample_volume(volume: np.ndarray, max_dim: int = 128) -> tuple[np.ndarray, int]:
    if max(volume.shape) <= max_dim:
        return volume, 1
    stride = max(1, int(round(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride], stride


def _marching_surface(
    volume: np.ndarray,
    mesh_spacing: tuple[float, float, float],
    level: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if np.nanmax(volume) <= level:
        return None
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=level,
        spacing=(mesh_spacing[2], mesh_spacing[0], mesh_spacing[1]),
    )
    if len(verts) == 0 or len(faces) == 0:
        return None
    xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]).astype(np.float64)
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
