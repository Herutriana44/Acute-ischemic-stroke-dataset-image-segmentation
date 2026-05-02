from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from skimage.measure import marching_cubes
from PIL import Image

from infer_dicom_unet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_unet_from_checkpoint,
    resize_if_needed,
)
from unet_segmentation.dicom_pipeline import (
    dicom_affine_from_slices,
    load_dicom_series,
    postprocess_mask2d,
    window_hu,
)


class InferenceError(Exception):
    pass


@dataclass
class InferenceResult:
    run_id: str
    dicom_dir: str
    out_dir: str
    ct_nii: str
    ct_hu_nii: str
    mask_nii: str
    ct_view_nii: str
    mask_view_nii: str
    overlay_slices_dir: str
    dicom_series_dir: str
    dicom_series_zip: str
    lesion_voxels: int
    lesion_volume_mm3: float
    lesion_volume_ml: float
    spacing: tuple[float, float, float]
    slices: int
    shape_hw: tuple[int, int]
    hu_mesh: dict | None
    lesion_mesh: dict | None


def _downsample_volume(volume: np.ndarray, max_dim: int = 128) -> tuple[np.ndarray, int]:
    """Turunkan resolusi isotropik; kembalikan juga stride agar spacing fisik marching_cubes benar."""
    if max(volume.shape) <= max_dim:
        return volume, 1
    stride = max(1, int(math.ceil(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride], stride


def _mesh_to_json(volume: np.ndarray, spacing: tuple[float, float, float], level: float) -> dict | None:
    if np.nanmax(volume) <= level:
        return None

    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=level,
        spacing=(spacing[2], spacing[0], spacing[1]),
    )
    if len(verts) == 0 or len(faces) == 0:
        return None

    return {
        "x": verts[:, 2].round(4).tolist(),
        "y": verts[:, 1].round(4).tolist(),
        "z": verts[:, 0].round(4).tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


def run_inference(dicom_dir: Path, run_id: str, model_path: Path, runs_dir: Path) -> dict:
    if not model_path.exists():
        raise InferenceError(f"Model tidak ditemukan di: {model_path}")

    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy DICOM series into runs/<run_id>/dicom_series so it can be accessed by URL.
    # This avoids the user needing to manually upload/select a DICOM folder in the viewer.
    dicom_series_dir = out_dir / "dicom_series"
    dicom_series_dir.mkdir(parents=True, exist_ok=True)
    dicom_src = Path(dicom_dir)
    dicom_files = sorted([p for p in dicom_src.glob("*.dcm") if p.is_file()])
    if not dicom_files:
        # If the series uses different extensions, we still keep the folder copy minimal by copying all files.
        dicom_files = sorted([p for p in dicom_src.iterdir() if p.is_file()])
    for fp in dicom_files:
        shutil.copy2(fp, dicom_series_dir / fp.name)
    dicom_zip_path = out_dir / "dicom_series.zip"
    shutil.make_archive(str(dicom_zip_path.with_suffix("")), "zip", root_dir=str(dicom_series_dir))

    try:
        slices = load_dicom_series(dicom_dir)
    except Exception as exc:
        raise InferenceError(f"Gagal membaca DICOM series: {exc}") from exc

    # Urutan slice untuk Papaya/manifest: harus sama dengan stacking volume + mask (bukan sort nama file).
    (dicom_series_dir / "slice_order.json").write_text(
        json.dumps({"ordered_filenames": [Path(s.path).name for s in slices]}),
        encoding="utf-8",
    )

    _, spacing = dicom_affine_from_slices(slices)
    ps_row, ps_col, ps_z = spacing
    hu_vol = np.stack([s.hu for s in slices], axis=0).astype(np.float32)
    vol01 = window_hu(hu_vol, center=40.0, width=80.0).astype(np.float32)

    model, _ = build_unet_from_checkpoint(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    voxel_mm3 = float(ps_row * ps_col * ps_z)
    lesion_vox = int(masks.sum())
    lesion_mm3 = float(lesion_vox * voxel_mm3)
    lesion_ml = lesion_mm3 / 1000.0

    # Mesh CT dan lesi harus memakai grid + spacing fisik yang sama. Sebelumnya CT
    # di-downsample tanpa menaikkan spacing → kontur CT "mengecil" ke dekat origin
    # sementara lesi memakai voxel penuh → tampak jauh dan tidak proporsional.
    mesh_max_dim = 100
    hu_volume_for_mesh, stride = _downsample_volume(hu_vol, max_dim=mesh_max_dim)
    mask_for_mesh = masks.astype(np.float32)[::stride, ::stride, ::stride]
    mesh_spacing = (stride * ps_row, stride * ps_col, stride * ps_z)
    hu_level = float(np.percentile(hu_volume_for_mesh, 60))
    hu_mesh = _mesh_to_json(hu_volume_for_mesh, mesh_spacing, level=hu_level)
    lesion_mesh = _mesh_to_json(mask_for_mesh, mesh_spacing, level=0.5) if lesion_vox > 0 else None

    np.save(out_dir / "hu_volume.npy", hu_vol)
    np.save(out_dir / "mask_pred.npy", masks)

    affine, _ = dicom_affine_from_slices(slices)
    ct_hu_nii_path = out_dir / "ct_hu.nii.gz"
    ct_nii_path = out_dir / "ct_window_u8.nii.gz"
    mask_nii_path = out_dir / "mask_pred.nii.gz"
    nib.save(nib.Nifti1Image(hu_vol.astype(np.float32), affine), str(ct_hu_nii_path))
    ct_u8 = np.clip(vol01 * 255.0, 0, 255).round().astype(np.uint8)
    nib.save(nib.Nifti1Image(ct_u8, affine), str(ct_nii_path))
    nib.save(nib.Nifti1Image(masks.astype(np.uint8), affine), str(mask_nii_path))

    # NIfTI khusus viewer Papaya:
    # Papaya (dan kebanyakan viewer NIfTI) menganggap dim-3 sebagai axis slice (Z).
    # Volume kita saat ini (Z,H,W). Untuk viewer yang konsisten, simpan sebagai (H,W,Z)
    # dengan affine sederhana berbasis spacing.
    ct_view_nii_path = out_dir / "ct_view_u8_hwz.nii.gz"
    mask_view_nii_path = out_dir / "mask_view_hwz.nii.gz"
    ct_hwz = ct_u8.transpose(1, 2, 0)  # (H,W,Z)
    mask_hwz = masks.transpose(1, 2, 0)  # (H,W,Z)
    affine_view = np.eye(4, dtype=np.float64)
    affine_view[0, 0] = float(ps_col)
    affine_view[1, 1] = float(ps_row)
    affine_view[2, 2] = float(ps_z)
    nib.save(nib.Nifti1Image(ct_hwz.astype(np.uint8), affine_view), str(ct_view_nii_path))
    nib.save(nib.Nifti1Image(mask_hwz.astype(np.uint8), affine_view), str(mask_view_nii_path))

    # Save per-slice overlay PNGs for a deterministic 2D viewer.
    overlay_dir = out_dir / "overlay_slices"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    alpha = 0.35
    for z in range(ct_u8.shape[0]):
        gray = ct_u8[z]  # (H,W) uint8
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        m = masks[z].astype(bool)
        if m.any():
            overlay = np.zeros_like(rgb)
            overlay[..., 0] = 255.0  # red
            overlay[..., 1] = 70.0
            overlay[..., 2] = 70.0
            rgb[m] = (1.0 - alpha) * rgb[m] + alpha * overlay[m]
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        img.save(overlay_dir / f"{z:04d}.png", optimize=True)

    result = InferenceResult(
        run_id=run_id,
        dicom_dir=str(dicom_dir),
        out_dir=str(out_dir),
        ct_nii=ct_nii_path.name,
        ct_hu_nii=ct_hu_nii_path.name,
        mask_nii=mask_nii_path.name,
        ct_view_nii=ct_view_nii_path.name,
        mask_view_nii=mask_view_nii_path.name,
        overlay_slices_dir=overlay_dir.name,
        dicom_series_dir=dicom_series_dir.name,
        dicom_series_zip=dicom_zip_path.name,
        lesion_voxels=lesion_vox,
        lesion_volume_mm3=round(lesion_mm3, 2),
        lesion_volume_ml=round(lesion_ml, 4),
        spacing=(ps_row, ps_col, ps_z),
        slices=masks.shape[0],
        shape_hw=(masks.shape[1], masks.shape[2]),
        hu_mesh=hu_mesh,
        lesion_mesh=lesion_mesh,
    )

    payload = asdict(result)
    (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload
