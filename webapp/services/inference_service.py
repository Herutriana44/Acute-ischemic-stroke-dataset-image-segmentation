from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from skimage.measure import marching_cubes

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
    lesion_voxels: int
    lesion_volume_mm3: float
    lesion_volume_ml: float
    spacing: tuple[float, float, float]
    slices: int
    shape_hw: tuple[int, int]
    hu_mesh: dict | None
    lesion_mesh: dict | None


def _downsample_volume(volume: np.ndarray, max_dim: int = 128) -> np.ndarray:
    if max(volume.shape) <= max_dim:
        return volume
    stride = max(1, int(math.ceil(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride]


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

    try:
        slices = load_dicom_series(dicom_dir)
    except Exception as exc:
        raise InferenceError(f"Gagal membaca DICOM series: {exc}") from exc

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

    hu_volume_for_mesh = _downsample_volume(hu_vol, max_dim=100)
    hu_level = float(np.percentile(hu_volume_for_mesh, 60))
    hu_mesh = _mesh_to_json(hu_volume_for_mesh, spacing, level=hu_level)
    lesion_mesh = _mesh_to_json(masks, spacing, level=0.5) if lesion_vox > 0 else None

    np.save(out_dir / "hu_volume.npy", hu_vol)
    np.save(out_dir / "mask_pred.npy", masks)

    result = InferenceResult(
        run_id=run_id,
        dicom_dir=str(dicom_dir),
        out_dir=str(out_dir),
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
