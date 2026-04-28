#!/usr/bin/env python3
"""Inferensi segmentasi untuk satu series DICOM (slice-by-slice) memakai best_unet.pt.

Output berupa file yang bisa dibuka:
- NIfTI mask: mask_pred.nii.gz
- Numpy: mask_pred.npy + hu_volume.npy
- Overlay PNG (montage): overlay_montage.png
- Ringkasan volume: summary.txt dan volume.csv
- Mesh PLY (opsional jika ada lesi): lesion_mesh.ply
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from unet_segmentation.dicom_pipeline import (
    dicom_affine_from_slices,
    load_dicom_series,
    postprocess_mask2d,
    window_hu,
    write_ply,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_unet_from_checkpoint(ckpt_path: Path) -> tuple[torch.nn.Module, str]:
    import segmentation_models_pytorch as smp

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    encoder = "resnet34"
    state = ckpt
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        encoder = ckpt.get("encoder") or encoder
        state = ckpt["model_state"]

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, encoder


def resize_if_needed(x: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return x
    h, w = x.shape[:2]
    if h == size and w == size:
        return x
    from skimage.transform import resize

    if x.ndim == 2:
        return resize(x, (size, size), order=1, preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
    return resize(x, (size, size, x.shape[2]), order=1, preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )


def make_overlay_montage(
    out_path: Path,
    vol01: np.ndarray,
    mask: np.ndarray,
    max_tiles: int = 16,
) -> None:
    import matplotlib.pyplot as plt

    z = vol01.shape[0]
    take = min(z, max_tiles)
    idxs = np.linspace(0, z - 1, take).round().astype(int).tolist()
    cols = int(math.ceil(math.sqrt(take)))
    rows = int(math.ceil(take / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= take:
            continue
        i = idxs[k]
        ax.imshow(vol01[i], cmap="gray", vmin=0, vmax=1)
        ax.imshow(mask[i], cmap="Reds", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"z={i}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="DICOM -> UNet -> mask 3D (NIfTI + PNG overlay)")
    ap.add_argument(
        "--dicom",
        type=Path,
        default=root / "dicom_example" / "0019983" / "CT",
        help="Folder series DICOM (default: dicom_example/0019983/CT)",
    )
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=root / "best_unet.pt",
        help="Checkpoint model (default: ./best_unet.pt)",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window-center", type=float, default=40.0)
    ap.add_argument("--window-width", type=float, default=80.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--resize", type=int, default=None, help="Resize sisi (opsional)")
    ap.add_argument("--min-area", type=int, default=64, help="Remove small objects per-slice")
    ap.add_argument("--closing-radius", type=int, default=2, help="Binary closing radius per-slice")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "outputs_dicom" / "0019983_CT",
        help="Folder output",
    )
    ap.add_argument("--save-slice-png", action="store_true", help="Simpan overlay per-slice (png)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    slices = load_dicom_series(args.dicom)
    affine, spacing = dicom_affine_from_slices(slices)
    ps_row, ps_col, ps_z = spacing

    model, encoder = build_unet_from_checkpoint(args.ckpt)
    device = torch.device(args.device)
    model = model.to(device)

    hu_vol = np.stack([s.hu for s in slices], axis=0).astype(np.float32)  # (Z,H,W)
    vol01 = window_hu(hu_vol, args.window_center, args.window_width).astype(np.float32)
    if args.resize is not None:
        vol01 = np.stack([resize_if_needed(vol01[i], args.resize) for i in range(vol01.shape[0])], axis=0)

    masks = np.zeros_like(vol01, dtype=np.uint8)
    with torch.no_grad():
        for i in range(vol01.shape[0]):
            img = vol01[i]
            rgb = np.stack([img, img, img], axis=-1)  # (H,W,3)
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            logits = model(t)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            m = (prob > args.threshold).astype(np.uint8)
            m = postprocess_mask2d(m, min_area=args.min_area, closing_radius=args.closing_radius)
            masks[i] = m

    # Save arrays
    np.save(args.out_dir / "hu_volume.npy", hu_vol)
    np.save(args.out_dir / "mask_pred.npy", masks)

    # Save NIfTI
    import nibabel as nib

    nii = nib.Nifti1Image(masks.astype(np.uint8), affine)
    nib.save(nii, str(args.out_dir / "mask_pred.nii.gz"))

    # Summary volume
    voxel_mm3 = float(ps_row * ps_col * ps_z)
    lesion_vox = int(masks.sum())
    lesion_mm3 = float(lesion_vox * voxel_mm3)
    lesion_ml = lesion_mm3 / 1000.0
    (args.out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"dicom_dir: {args.dicom.resolve()}",
                f"ckpt: {args.ckpt.resolve()}",
                f"encoder: {encoder}",
                f"slices: {masks.shape[0]}",
                f"shape: {masks.shape[1]}x{masks.shape[2]}",
                f"pixel_spacing_row_mm: {ps_row}",
                f"pixel_spacing_col_mm: {ps_col}",
                f"slice_spacing_mm: {ps_z}",
                f"voxel_volume_mm3: {voxel_mm3}",
                f"lesion_voxels: {lesion_vox}",
                f"lesion_volume_mm3: {lesion_mm3}",
                f"lesion_volume_ml: {lesion_ml}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with (args.out_dir / "volume.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["z_index", "lesion_pixels", "lesion_area_mm2"],
        )
        w.writeheader()
        area_mm2 = float(ps_row * ps_col)
        for z in range(masks.shape[0]):
            px = int(masks[z].sum())
            w.writerow({"z_index": z, "lesion_pixels": px, "lesion_area_mm2": px * area_mm2})

    # Montage PNG
    make_overlay_montage(args.out_dir / "overlay_montage.png", vol01, masks, max_tiles=16)

    # Optional per-slice overlay PNGs
    if args.save_slice_png:
        import matplotlib.pyplot as plt

        od = args.out_dir / "overlay_slices"
        od.mkdir(parents=True, exist_ok=True)
        for z in range(vol01.shape[0]):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.axis("off")
            ax.imshow(vol01[z], cmap="gray", vmin=0, vmax=1)
            ax.imshow(masks[z], cmap="Reds", alpha=0.35, vmin=0, vmax=1)
            fig.tight_layout()
            fig.savefig(od / f"{z:04d}.png", dpi=150)
            plt.close(fig)

    # Mesh PLY (jika ada lesi)
    if masks.sum() > 0:
        from skimage.measure import marching_cubes

        # marching_cubes expects (Z,Y,X). spacing in same order.
        verts, faces, _, _ = marching_cubes(
            masks.astype(np.float32),
            level=0.5,
            spacing=(ps_z, ps_row, ps_col),
        )
        write_ply(args.out_dir / "lesion_mesh.ply", verts, faces)

    print(f"Selesai. Output: {args.out_dir}")
    if lesion_vox > 0:
        print(f"Lesi terdeteksi: {lesion_ml:.3f} mL ({lesion_mm3:.1f} mm^3)")
    else:
        print("Tidak ada lesi terdeteksi (mask kosong).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)

