from __future__ import annotations

import json
import math
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from skimage.measure import marching_cubes
from PIL import Image
from werkzeug.utils import secure_filename

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
    mesh_ply_colored: str
    mesh_3d_colored_zip: str
    lesion_voxels: int
    lesion_volume_mm3: float
    lesion_volume_ml: float
    spacing: tuple[float, float, float]
    slices: int
    shape_hw: tuple[int, int]
    hu_mesh: dict | None
    lesion_mesh: dict | None
    enable_3d: bool


def _downsample_volume(volume: np.ndarray, max_dim: int = 128) -> tuple[np.ndarray, int]:
    """Turunkan resolusi isotropik; kembalikan juga stride agar spacing fisik marching_cubes benar."""
    if max(volume.shape) <= max_dim:
        return volume, 1
    stride = max(1, int(math.ceil(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride], stride


def _marching_surface(
    volume: np.ndarray,
    mesh_spacing: tuple[float, float, float],
    level: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    # GLTF bounds: X:[-12.02, 9.87], Y:[-26.26, -3.65], Z:[-24.20, 0.80]
    # Center: [-1.075, -14.955, -11.7]
    # Extents: [21.89, 22.61, 25.0]
    brain_min = np.array([-12.02, -26.26, -24.20])
    brain_max = np.array([9.87, -3.65, 0.80])
    brain_center = (brain_min + brain_max) / 2.0
    brain_size = brain_max - brain_min

    if np.nanmax(volume) <= level:
        return None
    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=level,
        spacing=(mesh_spacing[2], mesh_spacing[0], mesh_spacing[1]),
    )
    if len(verts) == 0 or len(faces) == 0:
        return None

    # Transform to GLTF space: center lesion, then scale to brain size
    xyz_raw = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]).astype(np.float64)
    c_min, c_max = xyz_raw.min(axis=0), xyz_raw.max(axis=0)
    c_center = (c_min + c_max) / 2.0
    c_size = np.maximum(c_max - c_min, 1e-6)

    # Normalize lesion to [-0.5, 0.5] then map to brain space
    xyz = ((xyz_raw - c_center) / c_size) * brain_size + brain_center
    return xyz, faces.astype(np.int64)


def _mesh_to_json_from_surface(xyz: np.ndarray, faces: np.ndarray) -> dict:
    return {
        "x": xyz[:, 0].round(4).tolist(),
        "y": xyz[:, 1].round(4).tolist(),
        "z": xyz[:, 2].round(4).tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


def _write_colored_combined_ply(
    path: Path,
    parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
) -> None:
    """Gabung beberapa mesh ke satu PLY ASCII dengan warna per vertex (RGB 0–255)."""
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
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i in range(len(verts)):
            x, y, z = verts[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _write_colored_obj_zip(
    out_dir: Path,
    parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]],
) -> str | None:
    """ZIP berisi OBJ+MTL: satu objek per mesh dengan warna diffuse (nama objek = label)."""
    if not parts:
        return None

    mtl_lines = []
    obj_lines: list[str] = []
    v_base = 1
    for idx, (xyz, faces, rgb, name) in enumerate(parts):
        if len(xyz) == 0:
            continue
        mat = f"mat_{idx}_{name}"
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        mtl_lines.append(
            f"newmtl {mat}\nKd {r:.6f} {g:.6f} {b:.6f}\nKa 0.2 0.2 0.2\nKs 0.3 0.3 0.3\nd 1.0\n"
        )
        obj_lines.append(f"o {name}\nusemtl {mat}\n")
        for row in xyz:
            obj_lines.append(f"v {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        for tri in faces:
            a, b_, c = int(tri[0]) + v_base, int(tri[1]) + v_base, int(tri[2]) + v_base
            obj_lines.append(f"f {a} {b_} {c}\n")
        v_base += len(xyz)

    if not obj_lines:
        return None

    mtl_name = "mesh_surfaces.mtl"
    obj_name = "mesh_ct_lesion.obj"
    zip_path = out_dir / "mesh_3d_colored.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(mtl_name, "".join(mtl_lines))
        zf.writestr(obj_name, f"mtllib {mtl_name}\n" + "".join(obj_lines))
    return zip_path.name


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

    enable_3d = len(slices) > 1

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

    hu_mesh = None
    lesion_mesh = None
    mesh_ply_name = ""
    mesh_3d_zip = ""
    if enable_3d:
        # Mesh CT dan lesi harus memakai grid + spacing fisik yang sama. Sebelumnya CT
        # di-downsample tanpa menaikkan spacing → kontur CT "mengecil" ke dekat origin
        # sementara lesi memakai voxel penuh → tampak jauh dan tidak proporsional.
        mesh_max_dim = 100
        hu_volume_for_mesh, stride = _downsample_volume(hu_vol, max_dim=mesh_max_dim)
        mask_for_mesh = masks.astype(np.float32)[::stride, ::stride, ::stride]
        mesh_spacing = (stride * ps_row, stride * ps_col, stride * ps_z)
        hu_level = float(np.percentile(hu_volume_for_mesh, 60))
        hu_surf = _marching_surface(hu_volume_for_mesh, mesh_spacing, hu_level)
        lesion_surf = (
            _marching_surface(mask_for_mesh, mesh_spacing, 0.5) if lesion_vox > 0 else None
        )
        hu_mesh = _mesh_to_json_from_surface(*hu_surf) if hu_surf else None
        lesion_mesh = _mesh_to_json_from_surface(*lesion_surf) if lesion_surf else None

        # Warna selaras viewer 3D: CT abu-biru terang, lesi oranye (RGB 0–255).
        ct_rgb = (188, 200, 218)
        lesion_rgb = (234, 88, 12)
        mesh_ply_name = "mesh_ct_lesion_colored.ply"
        ply_parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
        obj_parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]] = []
        if hu_surf:
            ply_parts.append((*hu_surf, ct_rgb))
            obj_parts.append((*hu_surf, ct_rgb, "ct_surface"))
        if lesion_surf:
            ply_parts.append((*lesion_surf, lesion_rgb))
            obj_parts.append((*lesion_surf, lesion_rgb, "lesion_mask"))
        if ply_parts:
            _write_colored_combined_ply(out_dir / mesh_ply_name, ply_parts)
            zname = _write_colored_obj_zip(out_dir, obj_parts)
            if zname:
                mesh_3d_zip = zname
        else:
            mesh_ply_name = ""

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
    # Mask 0/255 untuk viewer (Papaya): rentang lebih lebar dari 0/1 agar overlay terbaca stabil.
    nib.save(
        nib.Nifti1Image((mask_hwz.astype(np.uint8) * 255), affine_view),
        str(mask_view_nii_path),
    )

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
        mesh_ply_colored=mesh_ply_name,
        mesh_3d_colored_zip=mesh_3d_zip,
        lesion_voxels=lesion_vox,
        lesion_volume_mm3=round(lesion_mm3, 2),
        lesion_volume_ml=round(lesion_ml, 4),
        spacing=(ps_row, ps_col, ps_z),
        slices=masks.shape[0],
        shape_hw=(masks.shape[1], masks.shape[2]),
        hu_mesh=hu_mesh,
        lesion_mesh=lesion_mesh,
        enable_3d=enable_3d,
    )

    payload = asdict(result)
    (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


@dataclass
class ImageInferenceResult:
    run_id: str
    out_dir: str
    input_name: str
    original_png: str
    mask_png: str
    overlay_png: str
    lesion_pixels: int
    shape_hw: tuple[int, int]
    enable_3d: bool


def run_inference_image(image_path: Path, run_id: str, model_path: Path, runs_dir: Path) -> dict:
    if not model_path.exists():
        raise InferenceError(f"Model tidak ditemukan di: {model_path}")

    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(image_path).convert("L")
    except Exception as exc:
        raise InferenceError(f"Gagal membaca image: {exc}") from exc

    arr_u8 = np.array(img, dtype=np.uint8)
    vol01 = (arr_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    model, _ = build_unet_from_checkpoint(model_path)
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

    # Jika resize_if_needed mengubah ukuran, kembalikan ke ukuran input.
    if mask.shape != arr_u8.shape:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((arr_u8.shape[1], arr_u8.shape[0]), resample=Image.NEAREST)
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    lesion_px = int(mask.sum())

    original_png = "input.png"
    mask_png = "mask_pred.png"
    overlay_png = "overlay.png"

    Image.fromarray(arr_u8).save(out_dir / original_png, optimize=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / mask_png, optimize=True)

    # Overlay merah pada area mask.
    rgb = np.stack([arr_u8, arr_u8, arr_u8], axis=-1).astype(np.float32)
    alpha = 0.35
    m = mask.astype(bool)
    if m.any():
        overlay = np.zeros_like(rgb)
        overlay[..., 0] = 255.0
        overlay[..., 1] = 70.0
        overlay[..., 2] = 70.0
        rgb[m] = (1.0 - alpha) * rgb[m] + alpha * overlay[m]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(out_dir / overlay_png, optimize=True)

    input_name = secure_filename(image_path.name) or "image"
    result = ImageInferenceResult(
        run_id=run_id,
        out_dir=str(out_dir),
        input_name=input_name,
        original_png=original_png,
        mask_png=mask_png,
        overlay_png=overlay_png,
        lesion_pixels=lesion_px,
        shape_hw=(int(arr_u8.shape[0]), int(arr_u8.shape[1])),
        enable_3d=False,
    )

    payload = asdict(result)
    (out_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload
