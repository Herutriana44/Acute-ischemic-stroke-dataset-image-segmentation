"""
Inference adapter: wraps existing webapp services for FastAPI use.
Handles auto-GPU device selection and exception wrapping.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi_api.models import JobType
from webapp.services.inference_service import InferenceError

logger = logging.getLogger(__name__)


def infer_2d_image(
    image_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run single 2D image inference. Returns result dict.

    image_path: path to the uploaded image file
    run_id: unique identifier for this job
    model_path: path to model checkpoint
    runs_dir: base directory for output runs
    device: 'cuda' or 'cpu'
    """
    from webapp.services.inference_service import run_inference_image

    logger.info("infer_2d_image: %s → run %s (device=%s)", image_path.name, run_id, device)

    # Create output directory for this job
    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy image to output dir with proper name
    suffix = image_path.suffix.lower() or ".png"
    dest = out_dir / f"input{suffix}"
    if dest != image_path:
        shutil.copy2(str(image_path), str(dest))

    result = run_inference_image(
        image_path=dest,
        run_id=run_id,
        model_path=model_path,
        runs_dir=runs_dir,
    )
    result["device"] = device
    return result


def infer_single_dicom(
    dicom_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run single DICOM file inference. Returns result dict.

    dicom_path: path to the uploaded .dcm file
    """
    from webapp.services.inference_service import run_inference_image

    logger.info("infer_single_dicom: %s → run %s (device=%s)", dicom_path.name, run_id, device)

    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / "input.dcm"
    if dest != dicom_path:
        shutil.copy2(str(dicom_path), str(dest))

    # run_inference_image handles .dcm internally via Image.open or we can use
    # the DICOM pipeline directly. Let's route through run_inference_image
    # which will load via PIL (for non-DICOM) — for .dcm we need special handling.
    # Check suffix and route accordingly:
    if dicom_path.suffix.lower() == ".dcm":
        # Use the desktop app's single-dicom function (from infer_dicom_unet.py logic)
        result = _infer_single_dicom_via_pipeline(
            dicom_path=dest,
            run_id=run_id,
            model_path=model_path,
            runs_dir=runs_dir,
            device=device,
        )
    else:
        result = run_inference_image(
            image_path=dest,
            run_id=run_id,
            model_path=model_path,
            runs_dir=runs_dir,
        )
    result["device"] = device
    return result


def _infer_single_dicom_via_pipeline(
    dicom_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Infer on a single .dcm file using the DICOM pipeline (HU windowing + model)."""
    import numpy as np
    import pydicom
    import torch
    from PIL import Image

    from infer_dicom_unet import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_unet_from_checkpoint,
        resize_if_needed,
    )
    from unet_segmentation.dicom_pipeline import postprocess_mask2d, window_hu
    from webapp.services.inference_service import InferenceError

    if not model_path.exists():
        raise InferenceError(f"Model not found: {model_path}")

    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(np.float32)
    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        vol01 = window_hu(arr, center=40.0, width=80.0).astype(np.float32)
    except Exception:
        arr_min, arr_max = arr.min(), arr.max()
        vol01 = (arr - arr_min) / max(arr_max - arr_min, 1.0)

    arr_u8 = np.clip(vol01 * 255.0, 0, 255).astype(np.uint8)

    model, _ = build_unet_from_checkpoint(model_path)
    torch_device = torch.device(device)
    model = model.to(torch_device)

    with torch.no_grad():
        img_resized = resize_if_needed(vol01, None)
        rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(torch_device)
        logits = model(tensor_img)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8)
        mask = postprocess_mask2d(mask, min_area=64, closing_radius=2)

    if mask.shape != arr_u8.shape:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.resize((arr_u8.shape[1], arr_u8.shape[0]), resample=Image.NEAREST)
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    lesion_px = int(mask.sum())

    out_dir = runs_dir / run_id
    Image.fromarray(arr_u8).save(out_dir / "input.png", optimize=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "mask_pred.png", optimize=True)

    # Overlay
    rgb_arr = np.stack([arr_u8, arr_u8, arr_u8], axis=-1).astype(np.float32)
    alpha = 0.35
    m = mask.astype(bool)
    if m.any():
        overlay = np.zeros_like(rgb_arr)
        overlay[..., 0] = 255.0
        overlay[..., 1] = 70.0
        overlay[..., 2] = 70.0
        rgb_arr[m] = (1.0 - alpha) * rgb_arr[m] + alpha * overlay[m]
    Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8)).save(
        out_dir / "overlay.png", optimize=True
    )

    result = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "input_name": dicom_path.name,
        "original_png": "input.png",
        "mask_png": "mask_pred.png",
        "overlay_png": "overlay.png",
        "lesion_pixels": lesion_px,
        "shape_hw": [int(arr_u8.shape[0]), int(arr_u8.shape[1])],
        "enable_3d": False,
        "device": device,
    }
    return result


def infer_2d_image_yolo(
    image_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run single 2D image inference using YOLO detection model.

    image_path: path to the uploaded image file
    run_id: unique identifier for this job
    model_path: path to YOLO model checkpoint (.pt)
    runs_dir: base directory for output runs
    device: 'cuda' or 'cpu'
    """
    import json
    import numpy as np
    from PIL import Image, ImageDraw

    try:
        from ultralytics import YOLO
    except ImportError:
        raise InferenceError("ultralytics not installed. Run: pip install ultralytics")

    if not model_path.exists():
        raise InferenceError(f"YOLO model not found: {model_path}")

    logger.info("infer_2d_image_yolo: %s → run %s (device=%s)", image_path.name, run_id, device)

    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = image_path.suffix.lower() or ".png"
    dest = out_dir / f"input{suffix}"
    if dest != image_path:
        shutil.copy2(str(image_path), str(dest))

    img = Image.open(dest).convert("RGB")
    arr = np.array(img)

    yolo_model = YOLO(str(model_path))
    results = yolo_model.predict(
        source=arr,
        device=device,
        verbose=False,
    )

    bboxes = []
    overlay = arr.copy()

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            bboxes.append({
                "x_min": int(round(x1)),
                "y_min": int(round(y1)),
                "x_max": int(round(x2)),
                "y_max": int(round(y2)),
                "confidence": round(conf, 4),
            })
            # Draw bounding box on overlay
            import cv2
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    Image.fromarray(arr).save(out_dir / "input.png", optimize=True)

    # Save bboxes JSON
    bboxes_path = out_dir / "bboxes.json"
    bboxes_path.write_text(json.dumps(bboxes, indent=2), encoding="utf-8")

    # Save overlay with bounding boxes
    Image.fromarray(overlay).save(out_dir / "overlay.png", optimize=True)

    result = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "input_name": image_path.name,
        "original_png": "input.png",
        "bboxes_json": "bboxes.json",
        "overlay_png": "overlay.png",
        "bboxes": bboxes,
        "detection_count": len(bboxes),
        "avg_confidence": round(float(np.mean([b["confidence"] for b in bboxes])), 4) if bboxes else 0.0,
        "shape_hw": [int(arr.shape[0]), int(arr.shape[1])],
        "enable_3d": False,
        "device": device,
        "model_type": "yolo",
    }
    return result


def infer_single_dicom_yolo(
    dicom_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run single DICOM inference using YOLO detection (HU windowing → YOLO predict)."""
    import json
    import numpy as np
    import pydicom
    from PIL import Image, ImageDraw

    try:
        from ultralytics import YOLO
    except ImportError:
        raise InferenceError("ultralytics not installed. Run: pip install ultralytics")

    from unet_segmentation.dicom_pipeline import window_hu

    if not model_path.exists():
        raise InferenceError(f"YOLO model not found: {model_path}")

    logger.info("infer_single_dicom_yolo: %s → run %s (device=%s)", dicom_path.name, run_id, device)

    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / "input.dcm"
    if dest != dicom_path:
        shutil.copy2(str(dicom_path), str(dest))

    ds = pydicom.dcmread(str(dest))
    arr = ds.pixel_array.astype(np.float32)
    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        vol01 = window_hu(arr, center=40.0, width=80.0).astype(np.float32)
    except Exception:
        arr_min, arr_max = arr.min(), arr.max()
        vol01 = (arr - arr_min) / max(arr_max - arr_min, 1.0)

    arr_u8 = np.clip(vol01 * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([arr_u8, arr_u8, arr_u8], axis=-1)

    yolo_model = YOLO(str(model_path))
    results = yolo_model.predict(source=rgb, device=device, verbose=False)

    bboxes = []
    overlay = rgb.copy()

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            bboxes.append({
                "x_min": int(round(x1)),
                "y_min": int(round(y1)),
                "x_max": int(round(x2)),
                "y_max": int(round(y2)),
                "confidence": round(conf, 4),
            })
            # Draw bounding box on overlay
            import cv2
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    Image.fromarray(arr_u8).save(out_dir / "input.png", optimize=True)

    # Save bboxes JSON
    bboxes_path = out_dir / "bboxes.json"
    bboxes_path.write_text(json.dumps(bboxes, indent=2), encoding="utf-8")

    # Save overlay with bounding boxes
    Image.fromarray(overlay).save(out_dir / "overlay.png", optimize=True)

    return {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "input_name": dicom_path.name,
        "original_png": "input.png",
        "bboxes_json": "bboxes.json",
        "overlay_png": "overlay.png",
        "bboxes": bboxes,
        "detection_count": len(bboxes),
        "avg_confidence": round(float(np.mean([b["confidence"] for b in bboxes])), 4) if bboxes else 0.0,
        "shape_hw": [int(arr_u8.shape[0]), int(arr_u8.shape[1])],
        "enable_3d": False,
        "device": device,
        "model_type": "yolo",
    }


def infer_dicom_series_yolo(
    archive_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run DICOM series inference using YOLO per-slice + 3D reconstruction.

    Mirrors webapp run_inference() but uses YOLO for per-slice segmentation.
    """
    import json
    import math
    import zipfile

    import nibabel as nib
    import numpy as np
    from PIL import Image
    from skimage.measure import marching_cubes

    try:
        from ultralytics import YOLO
    except ImportError:
        raise InferenceError("ultralytics not installed. Run: pip install ultralytics")

    from unet_segmentation.dicom_pipeline import (
        dicom_affine_from_slices,
        load_dicom_series,
        postprocess_mask2d,
        window_hu,
    )
    from webapp.services.archive_service import find_dicom_series_dir

    if not model_path.exists():
        raise InferenceError(f"YOLO model not found: {model_path}")

    logger.info("infer_dicom_series_yolo: %s → run %s (device=%s)", archive_path.name, run_id, device)

    out_dir = runs_dir / run_id
    extracted_dir = out_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Extract archive
    logger.info("Extracting %s...", archive_path.name)
    suffixes = "".join(archive_path.suffixes).lower()
    try:
        if suffixes.endswith(".zip"):
            shutil.unpack_archive(str(archive_path), str(extracted_dir), format="zip")
        elif suffixes.endswith((".tar.gz", ".tgz")):
            shutil.unpack_archive(str(archive_path), str(extracted_dir), format="gztar")
        elif suffixes.endswith((".tar.bz2", ".tbz")):
            shutil.unpack_archive(str(archive_path), str(extracted_dir), format="bztar")
        else:
            shutil.unpack_archive(str(archive_path), str(extracted_dir))
    except Exception as exc:
        raise InferenceError(f"Failed to extract archive: {exc}") from exc

    series_dir = find_dicom_series_dir(extracted_dir)
    logger.info("Found series at: %s", series_dir)

    # Copy DICOM series
    dicom_series_dir = out_dir / "dicom_series"
    dicom_series_dir.mkdir(parents=True, exist_ok=True)
    dicom_files = sorted([p for p in Path(series_dir).glob("*.dcm") if p.is_file()])
    if not dicom_files:
        dicom_files = sorted([p for p in Path(series_dir).iterdir() if p.is_file()])
    for fp in dicom_files:
        shutil.copy2(fp, dicom_series_dir / fp.name)
    dicom_zip_path = out_dir / "dicom_series.zip"
    shutil.make_archive(str(dicom_zip_path.with_suffix("")), "zip", root_dir=str(dicom_series_dir))

    try:
        slices = load_dicom_series(series_dir)
    except Exception as exc:
        raise InferenceError(f"Gagal membaca DICOM series: {exc}") from exc

    enable_3d = len(slices) > 1

    (dicom_series_dir / "slice_order.json").write_text(
        json.dumps({"ordered_filenames": [Path(s.path).name for s in slices]}),
        encoding="utf-8",
    )

    _, spacing = dicom_affine_from_slices(slices)
    ps_row, ps_col, ps_z = spacing
    hu_vol = np.stack([s.hu for s in slices], axis=0).astype(np.float32)
    vol01 = window_hu(hu_vol, center=40.0, width=80.0).astype(np.float32)

    # Per-slice YOLO inference
    yolo_model = YOLO(str(model_path))
    all_bboxes = []
    ct_u8 = np.clip(vol01 * 255.0, 0, 255).round().astype(np.uint8)
    overlay_dir = out_dir / "overlay_slices"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for i in range(vol01.shape[0]):
        slice_u8 = np.clip(vol01[i] * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([slice_u8, slice_u8, slice_u8], axis=-1)
        results = yolo_model.predict(source=rgb, device=device, verbose=False)

        overlay = rgb.copy()
        bboxes = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                bboxes.append({
                    "x_min": int(round(x1)),
                    "y_min": int(round(y1)),
                    "x_max": int(round(x2)),
                    "y_max": int(round(y2)),
                    "confidence": round(conf, 4),
                })
                import cv2
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

        all_bboxes.extend(bboxes)
        Image.fromarray(overlay).save(overlay_dir / f"{i:04d}.png", optimize=True)

    # 3D mesh
    hu_mesh = None
    mesh_ply_name = ""
    mesh_3d_zip = ""
    if enable_3d:
        mesh_max_dim = 100
        hu_volume_for_mesh, stride = _downsample_volume(hu_vol, max_dim=mesh_max_dim)
        mesh_spacing = (stride * ps_row, stride * ps_col, stride * ps_z)
        hu_level = float(np.percentile(hu_volume_for_mesh, 60))
        hu_surf = _marching_surface(hu_volume_for_mesh, mesh_spacing, hu_level)
        hu_mesh = _mesh_to_json_from_surface(*hu_surf) if hu_surf else None

        ct_rgb = (188, 200, 218)
        mesh_ply_name = "mesh_ct_colored.ply"
        ply_parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
        obj_parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]] = []
        if hu_surf:
            ply_parts.append((*hu_surf, ct_rgb))
            obj_parts.append((*hu_surf, ct_rgb, "ct_surface"))
        if ply_parts:
            _write_colored_combined_ply(out_dir / mesh_ply_name, ply_parts)
            zname = _write_colored_obj_zip(out_dir, obj_parts)
            if zname:
                mesh_3d_zip = zname
        else:
            mesh_ply_name = ""

        np.save(out_dir / "hu_volume.npy", hu_vol)

    # NIfTI output (CT only, no mask for detection)
    affine, _ = dicom_affine_from_slices(slices)
    ct_hu_nii_path = out_dir / "ct_hu.nii.gz"
    ct_nii_path = out_dir / "ct_window_u8.nii.gz"
    nib.save(nib.Nifti1Image(hu_vol.astype(np.float32), affine), str(ct_hu_nii_path))
    nib.save(nib.Nifti1Image(ct_u8, affine), str(ct_nii_path))

    # Save bboxes JSON with slice info
    bboxes_path = out_dir / "bboxes.json"
    bboxes_path.write_text(json.dumps(all_bboxes, indent=2), encoding="utf-8")

    ct_view_nii_path = out_dir / "ct_view_u8_hwz.nii.gz"
    ct_hwz = ct_u8.transpose(1, 2, 0)
    affine_view = np.eye(4, dtype=np.float64)
    affine_view[0, 0] = float(ps_col)
    affine_view[1, 1] = float(ps_row)
    affine_view[2, 2] = float(ps_z)
    nib.save(nib.Nifti1Image(ct_hwz.astype(np.uint8), affine_view), str(ct_view_nii_path))

    detection_count = len(all_bboxes)
    avg_confidence = round(float(np.mean([b["confidence"] for b in all_bboxes])), 4) if all_bboxes else 0.0

    result = {
        "run_id": run_id,
        "dicom_dir": str(series_dir),
        "out_dir": str(out_dir),
        "ct_nii": ct_nii_path.name,
        "ct_hu_nii": ct_hu_nii_path.name,
        "ct_view_nii": ct_view_nii_path.name,
        "overlay_slices_dir": overlay_dir.name,
        "dicom_series_dir": dicom_series_dir.name,
        "dicom_series_zip": dicom_zip_path.name,
        "mesh_ply_colored": mesh_ply_name,
        "mesh_3d_colored_zip": mesh_3d_zip,
        "bboxes_json": "bboxes.json",
        "bboxes": all_bboxes,
        "detection_count": detection_count,
        "avg_confidence": avg_confidence,
        "spacing": (ps_row, ps_col, ps_z),
        "slices": vol01.shape[0],
        "shape_hw": (vol01.shape[1], vol01.shape[2]),
        "hu_mesh": hu_mesh,
        "enable_3d": enable_3d,
        "device": device,
        "archive_name": archive_path.name,
        "model_type": "yolo",
    }

    (out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


# ── 3D mesh helpers (shared by U-Net and YOLO series) ────────────────────


def _downsample_volume(volume: np.ndarray, max_dim: int = 128) -> tuple[np.ndarray, int]:
    import math

    import numpy as np

    if max(volume.shape) <= max_dim:
        return volume, 1
    stride = max(1, int(math.ceil(max(volume.shape) / max_dim)))
    return volume[::stride, ::stride, ::stride], stride


def _marching_surface(
    volume: np.ndarray,
    mesh_spacing: tuple[float, float, float],
    level: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    import numpy as np
    from skimage.measure import marching_cubes

    if np.nanmax(volume) <= level:
        return None
    verts, faces, _, _ = marching_cubes(
        volume.astype(np.float32),
        level=level,
        spacing=(mesh_spacing[2], mesh_spacing[0], mesh_spacing[1]),
    )
    if len(verts) == 0 or len(faces) == 0:
        return None
    xyz = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]]).astype(np.float64)
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
    import numpy as np

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
    import zipfile

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


def infer_dicom_series(
    archive_path: Path,
    run_id: str,
    model_path: Path,
    runs_dir: Path,
    device: str = "cpu",
) -> dict:
    """Run DICOM series inference from a ZIP archive. Returns result dict.

    archive_path: path to uploaded ZIP/RAR archive containing DICOM series
    """
    import shutil as _shutil
    from webapp.services.archive_service import find_dicom_series_dir, InferenceError
    from webapp.services.inference_service import run_inference

    logger.info("infer_dicom_series: %s → run %s (device=%s)", archive_path.name, run_id, device)

    # Create extraction directory
    out_dir = runs_dir / run_id
    extracted_dir = out_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Extract archive
    logger.info("Extracting %s...", archive_path.name)
    suffixes = "".join(archive_path.suffixes).lower()
    try:
        if suffixes.endswith(".zip"):
            _shutil.unpack_archive(str(archive_path), str(extracted_dir), format="zip")
        elif suffixes.endswith((".tar.gz", ".tgz")):
            _shutil.unpack_archive(str(archive_path), str(extracted_dir), format="gztar")
        elif suffixes.endswith((".tar.bz2", ".tbz")):
            _shutil.unpack_archive(str(archive_path), str(extracted_dir), format="bztar")
        else:
            _shutil.unpack_archive(str(archive_path), str(extracted_dir))
    except Exception as exc:
        raise InferenceError(f"Failed to extract archive: {exc}") from exc

    # Find DICOM series directory
    logger.info("Finding DICOM series in extracted files...")
    series_dir = find_dicom_series_dir(extracted_dir)
    logger.info("Found series at: %s", series_dir)

    # Run full inference (existing webapp logic)
    result = run_inference(
        dicom_dir=series_dir,
        run_id=run_id,
        model_path=model_path,
        runs_dir=runs_dir,
    )
    result["device"] = device
    result["archive_name"] = archive_path.name
    return result