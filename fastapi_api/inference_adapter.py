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
