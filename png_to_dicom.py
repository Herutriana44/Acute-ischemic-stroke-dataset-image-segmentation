"""png_to_dicom.py — Convert all PNG images in a folder to DICOM files.

Each PNG becomes one .dcm file, treated as a single CT slice.
Slices are ordered by filename (natural sort), and share the same
Series/Study UIDs so they form a proper DICOM series.

Usage:
    python png_to_dicom.py <input_folder> <output_folder> [options]

Examples:
    python png_to_dicom.py ./slices ./output_dcm
    python png_to_dicom.py ./slices ./output_dcm --patient-name "John Doe" --slice-thickness 5.0
    python png_to_dicom.py ./slices ./output_dcm --modality CT --window-center 40 --window-width 80

Requirements:
    pip install pydicom Pillow numpy
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
)
from PIL import Image


# ---------------------------------------------------------------------------
# Natural sort helper (so "slice_2.png" comes before "slice_10.png")
# ---------------------------------------------------------------------------
def _natural_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------
def png_folder_to_dicom(
    input_folder: str | Path,
    output_folder: str | Path,
    *,
    patient_name:     str   = "Anonymous",
    patient_id:       str   = "000000",
    study_description:str   = "PNG to DICOM Conversion",
    series_description:str  = "Converted Series",
    modality:         str   = "CT",
    slice_thickness:  float = 1.0,    # mm between slices
    pixel_spacing:    float = 1.0,    # mm per pixel (isotropic)
    window_center:    int   = 127,
    window_width:     int   = 255,
    bits_stored:      int   = 16,     # 8 or 16
) -> list[Path]:
    """Convert every PNG in *input_folder* to a DICOM file in *output_folder*.

    Returns the list of output .dcm paths.
    """
    input_folder  = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_folder}")

    png_files = sorted(input_folder.glob("*.png"), key=_natural_key)
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Shared UIDs / metadata for the whole series
    study_instance_uid  = generate_uid()
    series_instance_uid = generate_uid()
    study_date = datetime.now().strftime("%Y%m%d")
    study_time = datetime.now().strftime("%H%M%S")

    print(f"Found {len(png_files)} PNG file(s) in '{input_folder}'")
    print(f"Output → '{output_folder}'")
    print(f"Bits stored: {bits_stored}  |  Modality: {modality}")
    print("-" * 55)

    output_paths: list[Path] = []

    for idx, png_path in enumerate(png_files):
        slice_number = idx + 1   # 1-based

        # ── Load image ───────────────────────────────────────────────
        img = Image.open(png_path)

        # Convert to grayscale — DICOM CT stores single-channel
        img_gray = img.convert("L")   # 8-bit grayscale
        arr = np.array(img_gray, dtype=np.uint16 if bits_stored == 16 else np.uint8)

        # Scale to 16-bit range if requested
        if bits_stored == 16 and arr.dtype != np.uint16:
            arr = (arr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)

        rows, cols = arr.shape

        # ── File meta ────────────────────────────────────────────────
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

        # ── Dataset ──────────────────────────────────────────────────
        ds = FileDataset(
            str(output_folder / f"{png_path.stem}.dcm"),
            {},
            file_meta=file_meta,
            preamble=b"\x00" * 128,
        )
        ds.is_implicit_VR = False
        ds.is_little_endian = True

        # ── Patient ──────────────────────────────────────────────────
        ds.PatientName = patient_name
        ds.PatientID   = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex       = ""

        # ── Study ────────────────────────────────────────────────────
        ds.StudyInstanceUID  = study_instance_uid
        ds.StudyDate         = study_date
        ds.StudyTime         = study_time
        ds.StudyDescription  = study_description
        ds.AccessionNumber   = ""
        ds.ReferringPhysicianName = ""

        # ── Series ───────────────────────────────────────────────────
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber      = 1
        ds.SeriesDescription = series_description
        ds.Modality          = modality

        # ── Instance ─────────────────────────────────────────────────
        ds.SOPClassUID    = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.InstanceNumber = slice_number

        # ── Equipment ────────────────────────────────────────────────
        ds.Manufacturer           = "PNG2DICOM"
        ds.ManufacturerModelName  = "png_to_dicom.py"
        ds.SoftwareVersions       = "1.0"

        # ── Image geometry ───────────────────────────────────────────
        ds.SamplesPerPixel     = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows                = rows
        ds.Columns             = cols
        ds.PixelSpacing        = [pixel_spacing, pixel_spacing]
        ds.SliceThickness      = slice_thickness
        ds.SliceLocation       = float((slice_number - 1) * slice_thickness)

        # Image position: place slices along Z axis
        ds.ImagePositionPatient  = [0.0, 0.0, ds.SliceLocation]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        # ── Pixel data encoding ──────────────────────────────────────
        ds.BitsAllocated   = bits_stored
        ds.BitsStored      = bits_stored
        ds.HighBit         = bits_stored - 1
        ds.PixelRepresentation = 0          # unsigned
        ds.RescaleIntercept    = 0
        ds.RescaleSlope        = 1
        ds.RescaleType         = "HU" if modality == "CT" else "US"

        # ── Window / level (for display) ─────────────────────────────
        ds.WindowCenter = window_center
        ds.WindowWidth  = window_width

        # ── Pixel data ───────────────────────────────────────────────
        ds.PixelData = arr.tobytes()

        # ── Write ────────────────────────────────────────────────────
        out_path = output_folder / f"{png_path.stem}.dcm"
        pydicom.dcmwrite(str(out_path), ds)
        output_paths.append(out_path)

        print(f"  [{slice_number:3d}/{len(png_files)}] {png_path.name}"
              f"  →  {out_path.name}  ({rows}×{cols}px)")

    print("-" * 55)
    print(f"Done. {len(output_paths)} DICOM file(s) written to '{output_folder}'")
    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PNG images in a folder to DICOM files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_folder",  help="Folder containing PNG files")
    parser.add_argument("output_folder", help="Folder to write DICOM files")
    parser.add_argument("--patient-name",      default="Anonymous",
                        help="Patient name tag (default: Anonymous)")
    parser.add_argument("--patient-id",        default="000000",
                        help="Patient ID tag (default: 000000)")
    parser.add_argument("--study-description", default="PNG to DICOM Conversion")
    parser.add_argument("--series-description",default="Converted Series")
    parser.add_argument("--modality",          default="CT",
                        help="DICOM modality (default: CT)")
    parser.add_argument("--slice-thickness",   type=float, default=1.0,
                        help="Slice thickness in mm (default: 1.0)")
    parser.add_argument("--pixel-spacing",     type=float, default=1.0,
                        help="Pixel spacing in mm (default: 1.0)")
    parser.add_argument("--window-center",     type=int,   default=127,
                        help="Window center for display (default: 127)")
    parser.add_argument("--window-width",      type=int,   default=255,
                        help="Window width for display (default: 255)")
    parser.add_argument("--bits",              type=int,   default=16,
                        choices=[8, 16],
                        help="Bits stored per pixel: 8 or 16 (default: 16)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        png_folder_to_dicom(
            input_folder       = args.input_folder,
            output_folder      = args.output_folder,
            patient_name       = args.patient_name,
            patient_id         = args.patient_id,
            study_description  = args.study_description,
            series_description = args.series_description,
            modality           = args.modality,
            slice_thickness    = args.slice_thickness,
            pixel_spacing      = args.pixel_spacing,
            window_center      = args.window_center,
            window_width       = args.window_width,
            bits_stored        = args.bits,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
