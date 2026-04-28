"""Pipeline inferensi DICOM -> mask 3D untuk model U-Net 2D.

Fokus: CT head (HU + windowing), inferensi slice-by-slice, stacking 3D dengan
affine berbasis metadata DICOM (PixelSpacing, ImageOrientationPatient,
ImagePositionPatient).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DicomSlice:
    path: Path
    instance_number: int | None
    position: np.ndarray  # (3,)
    iop: np.ndarray  # (6,) row_cos(3) + col_cos(3)
    pixel_spacing: tuple[float, float]  # (row_spacing, col_spacing)
    slice_thickness: float | None
    spacing_between_slices: float | None
    hu: np.ndarray  # (H,W) float32


def window_hu(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(hu, lo, hi)
    return (x - lo) / max(hi - lo, 1e-6)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def load_dicom_series(series_dir: Path) -> list[DicomSlice]:
    """Load DICOM CT series dari satu folder (file .dcm)."""
    import pydicom  # local import agar error jelas jika deps belum terpasang

    series_dir = Path(series_dir)
    files = sorted([p for p in series_dir.glob("*.dcm") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"Tidak ada .dcm di {series_dir}")

    slices: list[DicomSlice] = []
    for fp in files:
        ds = pydicom.dcmread(str(fp), force=True)
        if not hasattr(ds, "PixelData"):
            continue
        px = ds.pixel_array.astype(np.float32)
        slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        hu = px * slope + intercept

        inst = _safe_int(getattr(ds, "InstanceNumber", None), None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            pos = np.array([0.0, 0.0, float(inst or 0)], dtype=np.float32)
        else:
            pos = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float32)

        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None:
            iop_arr = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        else:
            iop_arr = np.array([float(v) for v in iop], dtype=np.float32)

        ps = getattr(ds, "PixelSpacing", None)
        if ps is None:
            pixel_spacing = (1.0, 1.0)
        else:
            pixel_spacing = (float(ps[0]), float(ps[1]))

        st = _safe_float(getattr(ds, "SliceThickness", None), None)
        sbs = _safe_float(getattr(ds, "SpacingBetweenSlices", None), None)

        slices.append(
            DicomSlice(
                path=fp,
                instance_number=inst,
                position=pos,
                iop=iop_arr,
                pixel_spacing=pixel_spacing,
                slice_thickness=st,
                spacing_between_slices=sbs,
                hu=hu,
            )
        )

    if not slices:
        raise RuntimeError(f"Tidak ada slice dengan PixelData di {series_dir}")

    # Sort: jika IOP tersedia, pakai proyeksi pada slice normal; fallback InstanceNumber.
    row = slices[0].iop[:3]
    col = slices[0].iop[3:]
    normal = np.cross(row, col)
    norm = float(np.linalg.norm(normal))
    if norm > 0:
        normal = normal / norm
        slices.sort(key=lambda s: float(np.dot(s.position, normal)))
    else:
        slices.sort(key=lambda s: (s.instance_number is None, s.instance_number or 0))

    return slices


def dicom_affine_from_slices(slices: list[DicomSlice]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Buat affine NIfTI (RAS-ish patient coords) dari metadata DICOM."""
    s0 = slices[0]
    row = s0.iop[:3].astype(np.float64)
    col = s0.iop[3:].astype(np.float64)
    slice_dir = np.cross(row, col)
    slice_dir_norm = np.linalg.norm(slice_dir)
    if slice_dir_norm > 0:
        slice_dir = slice_dir / slice_dir_norm
    else:
        slice_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ps_row, ps_col = s0.pixel_spacing

    # Z spacing: prefer spacing_between_slices, else thickness, else posisi antar slice.
    z = s0.spacing_between_slices or s0.slice_thickness
    if z is None and len(slices) >= 2:
        z = float(np.linalg.norm((slices[1].position - slices[0].position).astype(np.float64)))
    if z is None:
        z = 1.0

    origin = s0.position.astype(np.float64)

    affine = np.eye(4, dtype=np.float64)
    # Konvensi umum: axis-0 (kolom/x image) mengikuti col cosine * ps_col
    #               axis-1 (baris/y image) mengikuti row cosine * ps_row
    #               axis-2 (slice) mengikuti slice_dir * z
    affine[:3, 0] = col * float(ps_col)
    affine[:3, 1] = row * float(ps_row)
    affine[:3, 2] = slice_dir * float(z)
    affine[:3, 3] = origin
    return affine, (float(ps_row), float(ps_col), float(z))


def postprocess_mask2d(
    m: np.ndarray,
    min_area: int = 0,
    closing_radius: int = 0,
) -> np.ndarray:
    """Postprocess mask 2D biner (H,W) -> biner."""
    mm = (m > 0).astype(bool)
    if closing_radius > 0:
        from skimage.morphology import binary_closing, disk

        mm = binary_closing(mm, footprint=disk(closing_radius))
    if min_area > 0:
        from skimage.morphology import remove_small_objects

        mm = remove_small_objects(mm, min_size=int(min_area))
    return mm.astype(np.uint8)


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Simpan mesh sebagai PLY ASCII."""
    path = Path(path)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

