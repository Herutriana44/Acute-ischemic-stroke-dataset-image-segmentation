"""
Inference 2D DICOM → Parquet (raw HU + mask + has_segmentation)
Menggunakan pipeline dari unet_segmentation/dicom_pipeline.py

Catatan anti-korupsi:
    Baris dikumpulkan lalu ditulis sekali secara atomik (temp file ->
    fsync -> os.replace) dengan validasi baca-ulang, sehingga file Parquet
    tidak pernah dalam kondisi setengah jadi / rusak. Lihat
    unet_segmentation/parquet_io.py.
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from unet_segmentation.dicom_pipeline import (
    DicomSlice,
    load_dicom_series,
    postprocess_mask2d,
    window_hu,
)
from unet_segmentation.parquet_io import atomic_write_table, relative_name, validate_parquet

# Tulis checkpoint atomik tiap N slice (0 = hanya sekali di akhir).
CHECKPOINT_EVERY = 50

# Skema eksplisit & ringkas: konsisten antar-baris -> tidak ada schema drift.
DICOM_SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("raw_hu", pa.list_(pa.float32())),  # HU di-flatten (panjang = H*W)
        ("mask", pa.list_(pa.uint8())),       # mask biner di-flatten
        ("has_segmentation", pa.bool_()),
        ("shape_h", pa.int32()),
        ("shape_w", pa.int32()),
    ]
)

# --- LOAD MODEL (Sesuaikan dengan model kamu) ---
MODEL_PATH = "unet_segmentation/models/unet_2d.pth"  # <--- GANTI INI


def load_model():
    """Load model U-Net 2D (contoh placeholder)."""
    class SimpleUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(32, 1, 1)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2)
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.up(x)
            x = self.conv3(x)
            return torch.sigmoid(x)

    model = SimpleUNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
    else:
        print(f"⚠️ Model tidak ditemukan di {MODEL_PATH}, gunakan model random.")
    return model


model = load_model()


# --- FUNGSI INFERENCE ---
def infer_2d_hu(hu: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Inferensi mask dari array HU satu slice.

    Args:
        hu: array (H, W) float32 dalam satuan Hounsfield Unit.
    Returns:
        (segmentation_mask uint8 {0,1}, has_segmentation)
    """
    # 1. Windowing untuk brain CT, lalu skala ke [0, 255].
    hu_windowed = window_hu(hu, center=40, width=80)
    hu_windowed = (hu_windowed * 255).astype(np.float32)

    # 2. Inferensi
    input_tensor = torch.from_numpy(hu_windowed).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        output = model(input_tensor)
    mask = output.squeeze().numpy()  # (H, W) float32 [0, 1]

    # 3. Postprocess mask
    mask_binary = postprocess_mask2d(mask, min_area=10, closing_radius=2)  # (H, W) uint8 {0, 1}
    has_segmentation = bool(np.any(mask_binary == 1))

    return mask_binary, has_segmentation


def _build_row(slice_: DicomSlice, input_folder: Path) -> dict:
    """Jalankan inferensi satu slice -> satu baris record sesuai DICOM_SCHEMA."""
    raw_hu = slice_.hu
    mask, has_seg = infer_2d_hu(raw_hu)
    return {
        "filename": relative_name(slice_.path, input_folder),
        "raw_hu": raw_hu.astype(np.float32, copy=False).ravel().tolist(),
        "mask": mask.astype(np.uint8, copy=False).ravel().tolist(),
        "has_segmentation": bool(has_seg),
        "shape_h": int(raw_hu.shape[0]),
        "shape_w": int(raw_hu.shape[1]),
    }


def _save_atomic(records: list[dict], output_parquet: Path) -> None:
    """Bangun Table dari records lalu tulis + validasi secara atomik."""
    table = pa.Table.from_pylist(records, schema=DICOM_SCHEMA)
    atomic_write_table(table, output_parquet)
    validate_parquet(output_parquet, expected_rows=len(records))


# --- PROSES SEMUA DICOM DI FOLDER ---
def process_all_dicom_to_parquet(input_folder: Path, output_parquet: Path):
    """
    Proses semua file DICOM di `input_folder` (rekursif) → simpan ke Parquet.

    Series dimuat SEKALI per folder (bukan per file) untuk menghindari
    pembacaan ulang O(n^2) seperti pada versi lama.
    """
    input_folder = Path(input_folder)
    output_parquet = Path(output_parquet)

    if not input_folder.exists():
        raise FileNotFoundError(f"Folder tidak ada: {input_folder.resolve()}")

    records: list[dict] = []
    count = 0
    for root, _, files in os.walk(input_folder):
        has_dicom = any(f.lower().endswith(('.dcm', '.dicom')) for f in files)
        if not has_dicom:
            continue

        series_dir = Path(root)
        try:
            slices = load_dicom_series(series_dir)  # sekali per folder
        except Exception as e:
            print(f"❌ Gagal load series {series_dir}: {e}")
            continue

        for slice_ in slices:
            try:
                records.append(_build_row(slice_, input_folder))
                count += 1
                print(f"✅ Processed: {slice_.path} (total {count})")
                if CHECKPOINT_EVERY and count % CHECKPOINT_EVERY == 0:
                    _save_atomic(records, output_parquet)
                    print(f"💾 Checkpoint atomik tersimpan ({count} baris).")
            except Exception as e:
                print(f"❌ Error {slice_.path}: {e}")

    # Tulis final sekali secara atomik + validasi.
    if not records:
        print("⚠️ Tidak ada DICOM yang berhasil diproses. Menulis Parquet kosong (skema tetap).")
    _save_atomic(records, output_parquet)
    print(f"📁 Saved to: {output_parquet.resolve()} ({count} files) — tervalidasi OK ✅")


# --- MAIN ---
if __name__ == "__main__":
    INPUT_FOLDER = Path("all_dicom")  # Folder berisi DICOM
    OUTPUT_PARQUET = Path("inference_2d_results.parquet")

    process_all_dicom_to_parquet(INPUT_FOLDER, OUTPUT_PARQUET)
