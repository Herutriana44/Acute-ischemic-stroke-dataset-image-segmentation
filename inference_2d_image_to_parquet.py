"""
Inference 2D image → Parquet (raw pixels + mask + has_segmentation)
Input: gambar dari folder all_data_gambar (jpg, png, dll)

Catatan anti-korupsi:
    Versi lama menulis ulang (overwrite) file Parquet yang sama di SETIAP
    iterasi loop (read_parquet -> concat -> to_parquet). Bila proses ter-
    interupsi saat menulis, footer Parquet tidak lengkap dan SELURUH file
    jadi rusak. Versi ini mengumpulkan baris lalu menulis sekali secara
    atomik (lihat unet_segmentation/parquet_io.py), plus checkpoint atomik
    berkala agar progres aman tanpa risiko korupsi.
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from unet_segmentation.dicom_pipeline import postprocess_mask2d
from unet_segmentation.parquet_io import atomic_write_table, relative_name, validate_parquet

# Tulis checkpoint atomik tiap N gambar (0 = hanya sekali di akhir).
# Setiap checkpoint adalah file Parquet lengkap & valid, jadi aman dari korupsi.
# Untuk dataset sangat besar, set 0 agar tidak menulis ulang berulang.
CHECKPOINT_EVERY = 50

# Skema eksplisit & ringkas: konsisten antar-baris -> tidak ada schema drift.
IMAGE_SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("raw_pixels", pa.list_(pa.float32())),  # piksel di-flatten (panjang = H*W)
        ("mask", pa.list_(pa.uint8())),           # mask biner di-flatten
        ("has_segmentation", pa.bool_()),
        ("shape_h", pa.int32()),
        ("shape_w", pa.int32()),
    ]
)

# --- LOAD MODEL ---
MODEL_PATH = "unet_segmentation/models/unet_2d.pth"


def load_model():
    """Load model U-Net 2D."""
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
def infer_2d_image(image_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Args:
        image_path: Path ke file gambar (.jpg, .png, dll)
    Returns:
        (raw_pixels, segmentation_mask, has_segmentation)
    """
    # 1. Load image
    img = Image.open(image_path)

    # 2. Convert ke grayscale jika RGB
    if img.mode != 'L':
        img = img.convert('L')

    # 3. Convert ke numpy array
    pixels = np.array(img, dtype=np.float32)  # (H, W) float32

    # 4. Normalize ke [0, 255] (jika belum)
    if pixels.max() <= 1.0:
        pixels = pixels * 255.0

    # 5. Inferensi
    input_tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        output = model(input_tensor)
    mask = output.squeeze().numpy()  # (H, W) float32 [0, 1]

    # 6. Postprocess mask
    mask_binary = postprocess_mask2d(mask, min_area=10, closing_radius=2)  # (H, W) uint8 {0, 1}
    has_segmentation = bool(np.any(mask_binary == 1))

    return pixels, mask_binary, has_segmentation


def _build_row(image_path: Path, input_folder: Path) -> dict:
    """Jalankan inferensi satu gambar -> satu baris record sesuai IMAGE_SCHEMA."""
    raw_pixels, mask, has_seg = infer_2d_image(image_path)
    return {
        "filename": relative_name(image_path, input_folder),
        # astype memastikan tipe cocok dengan skema (float32 / uint8).
        "raw_pixels": raw_pixels.astype(np.float32, copy=False).ravel().tolist(),
        "mask": mask.astype(np.uint8, copy=False).ravel().tolist(),
        "has_segmentation": bool(has_seg),
        "shape_h": int(raw_pixels.shape[0]),
        "shape_w": int(raw_pixels.shape[1]),
    }


def _save_atomic(records: list[dict], output_parquet: Path) -> None:
    """Bangun Table dari records lalu tulis + validasi secara atomik."""
    table = pa.Table.from_pylist(records, schema=IMAGE_SCHEMA)
    atomic_write_table(table, output_parquet)
    validate_parquet(output_parquet, expected_rows=len(records))


# --- PROSES SEMUA GAMBAR DI FOLDER ---
def process_all_images_to_parquet(input_folder: Path, output_parquet: Path):
    """
    Proses semua file gambar di `input_folder` (rekursif) → simpan ke Parquet.

    Strategi anti-korupsi: kumpulkan baris di memori, tulis sekali di akhir
    secara atomik. Checkpoint atomik berkala (CHECKPOINT_EVERY) menjaga
    progres tanpa pernah meninggalkan file setengah jadi.
    """
    input_folder = Path(input_folder)
    output_parquet = Path(output_parquet)

    if not input_folder.exists():
        raise FileNotFoundError(f"Folder tidak ada: {input_folder.resolve()}")

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    records: list[dict] = []
    count = 0
    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            if not file.lower().endswith(image_extensions):
                continue
            image_path = Path(root) / file
            try:
                records.append(_build_row(image_path, input_folder))
                count += 1
                print(f"✅ Processed: {image_path} (total {count})")
                # Checkpoint atomik berkala (tiap checkpoint = file valid utuh).
                if CHECKPOINT_EVERY and count % CHECKPOINT_EVERY == 0:
                    _save_atomic(records, output_parquet)
                    print(f"💾 Checkpoint atomik tersimpan ({count} baris).")
            except Exception as e:
                print(f"❌ Error {image_path}: {e}")

    # Tulis final sekali secara atomik + validasi (deteksi korup lebih awal).
    if not records:
        print("⚠️ Tidak ada gambar yang berhasil diproses. Menulis Parquet kosong (skema tetap).")
    _save_atomic(records, output_parquet)
    print(f"📁 Saved to: {output_parquet.resolve()} ({count} files) — tervalidasi OK ✅")


# --- MAIN ---
if __name__ == "__main__":
    INPUT_FOLDER = Path("all_data_gambar")  # Folder berisi gambar
    OUTPUT_PARQUET = Path("inference_2d_image_results.parquet")

    process_all_images_to_parquet(INPUT_FOLDER, OUTPUT_PARQUET)
