"""
Inference 2D image → Parquet (raw pixels + mask + has_segmentation)
Input: gambar dari folder all_data_gambar (jpg, png, dll)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from unet_segmentation.dicom_pipeline import postprocess_mask2d

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

# --- PROSES SEMUA GAMBAR DI FOLDER ---
def process_all_images_to_parquet(input_folder: Path, output_parquet: Path):
    """
    Proses semua file gambar di `input_folder` (rekursif) → simpan ke Parquet.
    """
    input_folder = Path(input_folder)
    output_parquet = Path(output_parquet)

    if not input_folder.exists():
        raise FileNotFoundError(f"Folder tidak ada: {input_folder.resolve()}")

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    count = 0
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_path = Path(root) / file
                try:
                    raw_pixels, mask, has_seg = infer_2d_image(image_path)
                    row = {
                        "filename": str(image_path.relative_to(input_folder)),
                        "raw_pixels": raw_pixels.flatten().tolist(),  # Flatten untuk simpan di Parquet
                        "mask": mask.flatten().tolist(),              # Flatten
                        "has_segmentation": has_seg,
                        "shape_h": raw_pixels.shape[0],
                        "shape_w": raw_pixels.shape[1],
                    }
                    # Append ke parquet yang sudah ada (atau buat baru)
                    df_new = pd.DataFrame([row])
                    if output_parquet.exists():
                        df_old = pd.read_parquet(output_parquet)
                        df_new = pd.concat([df_old, df_new], ignore_index=True)
                    df_new.to_parquet(output_parquet)
                    count += 1
                    print(f"✅ Processed & saved: {image_path} (total {count})")
                except Exception as e:
                    print(f"❌ Error {image_path}: {e}")

    print(f"📁 Saved to: {output_parquet.resolve()} ({count} files)")

# --- MAIN ---
if __name__ == "__main__":
    INPUT_FOLDER = Path("all_data_gambar")  # Folder berisi gambar
    OUTPUT_PARQUET = Path("inference_2d_image_results.parquet")

    process_all_images_to_parquet(INPUT_FOLDER, OUTPUT_PARQUET)
