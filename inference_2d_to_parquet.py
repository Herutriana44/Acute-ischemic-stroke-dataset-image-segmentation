 """
  Inference 2D single DICOM → Parquet (raw HU + mask + has_segmentation)
  Menggunakan pipeline dari unet_segmentation/dicom_pipeline.py
  """

  import os
  from pathlib import Path
  import numpy as np
  import pandas as pd
  import pydicom
  import torch
  from unet_segmentation.dicom_pipeline import load_dicom_series, window_hu, postprocess_mask2d

  # --- LOAD MODEL (Sesuaikan dengan model kamu) ---
  # Contoh: Load model U-Net yang sudah dilatih
  # Ganti path ini dengan path model kamu yang sebenarnya
  MODEL_PATH = "unet_segmentation/models/unet_2d.pth"  # <--- GANTI INI

  def load_model():
      """Load model U-Net 2D (contoh placeholder)."""
      # Contoh arsitektur sederhana (ganti dengan model asli kamu)
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
  def infer_2d_dicom(dicom_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
      """
      Args:
          dicom_path: Path ke file DICOM (.dcm)
      Returns:
          (raw_hu, segmentation_mask, has_segmentation)
      """
      # 1. Load DICOM (single slice)
      slices = load_dicom_series(dicom_path.parent)  # Load semua slice di folder
      slice_data = [s for s in slices if s.path.name == dicom_path.name]
      if not slice_data:
          raise FileNotFoundError(f"DICOM {dicom_path} tidak ditemukan di series.")

      slice_ = slice_data[0]
      hu = slice_.hu  # (H, W) float32

      # 2. Windowing (opsional, sesuaikan dengan kebutuhan)
      hu_windowed = window_hu(hu, center=40, width=80)  # Window untuk brain CT
      hu_windowed = (hu_windowed * 255).astype(np.float32)  # Normalize ke [0, 255]

      # 3. Inferensi
      input_tensor = torch.from_numpy(hu_windowed).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
      with torch.no_grad():
          output = model(input_tensor)
      mask = output.squeeze().numpy()  # (H, W) float32 [0, 1]

      # 4. Postprocess mask
      mask_binary = postprocess_mask2d(mask, min_area=10, closing_radius=2)  # (H, W) uint8 {0, 1}
      has_segmentation = bool(np.any(mask_binary == 1))

      return hu, mask_binary, has_segmentation

  # --- PROSES SEMUA DICOM DI FOLDER ---
  def process_all_dicom_to_parquet(input_folder: Path, output_parquet: Path):
      """
      Proses semua file DICOM di `input_folder` (rekursif) → simpan ke Parquet.
      """
      results = []
      input_folder = Path(input_folder)

      for root, _, files in os.walk(input_folder):
          for file in files:
              if file.lower().endswith(('.dcm', '.dicom')):
                  dicom_path = Path(root) / file
                  try:
                      raw_hu, mask, has_seg = infer_2d_dicom(dicom_path)
                      results.append({
                          "filename": str(dicom_path.relative_to(input_folder)),
                          "raw_hu": raw_hu.flatten().tolist(),  # Flatten untuk simpan di Parquet
                          "mask": mask.flatten().tolist(),      # Flatten
                          "has_segmentation": has_seg,
                          "shape_h": raw_hu.shape[0],
                          "shape_w": raw_hu.shape[1],
                      })
                      print(f"✅  Processed: {dicom_path}")
                  except Exception as e:
                      print(f"❌  Error {dicom_path}: {e}")

      # Simpan ke Parquet
      df = pd.DataFrame(results)
      df.to_parquet(output_parquet)
      print(f"📁 Saved to: {output_parquet} ({len(df)} files)")

  # --- MAIN ---
  if __name__ == "__main__":
      INPUT_FOLDER = Path("all_dicom")  # Folder berisi DICOM
      OUTPUT_PARQUET = Path("inference_2d_results.parquet")

      process_all_dicom_to_parquet(INPUT_FOLDER, OUTPUT_PARQUET)