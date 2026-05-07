# Acute Ischemic Stroke — Desktop App

PyQt6 desktop application for CT stroke segmentation using U-Net, with 3D visualization.

## Features

- **DICOM Series Prediction** — Upload ZIP archive containing DICOM CT series
- **Single Image Prediction** — Upload JPG/PNG/BMP images for 2D segmentation
- **3D Visualization** — Multiple viewers:
  - **Three.js** — Mesh rendering (CT surface + lesion overlay)
  - **Papaya Viewer** — DICOM/NIfTI slice viewer with overlay
  - **VTK.js** — Volume rendering (future enhancement)
- **Metrics Display** — Lesion volume, voxel count, spacing
- **Save Results** — Export all prediction files to any directory

## Requirements

See `requirements.txt`. Key dependencies:
- PyQt6 + PyQt6-WebEngine
- PyTorch + segmentation-models-pytorch
- pydicom, nibabel, scikit-image
- matplotlib, Pillow

## Running (Development)

```bash
pip install -r requirements.txt
python main.py
```

## Building .exe (Windows)

The GitHub Actions workflow (`.github/workflows/build-desktop.yml`) automatically builds the Windows executable on push to `main`.

Manual build with PyInstaller:

```bash
cd desktopapp
pyinstaller build.spec --clean --noconfirm
```

Output will be in `dist/AcuteStrokeSegmentation/`.

## Model File

Place `best_unet.pt` in the project root directory. The desktop app expects the model at:
```
Acute-ischemic-stroke-dataset-image-segmentation/best_unet.pt
```

## Project Structure

```
desktopapp/
├── main.py              # Entry point
├── requirements.txt     # Python dependencies
├── build.spec          # PyInstaller spec
├── gui/
│   ├── __init__.py
│   ├── main_window.py  # PyQt6 main window
│   └── workers.py      # QThread workers for inference
├── backend/
│   ├── __init__.py
│   └── inference.py    # Inference backend (wraps webapp code)
├── viewer/
│   ├── __init__.py
│   └── html_viewer.py  # 3D HTML visualization builder
└── resources/          # Icons, assets
```
