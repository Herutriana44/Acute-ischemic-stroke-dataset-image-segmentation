# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Project root
project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# Model file (best_unet.pt) - optional, not bundled by default
model_path = os.path.join(project_root, "best_unet.pt")

# Collect data files
datas = []
datas.append((os.path.join(project_root, "desktopapp"), "desktopapp"))
datas.append((os.path.join(project_root, "webapp"), "webapp"))
datas.append((os.path.join(project_root, "unet_segmentation"), "unet_segmentation"))
datas.append((os.path.join(project_root, "infer_dicom_unet.py"), "infer_dicom_unet.py"))

# Include the model file if it exists (optional - user can provide their own)
# datas.append((model_path, "."))  # Uncomment to bundle model (large file ~100MB+)

# Hidden imports for PyQt6 and other modules
hiddenimports = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtNetwork",
    "torch",
    "torchvision",
    "numpy",
    "nibabel",
    "pydicom",
    "skimage",
    "skimage.measure",
    "skimage.morphology",
    "segmentation_models_pytorch",
    "albumentations",
    "PIL",
    "PIL.Image",
    "matplotlib",
    "matplotlib.pyplot",
    "werkzeug",
    "werkzeug.datastructures",
    "werkzeug.utils",
    "patoolib",
    "json",
    "tempfile",
    "shutil",
    "uuid",
    "pathlib",
]

# Analysis
a = Analysis(
    [os.path.join(SPECPATH, "main.py")],
    pathex=[project_root, SPECPATH],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="AcuteStrokeSegmentation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPECPATH, "resources", "icon.ico") if os.path.exists(os.path.join(SPECPATH, "resources", "icon.ico")) else None,
)
