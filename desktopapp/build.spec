# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all

# =========================================================
# Project paths
# =========================================================

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# Optional model path
model_path = os.path.join(project_root, "best_unet.pt")

# =========================================================
# Collect PyQt6 dependencies
# =========================================================

datas_qt, binaries_qt, hiddenimports_qt = collect_all("PyQt6")

# =========================================================
# Application datas
# =========================================================

datas = [
    (os.path.join(project_root, "desktopapp"), "desktopapp"),
    (os.path.join(project_root, "webapp"), "webapp"),
    (os.path.join(project_root, "unet_segmentation"), "unet_segmentation"),
    (os.path.join(project_root, "infer_dicom_unet.py"), "."),
]

# Optional model bundle
# datas.append((model_path, "."))

# Add PyQt6 collected data
datas += datas_qt

# =========================================================
# Hidden imports
# =========================================================

hiddenimports = hiddenimports_qt + [
    # PyQt6
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtNetwork",

    # Uncomment ONLY if needed
    # QtWebEngine is fragile on Wine/Winlator
    # "PyQt6.QtWebEngineCore",
    # "PyQt6.QtWebEngineWidgets",

    # AI / scientific stack
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

    # Imaging
    "PIL",
    "PIL.Image",

    # Plotting
    "matplotlib",
    "matplotlib.pyplot",

    # Flask / werkzeug
    "werkzeug",
    "werkzeug.datastructures",
    "werkzeug.utils",

    # Misc
    "patoolib",
    "json",
    "tempfile",
    "shutil",
    "uuid",
    "pathlib",
]

# =========================================================
# Build mode
# =========================================================

onefile = os.environ.get(
    "PYINSTALLER_ONEFILE",
    "False"
).lower() == "true"

# =========================================================
# Analysis
# =========================================================

a = Analysis(
    [os.path.join(SPECPATH, "main.py")],
    pathex=[project_root, SPECPATH],

    # IMPORTANT:
    binaries=binaries_qt,

    datas=datas,
    hiddenimports=hiddenimports,

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[],

    noarchive=False,
    optimize=0,
)

# =========================================================
# PYZ
# =========================================================

pyz = PYZ(a.pure)

# =========================================================
# Common EXE options
# =========================================================

exe_options = dict(
    name="AcuteStrokeSegmentation",

    debug=False,
    bootloader_ignore_signals=False,

    strip=False,

    # VERY IMPORTANT FOR WINE/WINLATOR
    upx=False,

    # Enable console for debugging
    console=True,

    disable_windowed_traceback=False,

    argv_emulation=False,
    target_arch=None,

    codesign_identity=None,
    entitlements_file=None,

    icon=(
        os.path.join(SPECPATH, "resources", "icon.ico")
        if os.path.exists(
            os.path.join(SPECPATH, "resources", "icon.ico")
        )
        else None
    ),
)

# =========================================================
# ONEFILE BUILD
# =========================================================

if onefile:

    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        **exe_options
    )

# =========================================================
# ONEDIR BUILD
# =========================================================

else:

    exe = EXE(
        pyz,
        a.scripts,

        [],  # binaries handled by COLLECT
        [],
        [],

        **exe_options
    )

    coll = COLLECT(
        exe,

        a.binaries,
        a.zipfiles,
        a.datas,

        strip=False,

        # IMPORTANT
        upx=False,

        upx_exclude=[],

        name="AcuteStrokeSegmentation",
    )
