"""PyQt6 main window for the Acute Ischemic Stroke desktop app."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QUrl
from PyQt6.QtGui import QAction, QFont, QIcon
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from pyvistaqt import QtInteractor
# from viewer.html_viewer import build_result_html

from gui.workers import InferenceWorker
from gui.dicom_viewer import DicomViewer


class _ConsolePage(QWebEnginePage):
    """QWebEnginePage subclass that forwards JS console messages to a callback."""

    def __init__(self, log_callback, parent=None):
        super().__init__(parent)
        self._log_callback = log_callback

    def javaScriptConsoleMessage(self, level, message, line_number, source_id):
        level_name = {
            QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel: "INFO",
            QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel: "WARN",
            QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel: "ERROR",
        }.get(level, str(level))
        self._log_callback(f"js: [{level_name}] {message}  ({source_id}:{line_number})")


import sys
import logging

class EmittingStream:
    """Redirect writes to a QTextEdit and optionally to logger."""
    def __init__(self, widget, log_func=None):
        self.widget = widget
        self.log_func = log_func
        self.buffer = ''
    def write(self, text):
        # Write to widget
        self.widget.moveCursor(self.widget.textCursor().End)
        self.widget.insertPlainText(text)
        self.widget.moveCursor(self.widget.textCursor().End)
        # Log if provided
        if self.log_func:
            self.log_func(text.rstrip())
    def flush(self):
        pass

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # Initialize UI first so _log widget exists
        self._init_ui()
        # Set up logging to file and redirect output to GUI log
        self._logger = logging.getLogger('desktopapp_gui')
        self._logger.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.FileHandler) for h in self._logger.handlers):
            file_handler = logging.FileHandler('desktopapp_gui.log')
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        # Redirect stdout/stderr to QTextEdit and logger
        self._log_stream = EmittingStream(self._log, self._logger.debug)
        sys.stdout = self._log_stream
        sys.stderr = self._log_stream
        self._run_dir: Path | None = None
        self._result: dict | None = None

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        self.setWindowTitle("Acute Ischemic Stroke — DICOM Segmentation")
        self.resize(1400, 900)

        # ---------- central widget ----------
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Toolbar-like bar
        top_bar = QHBoxLayout()
        self._btn_dicom = QPushButton("Load DICOM Archive")
        self._btn_dicom.clicked.connect(self._load_dicom)
        self._btn_image = QPushButton("Load Image (2D)")
        self._btn_image.clicked.connect(self._load_image)
        self._btn_view = QPushButton("View Results")
        self._btn_view.clicked.connect(self._view_results)
        self._btn_view.setEnabled(False)
        self._btn_save = QPushButton("Save Results")
        self._btn_save.clicked.connect(self._save_results)
        self._btn_save.setEnabled(False)

        top_bar.addWidget(self._btn_dicom)
        top_bar.addWidget(self._btn_image)
        top_bar.addSpacing(20)
        top_bar.addWidget(self._btn_view)
        top_bar.addWidget(self._btn_save)
        top_bar.addStretch()
        self._status_label = QLabel("Ready.")
        top_bar.addWidget(self._status_label)
        layout.addLayout(top_bar)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Splitter: left = log, right = viewer
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: metrics + log
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._metrics_label = QLabel("<h3>Metrics</h3><p>Load data to start.</p>")
        self._metrics_label.setWordWrap(True)
        left_layout.addWidget(self._metrics_label)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        left_layout.addWidget(QLabel("<b>Log</b>"))
        left_layout.addWidget(self._log)
        splitter.addWidget(left)

        # Right: Tabbed viewer (3D + Dashboard)
        self._tabs = QTabWidget()
        
        # Tab 1: PyVista QtInteractor
        self._viewer = QtInteractor(self)
        self._viewer.setMinimumSize(QSize(600, 500))
        self._tabs.addTab(self._viewer, "3D Mesh")
        
        # Tab 2: 2D Image Viewer
        self._image_viewer = QLabel("Load an image to see results.")
        self._image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_viewer.setMinimumSize(QSize(600, 500))
        self._tabs.addTab(self._image_viewer, "2D Result")
        
        # Tab 3: Native DICOM multi-planar viewer (Axial / Coronal / Sagittal)
        self._dicom_viewer = DicomViewer()
        self._tabs.addTab(self._dicom_viewer, "DICOM Viewer")
        
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # ---------- menu bar ----------
        self._create_menu()

    def _create_menu(self) -> None:
        menubar = QMenuBar(self)
        self.setMenuBar(menubar)

        file_menu = menubar.addMenu("&File")
        act_dicom = QAction("Load DICOM Archive…", self)
        act_dicom.triggered.connect(self._load_dicom)
        file_menu.addAction(act_dicom)
        act_image = QAction("Load Image (2D)…", self)
        act_image.triggered.connect(self._load_image)
        file_menu.addAction(act_image)
        file_menu.addSeparator()
        act_save = QAction("Save Results…", self)
        act_save.triggered.connect(self._save_results)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = menubar.addMenu("&View")
        act_results = QAction("View Results", self)
        act_results.triggered.connect(self._view_results)
        view_menu.addAction(act_results)

        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About…", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _load_dicom(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DICOM Archive",
            "",
            "Archives (*.zip *.rar *.tar *.tar.gz *.tgz *.7z);;All Files (*)",
        )
        if not file_path:
            return
        self._run_inference_dicom(Path(file_path))

    def _load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not file_path:
            return
        self._run_inference_image(Path(file_path))

    def _run_inference_dicom(self, archive_path: Path) -> None:
        self._log.append(f"Loading DICOM archive: {archive_path.name}")
        self._status_label.setText("Processing DICOM series…")
        self._progress.setVisible(True)
        self._btn_view.setEnabled(False)
        self._btn_save.setEnabled(False)

        self._worker = InferenceWorker("dicom", archive_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_inference_done)
        self._worker.error.connect(self._on_inference_error)
        self._worker.start()

    def _run_inference_image(self, image_path: Path) -> None:
        self._log.append(f"Loading image: {image_path.name}")
        self._status_label.setText("Processing image…")
        self._progress.setVisible(True)
        self._btn_view.setEnabled(False)
        self._btn_save.setEnabled(False)

        self._worker = InferenceWorker("image", image_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_inference_done)
        self._worker.error.connect(self._on_inference_error)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._log.append(msg)

    def _on_inference_done(self, run_dir: Path, result: dict) -> None:
        self._progress.setVisible(False)
        self._run_dir = run_dir
        self._result = result
        self._status_label.setText("Inference complete.")
        self._log.append("Inference complete.")
        self._btn_view.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._update_metrics()
        self._view_results()

    def _on_inference_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._status_label.setText("Error.")
        self._log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Inference Error", msg)

    def _update_metrics(self) -> None:
        if not self._result:
            return
        r = self._result
        html = "<h3>Prediction Metrics</h3>"
        html += f"<p><b>Run ID:</b> {r.get('run_id', '-')}</p>"
        if r.get("enable_3d"):
            html += f"<p><b>Slices:</b> {r.get('slices', '-')}</p>"
            html += f"<p><b>Resolution:</b> {r.get('shape_hw', '-')}</p>"
            html += f"<p><b>Spacing:</b> {r.get('spacing', '-')}</p>"
        html += f"<p><b>Lesion Voxels:</b> {r.get('lesion_voxels', 0)}</p>"
        html += f"<p><b>Lesion Volume:</b> {r.get('lesion_volume_mm3', 0)} mm³ ({r.get('lesion_volume_ml', 0)} mL)</p>"
        self._metrics_label.setText(html)

    def _view_results(self) -> None:
        if not self._run_dir or not self._result:
            return
        self._display_viewer()

    def _display_viewer(self) -> None:
        """Display results using both PyVista (3D Mesh tab) and HTML (Dashboard tab)."""
        if not self._run_dir or not self._result:
            return

        is_3d = self._result.get("enable_3d", False)

        # Tab visibility
        self._tabs.setTabVisible(0, is_3d) # 3D Mesh
        self._tabs.setTabVisible(2, is_3d) # DICOM Viewer

        if is_3d:
            # ── 1. PyVista 3D Mesh tab ────────────────────────────────────────
            self._update_pyvista_tab()
            # ── 3. Native DICOM multi-planar viewer tab ───────────────────────
            self._update_dicom_viewer_tab()
            self._tabs.setCurrentIndex(0)
        else:
            # ── 2. 2D Image Viewer tab ─────────────────────────────────────────
            self._update_2d_viewer_tab()
            self._tabs.setCurrentIndex(1)

    def _update_2d_viewer_tab(self) -> None:
        """Load 2D overlay result into the QLabel."""
        from PyQt6.QtGui import QPixmap
        overlay_path = self._run_dir / self._result.get("overlay_png", "overlay.png")
        if overlay_path.exists():
            pixmap = QPixmap(str(overlay_path))
            self._image_viewer.setPixmap(pixmap.scaled(self._image_viewer.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._log.append(f"2D Viewer: loaded {overlay_path.name}")
        else:
            self._image_viewer.setText("Overlay image not found.")
            self._log.append("2D Viewer error: overlay image not found.")

    def _update_pyvista_tab(self) -> None:
        """Load DICOM-derived OBJ meshes into the PyVista QtInteractor."""
        import pyvista as pv

        self._viewer.clear()
        loaded_any = False

        brain_obj_path = self._run_dir / "brain.obj"
        if brain_obj_path.exists():
            try:
                brain_mesh = pv.read(str(brain_obj_path))
                self._viewer.add_mesh(
                    brain_mesh,
                    color="#bcc8da",   # blue-gray, matches HTML viewer
                    opacity=0.30,
                    label="CT Brain Surface",
                    smooth_shading=True,
                )
                loaded_any = True
                self._log.append(
                    f"3D Mesh: loaded brain.obj "
                    f"({brain_mesh.n_points} pts, {brain_mesh.n_cells} faces)"
                )
            except Exception as exc:
                self._log.append(f"3D Mesh: could not load brain.obj — {exc}")

        lesion_obj_path = self._run_dir / "lesion.obj"
        if lesion_obj_path.exists():
            try:
                lesion_mesh = pv.read(str(lesion_obj_path))
                self._viewer.add_mesh(
                    lesion_mesh,
                    color="#ea580c",   # orange, matches HTML viewer
                    opacity=0.85,
                    label="Ischemic Lesion",
                    smooth_shading=True,
                )
                loaded_any = True
                self._log.append(
                    f"3D Mesh: loaded lesion.obj "
                    f"({lesion_mesh.n_points} pts, {lesion_mesh.n_cells} faces)"
                )
            except Exception as exc:
                self._log.append(f"3D Mesh: could not load lesion.obj — {exc}")

        if not loaded_any:
            self._log.append("3D Mesh: no OBJ files found (single-image mode or no lesion).")

        # Axes, legend, camera
        self._viewer.add_axes()
        if loaded_any:
            self._viewer.add_legend()
        self._viewer.reset_camera()
        self._viewer.render()

    # def _update_html_tab(self) -> None:
    #     """Build and load the HTML dashboard into QWebEngineView."""
    #     # Use a run-specific temp dir so stale files from previous runs
    #     # don't bleed through.
    #     run_id = self._result.get("run_id", "unknown") if self._result else "unknown"
    #     temp_dir = Path(tempfile.gettempdir()) / "stroke_viewer" / run_id
    #     temp_dir.mkdir(parents=True, exist_ok=True)
    #     try:
    #         html_path = build_result_html(self._run_dir, self._result, temp_dir)
    #         self._browser.setUrl(QUrl.fromLocalFile(str(html_path)))
    #         self._log.append(f"Dashboard tab updated → {html_path}")
    #     except Exception as exc:
    #         self._log.append(f"Dashboard tab error: {exc}")

    def _update_dicom_viewer_tab(self) -> None:
        """Load CT and mask volumes into the native DICOM multi-planar viewer."""
        if not self._run_dir or not self._result:
            return

        result = self._result

        # Only available for DICOM (3-D) runs
        if not result.get("enable_3d"):
            self._log.append("DICOM Viewer: not available for single-image mode.")
            self._dicom_viewer.clear()
            return

        try:
            import numpy as np

            # Prefer pre-saved numpy volumes (fastest)
            hu_npy = self._run_dir / "hu_volume.npy"
            mask_npy = self._run_dir / "mask_pred.npy"

            if hu_npy.exists() and mask_npy.exists():
                ct_vol = np.load(str(hu_npy))   # raw HU, shape (Z, Y, X)
                mask_vol = np.load(str(mask_npy))
                use_hu = True
                self._log.append("DICOM Viewer: loaded volumes from .npy cache.")
            else:
                # Fall back to windowed NIfTI
                import nibabel as nib
                ct_nii_name = result.get("ct_hu_nii") or result.get("ct_nii", "")
                mask_nii_name = result.get("mask_nii", "")
                ct_nii_path = self._run_dir / ct_nii_name if ct_nii_name else None
                mask_nii_path = self._run_dir / mask_nii_name if mask_nii_name else None

                if ct_nii_path is None or not ct_nii_path.exists():
                    self._log.append("DICOM Viewer: CT NIfTI not found.")
                    return

                ct_img = nib.load(str(ct_nii_path))
                ct_vol = np.asarray(ct_img.dataobj, dtype=np.float32)
                # NIfTI from inference is stored as (Y, X, Z) → transpose to (Z, Y, X)
                if ct_vol.ndim == 3:
                    ct_vol = ct_vol.transpose(2, 0, 1)

                mask_vol = None
                if mask_nii_path and mask_nii_path.exists():
                    mask_img = nib.load(str(mask_nii_path))
                    mask_vol = np.asarray(mask_img.dataobj, dtype=np.float32)
                    if mask_vol.ndim == 3:
                        mask_vol = mask_vol.transpose(2, 0, 1)

                use_hu = "hu" in ct_nii_name.lower()
                self._log.append("DICOM Viewer: loaded volumes from NIfTI.")

            spacing_raw = result.get("spacing", [1.0, 1.0, 1.0])
            # spacing from inference is [ps_row, ps_col, ps_z]
            spacing = (
                float(spacing_raw[2]),  # z
                float(spacing_raw[0]),  # y (row)
                float(spacing_raw[1]),  # x (col)
            )

            self._dicom_viewer.load_volumes(
                ct_vol, mask_vol, spacing=spacing, use_hu=use_hu
            )
            self._log.append(
                f"DICOM Viewer: loaded {ct_vol.shape} volume, "
                f"spacing={spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.3f} mm"
            )

        except Exception as exc:
            import traceback
            self._log.append(f"DICOM Viewer error: {exc}")
            traceback.print_exc()

    def _save_results(self) -> None:
        if not self._run_dir:
            return
        dest = QFileDialog.getExistingDirectory(self, "Save Results To")
        if not dest:
            return
        dest_path = Path(dest) / (self._run_dir.name or "results")
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(self._run_dir, dest_path)
        self._log.append(f"Results saved to: {dest_path}")
        QMessageBox.information(self, "Saved", f"Results saved to:\n{dest_path}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            "Acute Ischemic Stroke — DICOM Segmentation\n\n"
            "Desktop application for CT stroke segmentation using U-Net.\n"
            "Supports 3D visualization with PyVista.",
        )

    def closeEvent(self, event) -> None:
        event.accept()
