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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from gui.workers import InferenceWorker


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

        # Right: PyVista QtInteractor for 3D visualization
        self._viewer = QtInteractor(self)
        self._viewer.setMinimumSize(QSize(600, 500))
        
        # Add a Fullscreen button overlay on the viewer
        self._btn_fullscreen = QPushButton("Fullscreen", self._viewer)
        self._btn_fullscreen.setStyleSheet("background-color: white; color: black; border: 1px solid gray;")
        self._btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        self._btn_fullscreen.move(10, 10)
        
        splitter.addWidget(self._viewer)
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

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode for the viewer."""
        if self._viewer.isFullScreen():
            self._viewer.showNormal()
            self._btn_fullscreen.setText("Fullscreen")
        else:
            self._viewer.showFullScreen()
            self._btn_fullscreen.setText("Exit Fullscreen")

    def _display_viewer(self) -> None:
        """Display results using PyVista with brain model and lesion mesh."""
        if not self._run_dir or not self._result:
            return

        # Clear existing
        self._viewer.clear()

        # Load Base Brain Model (assumed to be in 3D_model/Plastinated_Human_Brain/Plastinated_Human_Brain.gltf)
        # Note: You need to ensure the path to this model is accessible
        brain_model_path = Path("3D_model/Plastinated_Human_Brain/Plastinated_Human_Brain.gltf")
        if brain_model_path.exists():
            import pyvista as pv
            brain = pv.read(brain_model_path)
            self._viewer.add_mesh(brain, color="white", opacity=0.3, label="Brain")

        # Load lesion mesh
        mesh_path = self._run_dir / "lesion_mesh.ply"
        if mesh_path.exists():
            import pyvista as pv
            mesh = pv.read(mesh_path)
            self._viewer.add_mesh(mesh, color="red", opacity=0.8, label="Lesion")

        self._viewer.add_axes()
        self._viewer.add_legend()
        self._viewer.reset_camera()
        self._log.append("3D viewer updated with brain model and lesion.")

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
