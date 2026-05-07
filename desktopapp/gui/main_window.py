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
from PyQt6.QtWebEngineWidgets import QWebEngineView

from gui.workers import InferenceWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._run_dir: Path | None = None
        self._result: dict | None = None
        self._temp_dir: Path | None = None
        self._init_ui()

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

        # Right: QWebEngineView for 3D visualization
        self._viewer = QWebEngineView()
        self._viewer.setMinimumSize(QSize(600, 500))
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

    def _display_viewer(self) -> None:
        """Generate HTML and load it into the QWebEngineView."""
        from viewer.html_viewer import build_result_html

        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="stroke_app_"))
        html_path = build_result_html(self._run_dir, self._result, self._temp_dir)
        self._viewer.load(QUrl.fromLocalFile(str(html_path)))
        self._log.append(f"Viewer loaded: {html_path.name}")

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
            "Supports 3D visualization with VTK.js, Papaya, and Three.js.",
        )

    def closeEvent(self, event) -> None:
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        event.accept()
