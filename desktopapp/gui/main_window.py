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
from pyvistaqt import QtInteractor
from viewer.html_viewer import build_result_html

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

        # Right: Tabbed viewer (3D + Dashboard)
        self._tabs = QTabWidget()
        
        # Tab 1: PyVista QtInteractor
        self._viewer = QtInteractor(self)
        self._viewer.setMinimumSize(QSize(600, 500))
        self._tabs.addTab(self._viewer, "3D Mesh")
        
        # Tab 2: QWebEngineView for HTML/Papaya/VTK/Three.js
        self._browser = QWebEngineView()
        settings = self._browser.settings()
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(settings.WebAttribute.JavascriptEnabled, True)
        
        # Add Console Debugger
        self._browser.page().javaScriptConsoleMessage = self._handle_js_console
        
        self._tabs.addTab(self._browser, "Dashboard")
        
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

        # ── 1. PyVista 3D Mesh tab ────────────────────────────────────────
        self._update_pyvista_tab()

        # ── 2. HTML Dashboard tab ─────────────────────────────────────────
        self._update_html_tab()

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

    def _update_html_tab(self) -> None:
        """Build and load the HTML dashboard into QWebEngineView."""
        # Use a run-specific temp dir so stale files from previous runs
        # don't bleed through.
        run_id = self._result.get("run_id", "unknown") if self._result else "unknown"
        temp_dir = Path(tempfile.gettempdir()) / "stroke_viewer" / run_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            html_path = build_result_html(self._run_dir, self._result, temp_dir)
            self._browser.setUrl(QUrl.fromLocalFile(str(html_path)))
            self._log.append(f"Dashboard tab updated → {html_path}")
        except Exception as exc:
            self._log.append(f"Dashboard tab error: {exc}")

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

    def _handle_js_console(self, level, message, line, source_id):
        self._log.append(f"JS Console [{level}]: {message} (at {source_id}:{line})")

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
