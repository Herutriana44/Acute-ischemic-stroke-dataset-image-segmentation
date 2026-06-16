"""STL Viewer Window — PyQt6 + VTK native.

Displays two STL meshes side-by-side in a single VTK render window:
  • Brain surface  (blue-gray, semi-transparent)
  • Lesion surface (orange, solid)

Can be opened standalone or launched from MainWindow after inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QCheckBox,
    QSlider,
    QGroupBox,
)

# VTK imports — available via PyVista's dependency
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 — required for OpenGL backend
import vtkmodules.vtkInteractionStyle  # noqa: F401

# QVTKRenderWindowInteractor bridges VTK render window into a Qt widget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# ---------------------------------------------------------------------------
# Colour constants  (VTK uses 0.0–1.0 floats)
# ---------------------------------------------------------------------------
_BRAIN_RGB   = (0.737, 0.784, 0.855)   # #bcc8da  blue-gray
_LESION_RGB  = (0.918, 0.345, 0.047)   # #ea580c  orange
_BG_RGB      = (0.05,  0.05,  0.10)    # near-black


class STLViewerWindow(QMainWindow):
    """Standalone window that renders brain + lesion STL meshes with VTK.

    Usage (from MainWindow after inference)::

        self._stl_win = STLViewerWindow(
            brain_stl  = run_dir / "brain.stl",
            lesion_stl = run_dir / "lesion.stl",
        )
        self._stl_win.show()

    Or open empty and let the user load files manually::

        self._stl_win = STLViewerWindow()
        self._stl_win.show()
    """

    def __init__(
        self,
        brain_stl:  Optional[Path] = None,
        lesion_stl: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("STL 3D Viewer — Acute Ischemic Stroke")
        self.resize(1100, 750)

        self._brain_actor:  Optional[vtkActor] = None
        self._lesion_actor: Optional[vtkActor] = None

        self._build_ui()
        self._setup_vtk()

        if brain_stl or lesion_stl:
            self._load_meshes(brain_stl, lesion_stl)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── left: VTK render widget ───────────────────────────────────
        self._vtk_widget = QVTKRenderWindowInteractor(central)
        self._vtk_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._vtk_widget, stretch=1)

        # ── right: control panel ──────────────────────────────────────
        panel = QFrame()
        panel.setFixedWidth(210)
        panel.setStyleSheet(
            "QFrame { background:#0d1117; border-left:1px solid #30363d; }"
            "QLabel { color:#e6edf3; font-size:12px; }"
            "QPushButton { background:#21262d; color:#e6edf3; border:1px solid #30363d;"
            "  border-radius:4px; padding:5px 8px; }"
            "QPushButton:hover { background:#30363d; }"
            "QGroupBox { color:#8b949e; font-size:11px; border:1px solid #30363d;"
            "  border-radius:4px; margin-top:6px; padding-top:6px; }"
            "QCheckBox { color:#e6edf3; }"
            "QSlider::groove:horizontal { background:#21262d; height:4px; border-radius:2px; }"
            "QSlider::handle:horizontal { background:#58a6ff; width:12px; height:12px;"
            "  margin:-4px 0; border-radius:6px; }"
        )
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(10, 12, 10, 12)
        panel_lay.setSpacing(8)

        # Title
        title = QLabel("🧠  STL Viewer")
        title.setStyleSheet("color:#58a6ff; font-size:14px; font-weight:bold;")
        panel_lay.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("border:1px solid #30363d;")
        panel_lay.addWidget(sep)

        # Load buttons
        load_group = QGroupBox("Load Meshes")
        load_lay = QVBoxLayout(load_group)
        load_lay.setSpacing(6)

        self._btn_load_brain = QPushButton("Load Brain STL…")
        self._btn_load_brain.clicked.connect(self._on_load_brain)
        load_lay.addWidget(self._btn_load_brain)

        self._btn_load_lesion = QPushButton("Load Lesion STL…")
        self._btn_load_lesion.clicked.connect(self._on_load_lesion)
        load_lay.addWidget(self._btn_load_lesion)

        self._btn_load_any = QPushButton("Load Any STL…")
        self._btn_load_any.clicked.connect(self._on_load_any)
        load_lay.addWidget(self._btn_load_any)

        panel_lay.addWidget(load_group)

        # Visibility toggles
        vis_group = QGroupBox("Visibility")
        vis_lay = QVBoxLayout(vis_group)
        vis_lay.setSpacing(4)

        self._chk_brain = QCheckBox("Brain surface")
        self._chk_brain.setChecked(True)
        self._chk_brain.toggled.connect(self._on_toggle_brain)
        vis_lay.addWidget(self._chk_brain)

        self._chk_lesion = QCheckBox("Lesion")
        self._chk_lesion.setChecked(True)
        self._chk_lesion.toggled.connect(self._on_toggle_lesion)
        vis_lay.addWidget(self._chk_lesion)

        panel_lay.addWidget(vis_group)

        # Brain opacity
        opacity_group = QGroupBox("Brain Opacity")
        opacity_lay = QVBoxLayout(opacity_group)
        opacity_lay.setSpacing(4)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(30)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_lay.addWidget(self._opacity_slider)

        self._opacity_label = QLabel("30%")
        self._opacity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        opacity_lay.addWidget(self._opacity_label)

        panel_lay.addWidget(opacity_group)

        # Camera controls
        cam_group = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_group)
        cam_lay.setSpacing(6)

        self._btn_reset = QPushButton("Reset Camera")
        self._btn_reset.clicked.connect(self._on_reset_camera)
        cam_lay.addWidget(self._btn_reset)

        for label, direction in [
            ("Front  (+Y)", (0, 1, 0)),
            ("Top    (+Z)", (0, 0, 1)),
            ("Side   (+X)", (1, 0, 0)),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _chk, d=direction: self._set_camera_direction(*d)
            )
            cam_lay.addWidget(btn)

        panel_lay.addWidget(cam_group)

        # Info label
        panel_lay.addStretch()
        self._info_label = QLabel("No mesh loaded.")
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._info_label.setStyleSheet("color:#8b949e; font-size:10px;")
        panel_lay.addWidget(self._info_label)

        root.addWidget(panel)

    # ------------------------------------------------------------------
    # VTK setup
    # ------------------------------------------------------------------

    def _setup_vtk(self) -> None:
        """Initialise renderer, render window, and interactor."""
        self._renderer = vtkRenderer()
        self._renderer.SetBackground(*_BG_RGB)

        ren_win: vtkRenderWindow = self._vtk_widget.GetRenderWindow()
        ren_win.AddRenderer(self._renderer)

        interactor: vtkRenderWindowInteractor = ren_win.GetInteractor()
        style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        # Initialise the Qt-VTK bridge
        self._vtk_widget.Initialize()

    # ------------------------------------------------------------------
    # Mesh loading
    # ------------------------------------------------------------------

    def _read_stl(self, path: Path) -> Optional[vtkActor]:
        """Read an STL file and return a configured vtkActor, or None on error."""
        if not path.exists():
            return None

        reader = vtkSTLReader()
        reader.SetFileName(str(path))
        reader.Update()

        # Compute smooth normals for better shading
        normals = vtkPolyDataNormals()
        normals.SetInputConnection(reader.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtkActor()
        actor.SetMapper(mapper)
        return actor

    def _load_meshes(
        self,
        brain_stl:  Optional[Path],
        lesion_stl: Optional[Path],
    ) -> None:
        """Load brain and/or lesion STL files into the renderer."""
        # Remove existing actors
        if self._brain_actor:
            self._renderer.RemoveActor(self._brain_actor)
            self._brain_actor = None
        if self._lesion_actor:
            self._renderer.RemoveActor(self._lesion_actor)
            self._lesion_actor = None

        brain_loaded  = False
        lesion_loaded = False

        if brain_stl:
            actor = self._read_stl(brain_stl)
            if actor:
                r, g, b = _BRAIN_RGB
                actor.GetProperty().SetColor(r, g, b)
                actor.GetProperty().SetOpacity(self._opacity_slider.value() / 100.0)
                actor.GetProperty().SetAmbient(0.2)
                actor.GetProperty().SetDiffuse(0.7)
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(20)
                self._renderer.AddActor(actor)
                self._brain_actor = actor
                brain_loaded = True

        if lesion_stl:
            actor = self._read_stl(lesion_stl)
            if actor:
                r, g, b = _LESION_RGB
                actor.GetProperty().SetColor(r, g, b)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetAmbient(0.15)
                actor.GetProperty().SetDiffuse(0.75)
                actor.GetProperty().SetSpecular(0.4)
                actor.GetProperty().SetSpecularPower(30)
                self._renderer.AddActor(actor)
                self._lesion_actor = actor
                lesion_loaded = True

        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()
        self._update_info(brain_loaded, lesion_loaded, brain_stl, lesion_stl)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_load_brain(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Brain STL", "", "STL Files (*.stl);;All Files (*)"
        )
        if path:
            self._load_meshes(Path(path), None)

    def _on_load_lesion(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Lesion STL", "", "STL Files (*.stl);;All Files (*)"
        )
        if path:
            self._load_meshes(None, Path(path))

    def _on_load_any(self) -> None:
        """Load any STL and add it as a generic white actor."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load STL File", "", "STL Files (*.stl);;All Files (*)"
        )
        if not path:
            return
        actor = self._read_stl(Path(path))
        if actor:
            actor.GetProperty().SetColor(0.8, 0.8, 0.8)
            actor.GetProperty().SetOpacity(1.0)
            self._renderer.AddActor(actor)
            self._renderer.ResetCamera()
            self._vtk_widget.GetRenderWindow().Render()
            self._info_label.setText(f"Loaded:\n{Path(path).name}")

    def _on_toggle_brain(self, checked: bool) -> None:
        if self._brain_actor:
            self._brain_actor.SetVisibility(checked)
            self._vtk_widget.GetRenderWindow().Render()

    def _on_toggle_lesion(self, checked: bool) -> None:
        if self._lesion_actor:
            self._lesion_actor.SetVisibility(checked)
            self._vtk_widget.GetRenderWindow().Render()

    def _on_opacity_changed(self, value: int) -> None:
        self._opacity_label.setText(f"{value}%")
        if self._brain_actor:
            self._brain_actor.GetProperty().SetOpacity(value / 100.0)
            self._vtk_widget.GetRenderWindow().Render()

    def _on_reset_camera(self) -> None:
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def _set_camera_direction(self, vx: float, vy: float, vz: float) -> None:
        cam = self._renderer.GetActiveCamera()
        bounds = self._renderer.ComputeVisiblePropBounds()
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        dist = cam.GetDistance()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx + vx * dist, cy + vy * dist, cz + vz * dist)
        cam.SetViewUp(0, 0, 1) if vz < 0.9 else cam.SetViewUp(0, 1, 0)
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _update_info(
        self,
        brain_loaded: bool,
        lesion_loaded: bool,
        brain_path: Optional[Path],
        lesion_path: Optional[Path],
    ) -> None:
        lines = []
        if brain_loaded and brain_path:
            size_kb = brain_path.stat().st_size // 1024
            lines.append(f"🧠 Brain: {brain_path.name}\n   ({size_kb:,} KB)")
        elif brain_path:
            lines.append(f"⚠ Brain STL not found:\n   {brain_path.name}")

        if lesion_loaded and lesion_path:
            size_kb = lesion_path.stat().st_size // 1024
            lines.append(f"🔴 Lesion: {lesion_path.name}\n   ({size_kb:,} KB)")
        elif lesion_path:
            lines.append(f"⚠ Lesion STL not found:\n   {lesion_path.name}")

        if not lines:
            lines.append("No mesh loaded.")

        lines.append("\nControls:\nLeft drag — rotate\nRight drag — zoom\nMiddle drag — pan")
        self._info_label.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Properly shut down VTK interactor to avoid crash on close
        self._vtk_widget.GetRenderWindow().Finalize()
        self._vtk_widget.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = STLViewerWindow()
    win.show()
    sys.exit(app.exec())
