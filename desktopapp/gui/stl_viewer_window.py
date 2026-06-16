"""STL Viewer Window — PyQt6 + VTK native.

Layout inspired by dicom2stl.io:
  ┌─────────────────────────────────────────────────────────┐
  │  [Return]  Threshold: ──●──  Clipping: ◉  [Convert]    │  ← top bar
  ├────────────────────┬────────────────────────────────────┤
  │                    │                              [Clip]│
  │   preview (small)  │       main view (large)      │    │
  │                    │                              │    │
  ├────────────────────┴───────────────────────────── │ ───┤
  │           [Close]          [Wireframe]             │    │
  └─────────────────────────────────────────────────────────┘

Controls:
  • Threshold slider  — re-render from saved numpy volumes (if available)
    or just controls brain mesh opacity as a visual proxy
  • Clipping toggle + vertical slider — vtkClipPolyData plane along Z axis
  • Wireframe button  — toggle solid/wireframe rendering
  • Background gradient — light-blue sky (#a8d4e6 → #d6eaf8), matching ref
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPalette, QFont
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
    QSlider,
    QSplitter,
    QCheckBox,
)

# VTK
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkLight,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkClipPolyData
from vtkmodules.vtkCommonDataModel import vtkPlane
import vtkmodules.vtkRenderingOpenGL2   # noqa: F401
import vtkmodules.vtkInteractionStyle   # noqa: F401
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_BRAIN_RGB    = (0.780, 0.780, 0.760)   # warm white-gray, close to ref image
_LESION_RGB   = (0.918, 0.345, 0.047)   # orange
_BG_TOP       = (0.659, 0.831, 0.902)   # #a8d4e6  sky-blue top
_BG_BTM       = (0.839, 0.918, 0.973)   # #d6eaf8  light bottom
_TOOLBAR_CSS  = (
    "background:#e8f4fb;"
    "border-bottom:1px solid #b0c8d8;"
)
_BTN_CSS = (
    "QPushButton {"
    "  background:#4a9cc7; color:white; border:none;"
    "  border-radius:4px; padding:4px 14px; font-size:12px;"
    "}"
    "QPushButton:hover { background:#3a8ab5; }"
    "QPushButton:pressed { background:#2d7a9f; }"
)
_BTN_FLAT_CSS = (
    "QPushButton {"
    "  background:#d0e8f5; color:#1a3a50; border:1px solid #9ac0d8;"
    "  border-radius:4px; padding:4px 14px; font-size:12px;"
    "}"
    "QPushButton:hover { background:#b8d8ee; }"
    "QPushButton:checked { background:#4a9cc7; color:white; }"
)
_SLIDER_H_CSS = (
    "QSlider::groove:horizontal {"
    "  background:#b0c8d8; height:4px; border-radius:2px;"
    "}"
    "QSlider::handle:horizontal {"
    "  background:#2a7ab0; width:14px; height:14px;"
    "  margin:-5px 0; border-radius:7px;"
    "}"
    "QSlider::sub-page:horizontal { background:#2a7ab0; border-radius:2px; }"
)
_SLIDER_V_CSS = (
    "QSlider::groove:vertical {"
    "  background:#b0c8d8; width:4px; border-radius:2px;"
    "}"
    "QSlider::handle:vertical {"
    "  background:#2a7ab0; width:14px; height:14px;"
    "  margin:0 -5px; border-radius:7px;"
    "}"
    "QSlider::sub-page:vertical { background:#2a7ab0; border-radius:2px; }"
)


# ---------------------------------------------------------------------------
# Helper: create a styled toggle button (checkable)
# ---------------------------------------------------------------------------
def _make_toggle(label: str) -> QPushButton:
    btn = QPushButton(label)
    btn.setCheckable(True)
    btn.setStyleSheet(_BTN_FLAT_CSS)
    return btn


# ---------------------------------------------------------------------------
# _VtkPanel — one self-contained VTK render panel
# ---------------------------------------------------------------------------
class _VtkPanel(QWidget):
    """A QWidget wrapping a single VTK renderer with sky-blue gradient background."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._brain_actor:  vtkActor | None = None
        self._lesion_actor: vtkActor | None = None

        # Clip state
        self._clip_enabled   = False
        self._clip_value     = 0.5           # normalised 0-1
        self._clip_plane     = vtkPlane()
        self._clip_filters:  list[vtkClipPolyData] = []   # one per actor
        self._clip_actors:   list[vtkActor]         = []

        # Wireframe state
        self._wireframe = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._vtk_widget = QVTKRenderWindowInteractor(self)
        self._vtk_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._vtk_widget)

        self._renderer = vtkRenderer()
        self._renderer.GradientBackgroundOn()
        self._renderer.SetBackground(*_BG_BTM)
        self._renderer.SetBackground2(*_BG_TOP)

        ren_win: vtkRenderWindow = self._vtk_widget.GetRenderWindow()
        ren_win.AddRenderer(self._renderer)

        interactor: vtkRenderWindowInteractor = ren_win.GetInteractor()
        style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        self._setup_lights()
        self._vtk_widget.Initialize()

    # ── Lighting ──────────────────────────────────────────────────────
    def _setup_lights(self) -> None:
        self._renderer.RemoveAllLights()

        key = vtkLight()
        key.SetLightTypeToSceneLight()
        key.SetPosition(1, 1, 1)
        key.SetIntensity(0.85)
        key.SetColor(1, 1, 1)
        self._renderer.AddLight(key)

        fill = vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(-1, -0.5, 0.5)
        fill.SetIntensity(0.4)
        fill.SetColor(0.8, 0.9, 1.0)
        self._renderer.AddLight(fill)

    # ── Actor management ──────────────────────────────────────────────
    def clear_actors(self) -> None:
        for a in list(self._clip_actors):
            self._renderer.RemoveActor(a)
        self._clip_actors.clear()
        self._clip_filters.clear()

        if self._brain_actor:
            self._renderer.RemoveActor(self._brain_actor)
            self._brain_actor = None
        if self._lesion_actor:
            self._renderer.RemoveActor(self._lesion_actor)
            self._lesion_actor = None

    def set_actors(
        self,
        brain_actor:  vtkActor | None,
        lesion_actor: vtkActor | None,
    ) -> None:
        self.clear_actors()
        self._brain_actor  = brain_actor
        self._lesion_actor = lesion_actor

        for actor in (brain_actor, lesion_actor):
            if actor:
                self._renderer.AddActor(actor)

        self._apply_wireframe()
        self._apply_clip()
        self._renderer.ResetCamera()
        self.render()

    # ── Rendering ─────────────────────────────────────────────────────
    def render(self) -> None:
        self._vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        self._renderer.ResetCamera()
        self.render()

    # ── Wireframe ─────────────────────────────────────────────────────
    def set_wireframe(self, enabled: bool) -> None:
        self._wireframe = enabled
        self._apply_wireframe()
        self.render()

    def _apply_wireframe(self) -> None:
        for actor in (self._brain_actor, self._lesion_actor):
            if actor:
                prop = actor.GetProperty()
                if self._wireframe:
                    prop.SetRepresentationToWireframe()
                    prop.SetLineWidth(1.0)
                else:
                    prop.SetRepresentationToSurface()

    # ── Brain opacity ─────────────────────────────────────────────────
    def set_brain_opacity(self, opacity: float) -> None:
        if self._brain_actor:
            self._brain_actor.GetProperty().SetOpacity(opacity)
            self.render()

    # ── Visibility ────────────────────────────────────────────────────
    def set_brain_visible(self, visible: bool) -> None:
        if self._brain_actor:
            self._brain_actor.SetVisibility(visible)
            self.render()

    def set_lesion_visible(self, visible: bool) -> None:
        if self._lesion_actor:
            self._lesion_actor.SetVisibility(visible)
            self.render()

    # ── Clip ──────────────────────────────────────────────────────────
    def set_clip_enabled(self, enabled: bool) -> None:
        self._clip_enabled = enabled
        self._apply_clip()
        self.render()

    def set_clip_value(self, value: float) -> None:
        """value: 0.0 = bottom, 1.0 = top of bounding box."""
        self._clip_value = float(value)
        if self._clip_enabled:
            self._apply_clip()
            self.render()

    def _apply_clip(self) -> None:
        """Remove old clip actors and rebuild if clip is enabled."""
        # Remove previous clip actors
        for a in self._clip_actors:
            self._renderer.RemoveActor(a)
        self._clip_actors.clear()
        self._clip_filters.clear()

        if not self._clip_enabled:
            # Restore original actors visibility
            for actor in (self._brain_actor, self._lesion_actor):
                if actor:
                    actor.SetVisibility(True)
            return

        # Hide original actors while showing clipped versions
        for actor in (self._brain_actor, self._lesion_actor):
            if actor:
                actor.SetVisibility(False)

        # Determine Z range from bounding box
        bounds = self._renderer.ComputeVisiblePropBounds()
        z_min = bounds[4]
        z_max = bounds[5]
        if z_max <= z_min:
            return

        z_cut = z_min + self._clip_value * (z_max - z_min)

        self._clip_plane.SetOrigin(0, 0, z_cut)
        self._clip_plane.SetNormal(0, 0, -1)  # keep below the plane

        for actor in (self._brain_actor, self._lesion_actor):
            if actor is None:
                continue
            mapper = actor.GetMapper()
            if mapper is None:
                continue

            clipper = vtkClipPolyData()
            clipper.SetInputConnection(mapper.GetInputConnection(0, 0))
            clipper.SetClipFunction(self._clip_plane)
            clipper.InsideOutOn()
            clipper.Update()

            clip_mapper = vtkPolyDataMapper()
            clip_mapper.SetInputConnection(clipper.GetOutputPort())
            clip_mapper.ScalarVisibilityOff()

            clip_actor = vtkActor()
            clip_actor.SetMapper(clip_mapper)
            clip_actor.GetProperty().DeepCopy(actor.GetProperty())

            self._renderer.AddActor(clip_actor)
            self._clip_actors.append(clip_actor)
            self._clip_filters.append(clipper)

    def finalize(self) -> None:
        self._vtk_widget.GetRenderWindow().Finalize()
        self._vtk_widget.close()


# ---------------------------------------------------------------------------
# STLViewerWindow
# ---------------------------------------------------------------------------
class STLViewerWindow(QMainWindow):
    """DICOM→STL viewer matching the dicom2stl.io layout.

    Usage::

        win = STLViewerWindow(brain_stl=Path("brain.stl"),
                              lesion_stl=Path("lesion.stl"))
        win.show()
    """

    def __init__(
        self,
        brain_stl:  Path | None = None,
        lesion_stl: Path | None = None,
        parent:     QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("STL 3D Viewer — Acute Ischemic Stroke")
        self.resize(1200, 760)

        self._brain_path:  Path | None = brain_stl
        self._lesion_path: Path | None = lesion_stl

        # Shared actor sources (re-read once, shared between panels)
        self._brain_actor_main:    vtkActor | None = None
        self._lesion_actor_main:   vtkActor | None = None
        self._brain_actor_prev:    vtkActor | None = None
        self._lesion_actor_prev:   vtkActor | None = None

        self._build_ui()
        self._setup_panels()

        if brain_stl or lesion_stl:
            self._load_meshes(brain_stl, lesion_stl)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        central.setStyleSheet("background:#daeef8;")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top toolbar ───────────────────────────────────────────────
        root.addWidget(self._build_toolbar())

        # ── Main area: preview (left) + main viewer (right) + clip bar ─
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Two-panel splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(4)
        self._splitter.setStyleSheet("QSplitter::handle { background:#9ac4d8; }")

        # Preview panel (small, left)
        self._panel_preview = _VtkPanel()
        self._panel_preview.setMinimumWidth(200)
        self._splitter.addWidget(self._panel_preview)

        # Main panel (large, right)
        self._panel_main = _VtkPanel()
        self._splitter.addWidget(self._panel_main)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)

        body.addWidget(self._splitter, stretch=1)

        # Clipping vertical slider (far right, only visible when clipping on)
        self._clip_bar = self._build_clip_bar()
        body.addWidget(self._clip_bar)

        root.addLayout(body, stretch=1)

        # ── Bottom bar ────────────────────────────────────────────────
        root.addWidget(self._build_bottom_bar())

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(_TOOLBAR_CSS)

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(12)

        # Return / load buttons
        self._btn_load = QPushButton("Load STL…")
        self._btn_load.setStyleSheet(_BTN_FLAT_CSS)
        self._btn_load.clicked.connect(self._on_load_any)
        lay.addWidget(self._btn_load)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("border:none; background:#b0c8d8;")
        sep.setFixedWidth(1)
        lay.addWidget(sep)

        # Threshold label + slider
        lay.addWidget(QLabel("Threshold:"))
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(30)
        self._threshold_slider.setFixedWidth(160)
        self._threshold_slider.setStyleSheet(_SLIDER_H_CSS)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        lay.addWidget(self._threshold_slider)
        self._threshold_label = QLabel("30%")
        self._threshold_label.setFixedWidth(36)
        lay.addWidget(self._threshold_label)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("border:none; background:#b0c8d8;")
        sep2.setFixedWidth(1)
        lay.addWidget(sep2)

        # Clipping toggle
        lay.addWidget(QLabel("Clipping:"))
        self._btn_clipping = _make_toggle("OFF")
        self._btn_clipping.setFixedWidth(52)
        self._btn_clipping.toggled.connect(self._on_clipping_toggled)
        lay.addWidget(self._btn_clipping)

        lay.addStretch()

        # Export button
        self._btn_export = QPushButton("Export")
        self._btn_export.setStyleSheet(_BTN_CSS)
        self._btn_export.clicked.connect(self._on_export)
        lay.addWidget(self._btn_export)

        return bar

    def _build_clip_bar(self) -> QWidget:
        """Vertical clipping slider on the right edge."""
        bar = QWidget()
        bar.setFixedWidth(36)
        bar.setStyleSheet("background:#c8e4f2;")
        bar.setVisible(False)   # hidden until clipping is enabled

        lay = QVBoxLayout(bar)
        lay.setContentsMargins(4, 8, 4, 8)
        lay.setSpacing(4)

        lbl = QLabel("Clip")
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setStyleSheet("color:#1a3a50; font-size:9px;")
        lay.addWidget(lbl)

        self._clip_slider = QSlider(Qt.Orientation.Vertical)
        self._clip_slider.setRange(0, 100)
        self._clip_slider.setValue(100)          # 100 = no clipping (full mesh)
        self._clip_slider.setInvertedAppearance(True)   # drag down to clip from top
        self._clip_slider.setStyleSheet(_SLIDER_V_CSS)
        self._clip_slider.valueChanged.connect(self._on_clip_value_changed)
        lay.addWidget(self._clip_slider, stretch=1)

        return bar

    def _build_bottom_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(46)
        bar.setStyleSheet(
            "background:#d0e8f5; border-top:1px solid #b0c8d8;"
        )

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(10)

        # Visibility checkboxes (compact, left-aligned)
        self._chk_brain = QCheckBox("Brain")
        self._chk_brain.setChecked(True)
        self._chk_brain.setStyleSheet("QCheckBox{color:#1a3a50; font-size:12px;}")
        self._chk_brain.toggled.connect(self._on_toggle_brain)
        lay.addWidget(self._chk_brain)

        self._chk_lesion = QCheckBox("Lesion")
        self._chk_lesion.setChecked(True)
        self._chk_lesion.setStyleSheet("QCheckBox{color:#1a3a50; font-size:12px;}")
        self._chk_lesion.toggled.connect(self._on_toggle_lesion)
        lay.addWidget(self._chk_lesion)

        lay.addStretch()

        # Close
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(_BTN_FLAT_CSS)
        btn_close.clicked.connect(self.close)
        lay.addWidget(btn_close)

        # Wireframe
        self._btn_wireframe = _make_toggle("Wireframe")
        self._btn_wireframe.setStyleSheet(_BTN_FLAT_CSS)
        self._btn_wireframe.toggled.connect(self._on_wireframe_toggled)
        lay.addWidget(self._btn_wireframe)

        return bar

    # ------------------------------------------------------------------
    # VTK panel setup
    # ------------------------------------------------------------------
    def _setup_panels(self) -> None:
        """Nothing extra needed — panels are self-initialising."""
        pass

    # ------------------------------------------------------------------
    # STL loading
    # ------------------------------------------------------------------
    def _read_stl_actor(self, path: Path) -> vtkActor | None:
        if not path or not path.exists():
            return None

        reader = vtkSTLReader()
        reader.SetFileName(str(path))
        reader.Update()

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

    def _make_brain_actor(self) -> vtkActor | None:
        actor = self._read_stl_actor(self._brain_path)
        if actor:
            r, g, b = _BRAIN_RGB
            prop = actor.GetProperty()
            prop.SetColor(r, g, b)
            prop.SetOpacity(self._threshold_slider.value() / 100.0)
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.70)
            prop.SetSpecular(0.20)
            prop.SetSpecularPower(15)
        return actor

    def _make_lesion_actor(self) -> vtkActor | None:
        actor = self._read_stl_actor(self._lesion_path)
        if actor:
            r, g, b = _LESION_RGB
            prop = actor.GetProperty()
            prop.SetColor(r, g, b)
            prop.SetOpacity(1.0)
            prop.SetAmbient(0.15)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.35)
            prop.SetSpecularPower(25)
        return actor

    def _load_meshes(
        self,
        brain_stl:  Path | None,
        lesion_stl: Path | None,
    ) -> None:
        self._brain_path  = brain_stl
        self._lesion_path = lesion_stl

        # Build separate actor instances for each panel
        self._brain_actor_main   = self._make_brain_actor()
        self._lesion_actor_main  = self._make_lesion_actor()
        self._brain_actor_prev   = self._make_brain_actor()
        self._lesion_actor_prev  = self._make_lesion_actor()

        self._panel_main.set_actors(self._brain_actor_main, self._lesion_actor_main)
        self._panel_preview.set_actors(self._brain_actor_prev, self._lesion_actor_prev)

    # ------------------------------------------------------------------
    # Toolbar handlers
    # ------------------------------------------------------------------
    def _on_load_any(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load STL File", "", "STL Files (*.stl);;All Files (*)"
        )
        if path:
            self._load_meshes(Path(path), self._lesion_path)

    def _on_threshold_changed(self, value: int) -> None:
        """Threshold slider — controls brain surface opacity as visual proxy."""
        self._threshold_label.setText(f"{value}%")
        opacity = value / 100.0
        for actor in (self._brain_actor_main, self._brain_actor_prev):
            if actor:
                actor.GetProperty().SetOpacity(opacity)
        self._panel_main.render()
        self._panel_preview.render()

    def _on_clipping_toggled(self, checked: bool) -> None:
        self._btn_clipping.setText("ON" if checked else "OFF")
        self._clip_bar.setVisible(checked)
        clip_val = self._clip_slider.value() / 100.0
        self._panel_main.set_clip_enabled(checked)
        self._panel_preview.set_clip_enabled(checked)
        if checked:
            self._panel_main.set_clip_value(clip_val)
            self._panel_preview.set_clip_value(clip_val)

    def _on_clip_value_changed(self, value: int) -> None:
        clip_val = value / 100.0
        self._panel_main.set_clip_value(clip_val)
        self._panel_preview.set_clip_value(clip_val)

    def _on_export(self) -> None:
        """Export: copy both STL files to a user-selected directory."""
        dest_dir = QFileDialog.getExistingDirectory(self, "Export STL files to…")
        if not dest_dir:
            return
        import shutil
        exported = []
        for src in (self._brain_path, self._lesion_path):
            if src and src.exists():
                shutil.copy2(src, Path(dest_dir) / src.name)
                exported.append(src.name)
        from PyQt6.QtWidgets import QMessageBox
        if exported:
            QMessageBox.information(
                self, "Export", "Exported:\n" + "\n".join(exported)
            )

    # ------------------------------------------------------------------
    # Bottom bar handlers
    # ------------------------------------------------------------------
    def _on_toggle_brain(self, checked: bool) -> None:
        self._panel_main.set_brain_visible(checked)
        self._panel_preview.set_brain_visible(checked)

    def _on_toggle_lesion(self, checked: bool) -> None:
        self._panel_main.set_lesion_visible(checked)
        self._panel_preview.set_lesion_visible(checked)

    def _on_wireframe_toggled(self, checked: bool) -> None:
        self._btn_wireframe.setText("Solid" if checked else "Wireframe")
        self._panel_main.set_wireframe(checked)
        self._panel_preview.set_wireframe(checked)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:   # type: ignore[override]
        self._panel_main.finalize()
        self._panel_preview.finalize()
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
