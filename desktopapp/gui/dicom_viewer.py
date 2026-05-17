"""Native DICOM/NIfTI slice viewer widget for the desktop app.

Provides a Papaya-like multi-planar reconstruction (MPR) viewer
directly in PyQt6 — no HTML/WebEngine required.

Layout (mirrors Papaya):
  ┌──────────────┬──────────────┐
  │   AXIAL      │   CORONAL    │
  │   (Z slice)  │   (Y slice)  │
  ├──────────────┼──────────────┤
  │   SAGITTAL   │   INFO PANEL │
  │   (X slice)  │              │
  └──────────────┴──────────────┘

Controls:
  • Sliders below each view to navigate slices
  • Window / Level spinboxes (HU windowing)
  • Overlay opacity slider
  • Crosshair lines linking all three views
  • Click on any view to jump to that position
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Colour-map helpers
# ---------------------------------------------------------------------------

def _gray_lut() -> np.ndarray:
    """256-entry grayscale LUT → shape (256, 3) uint8."""
    lut = np.arange(256, dtype=np.uint8)
    return np.stack([lut, lut, lut], axis=1)


def _hot_lut() -> np.ndarray:
    """256-entry 'hot' (black→red→yellow→white) LUT for lesion overlay."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    # 0 → transparent (handled separately), 1-85 black→red
    for i in range(1, 86):
        lut[i] = [int(i * 3), 0, 0]
    # 86-170 red→yellow
    for i in range(86, 171):
        lut[i] = [255, int((i - 86) * 3), 0]
    # 171-255 yellow→white
    for i in range(171, 256):
        lut[i] = [255, 255, int((i - 171) * 3)]
    return lut


# ---------------------------------------------------------------------------
# Single-plane canvas
# ---------------------------------------------------------------------------

class _SliceCanvas(QLabel):
    """QLabel that renders one CT slice + optional mask overlay + crosshairs.

    Emits ``clicked(x_frac, y_frac)`` with normalised [0,1] coordinates
    when the user clicks on the image.
    """

    clicked = pyqtSignal(float, float)  # (x_frac, y_frac) in image space

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._ct_slice: Optional[np.ndarray] = None    # 2-D float32 [0,1]
        self._mask_slice: Optional[np.ndarray] = None  # 2-D uint8 {0,1}
        self._cx: float = 0.5   # crosshair x (normalised)
        self._cy: float = 0.5   # crosshair y (normalised)
        self._overlay_alpha: float = 0.45
        self._show_crosshair: bool = True

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet("background: #0a0a1a; border: 1px solid #333;")
        self.setText(f"<span style='color:#555;'>{title}</span>")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_slices(
        self,
        ct: Optional[np.ndarray],
        mask: Optional[np.ndarray],
    ) -> None:
        """Update the displayed CT and mask slices (2-D arrays)."""
        self._ct_slice = ct
        self._mask_slice = mask
        self._render()

    def set_crosshair(self, cx: float, cy: float) -> None:
        """Set crosshair position (normalised 0-1)."""
        self._cx = float(np.clip(cx, 0.0, 1.0))
        self._cy = float(np.clip(cy, 0.0, 1.0))
        self._render()

    def set_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = float(np.clip(alpha, 0.0, 1.0))
        self._render()

    def set_show_crosshair(self, show: bool) -> None:
        self._show_crosshair = show
        self._render()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._ct_slice is None:
            return
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        # Account for letterboxing (QLabel centres the pixmap)
        pm = self.pixmap()
        if pm is None:
            return
        pw, ph = pm.width(), pm.height()
        ox = (w - pw) // 2
        oy = (h - ph) // 2
        px = event.position().x() - ox
        py = event.position().y() - oy
        if px < 0 or py < 0 or px >= pw or py >= ph:
            return
        self.clicked.emit(float(px / pw), float(py / ph))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._ct_slice is None:
            self.setText(f"<span style='color:#555;'>{self._title}</span>")
            return

        ct = self._ct_slice  # float32 [0,1], shape (H, W)
        H, W = ct.shape

        # CT → grayscale RGB
        ct_u8 = np.clip(ct * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1)  # (H, W, 3)

        # Overlay mask (orange-red)
        if self._mask_slice is not None and self._overlay_alpha > 0:
            mask = self._mask_slice.astype(bool)
            if mask.any():
                a = self._overlay_alpha
                overlay_r = np.full((H, W), 234, dtype=np.float32)
                overlay_g = np.full((H, W), 88, dtype=np.float32)
                overlay_b = np.full((H, W), 12, dtype=np.float32)
                rgb_f = rgb.astype(np.float32)
                rgb_f[mask, 0] = (1 - a) * rgb_f[mask, 0] + a * overlay_r[mask]
                rgb_f[mask, 1] = (1 - a) * rgb_f[mask, 1] + a * overlay_g[mask]
                rgb_f[mask, 2] = (1 - a) * rgb_f[mask, 2] + a * overlay_b[mask]
                rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

        # Convert to QImage → QPixmap
        rgb_c = np.ascontiguousarray(rgb)
        qimg = QImage(
            rgb_c.data, W, H, W * 3, QImage.Format.Format_RGB888
        )
        pm = QPixmap.fromImage(qimg)

        # Scale to fit label while keeping aspect ratio
        lw, lh = self.width(), self.height()
        if lw > 0 and lh > 0:
            pm = pm.scaled(
                lw, lh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Draw crosshair on top
        if self._show_crosshair:
            painter = QPainter(pm)
            pen = QPen(QColor(0, 255, 200, 180))
            pen.setWidth(1)
            painter.setPen(pen)
            cx_px = int(self._cx * pm.width())
            cy_px = int(self._cy * pm.height())
            painter.drawLine(cx_px, 0, cx_px, pm.height())
            painter.drawLine(0, cy_px, pm.width(), cy_px)
            painter.end()

        # Title overlay
        painter2 = QPainter(pm)
        painter2.setPen(QColor(200, 200, 200, 200))
        painter2.drawText(6, 16, self._title)
        painter2.end()

        self.setPixmap(pm)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render()


# ---------------------------------------------------------------------------
# Main viewer widget
# ---------------------------------------------------------------------------

class DicomViewer(QWidget):
    """Multi-planar DICOM viewer widget (Axial / Coronal / Sagittal).

    Usage::

        viewer = DicomViewer()
        viewer.load_volumes(ct_volume, mask_volume, spacing)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Volumes (Z, Y, X) float32 [0,1] and uint8 {0,1}
        self._ct: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (z, y, x) mm

        # Current slice indices
        self._iz: int = 0
        self._iy: int = 0
        self._ix: int = 0

        # Window / level (applied to raw HU or normalised)
        self._ww: float = 80.0   # window width
        self._wl: float = 40.0   # window level (centre)
        self._use_hu: bool = False  # True when raw HU is available

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_volumes(
        self,
        ct_volume: np.ndarray,
        mask_volume: Optional[np.ndarray],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_hu: bool = False,
    ) -> None:
        """Load CT and optional mask volumes.

        Parameters
        ----------
        ct_volume:
            3-D array (Z, Y, X).  If ``use_hu`` is True, values are raw HU;
            otherwise values should be in [0, 1] or [0, 255].
        mask_volume:
            3-D binary array (Z, Y, X) with values 0/1 or 0/255.
        spacing:
            Voxel spacing in mm as (z, y, x).
        use_hu:
            If True, apply HU windowing using the W/L spinboxes.
        """
        self._use_hu = use_hu
        self._spacing = spacing

        # Normalise CT to float32 [0, 1]
        ct = ct_volume.astype(np.float32)
        if use_hu:
            self._ct_raw = ct  # keep raw HU for re-windowing
            self._ct = self._apply_window(ct)
        else:
            # Assume [0, 255] or [0, 1]
            if ct.max() > 1.5:
                ct = ct / 255.0
            self._ct_raw = None
            self._ct = np.clip(ct, 0.0, 1.0)

        # Normalise mask to uint8 {0, 1}
        if mask_volume is not None:
            m = mask_volume.astype(np.float32)
            if m.max() > 1.5:
                m = m / 255.0
            self._mask = (m > 0.5).astype(np.uint8)
        else:
            self._mask = None

        # Reset slice positions to centre
        Z, Y, X = self._ct.shape
        self._iz = Z // 2
        self._iy = Y // 2
        self._ix = X // 2

        self._update_sliders()
        self._update_info()
        self._refresh_all()

    def clear(self) -> None:
        """Clear all volumes and reset the viewer."""
        self._ct = None
        self._mask = None
        self._ct_raw = None
        for canvas in (self._axial, self._coronal, self._sagittal):
            canvas.set_slices(None, None)
            canvas.setText(
                f"<span style='color:#555;'>{canvas._title}</span>"
            )
        self._info_label.setText("<span style='color:#666;'>No data loaded.</span>")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Top controls ──────────────────────────────────────────────
        ctrl_bar = QHBoxLayout()

        # Window / Level
        wl_box = QGroupBox("Window / Level (HU)")
        wl_layout = QHBoxLayout(wl_box)
        wl_layout.setContentsMargins(4, 2, 4, 2)

        wl_layout.addWidget(QLabel("Width:"))
        self._ww_spin = QSpinBox()
        self._ww_spin.setRange(1, 4000)
        self._ww_spin.setValue(int(self._ww))
        self._ww_spin.setSuffix(" HU")
        self._ww_spin.valueChanged.connect(self._on_window_changed)
        wl_layout.addWidget(self._ww_spin)

        wl_layout.addWidget(QLabel("Level:"))
        self._wl_spin = QSpinBox()
        self._wl_spin.setRange(-2000, 2000)
        self._wl_spin.setValue(int(self._wl))
        self._wl_spin.setSuffix(" HU")
        self._wl_spin.valueChanged.connect(self._on_window_changed)
        wl_layout.addWidget(self._wl_spin)

        ctrl_bar.addWidget(wl_box)

        # Overlay opacity
        ov_box = QGroupBox("Overlay Opacity")
        ov_layout = QHBoxLayout(ov_box)
        ov_layout.setContentsMargins(4, 2, 4, 2)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.setFixedWidth(120)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self._opacity_label = QLabel("45%")
        ov_layout.addWidget(self._opacity_slider)
        ov_layout.addWidget(self._opacity_label)
        ctrl_bar.addWidget(ov_box)

        ctrl_bar.addStretch()
        root.addLayout(ctrl_bar)

        # ── Canvases ──────────────────────────────────────────────────
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        self._axial = _SliceCanvas("AXIAL  (Z)")
        self._coronal = _SliceCanvas("CORONAL  (Y)")
        self._sagittal = _SliceCanvas("SAGITTAL  (X)")

        self._axial.clicked.connect(self._on_axial_click)
        self._coronal.clicked.connect(self._on_coronal_click)
        self._sagittal.clicked.connect(self._on_sagittal_click)

        grid.addWidget(self._axial, 0, 0)
        grid.addWidget(self._coronal, 0, 1)
        grid.addWidget(self._sagittal, 1, 0)

        # Info panel (bottom-right)
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_frame.setStyleSheet("background: #0f1a2e; border: 1px solid #333;")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        self._info_label = QLabel("<span style='color:#666;'>No data loaded.</span>")
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._info_label.setStyleSheet("color: #ccc; font-size: 11px;")
        info_layout.addWidget(self._info_label)
        info_layout.addStretch()
        grid.addWidget(info_frame, 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        root.addWidget(grid_widget, stretch=1)

        # ── Slice sliders ─────────────────────────────────────────────
        sliders_bar = QHBoxLayout()

        self._z_slider = self._make_slider("Axial (Z):", sliders_bar)
        self._y_slider = self._make_slider("Coronal (Y):", sliders_bar)
        self._x_slider = self._make_slider("Sagittal (X):", sliders_bar)

        self._z_slider.valueChanged.connect(self._on_z_changed)
        self._y_slider.valueChanged.connect(self._on_y_changed)
        self._x_slider.valueChanged.connect(self._on_x_changed)

        root.addLayout(sliders_bar)

    @staticmethod
    def _make_slider(label: str, layout: QHBoxLayout) -> QSlider:
        box = QGroupBox(label)
        bl = QHBoxLayout(box)
        bl.setContentsMargins(4, 2, 4, 2)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 0)
        sl.setValue(0)
        bl.addWidget(sl)
        layout.addWidget(box)
        return sl

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _on_z_changed(self, val: int) -> None:
        if self._ct is None:
            return
        self._iz = int(np.clip(val, 0, self._ct.shape[0] - 1))
        self._refresh_axial()
        self._update_crosshairs()

    def _on_y_changed(self, val: int) -> None:
        if self._ct is None:
            return
        self._iy = int(np.clip(val, 0, self._ct.shape[1] - 1))
        self._refresh_coronal()
        self._update_crosshairs()

    def _on_x_changed(self, val: int) -> None:
        if self._ct is None:
            return
        self._ix = int(np.clip(val, 0, self._ct.shape[2] - 1))
        self._refresh_sagittal()
        self._update_crosshairs()

    def _on_axial_click(self, xf: float, yf: float) -> None:
        """Click on axial view → update X (col) and Y (row) positions."""
        if self._ct is None:
            return
        _, Y, X = self._ct.shape
        self._ix = int(np.clip(xf * X, 0, X - 1))
        self._iy = int(np.clip(yf * Y, 0, Y - 1))
        self._x_slider.setValue(self._ix)
        self._y_slider.setValue(self._iy)
        self._refresh_coronal()
        self._refresh_sagittal()
        self._update_crosshairs()

    def _on_coronal_click(self, xf: float, yf: float) -> None:
        """Click on coronal view → update X (col) and Z (row) positions."""
        if self._ct is None:
            return
        Z, _, X = self._ct.shape
        self._ix = int(np.clip(xf * X, 0, X - 1))
        self._iz = int(np.clip(yf * Z, 0, Z - 1))
        self._x_slider.setValue(self._ix)
        self._z_slider.setValue(self._iz)
        self._refresh_axial()
        self._refresh_sagittal()
        self._update_crosshairs()

    def _on_sagittal_click(self, xf: float, yf: float) -> None:
        """Click on sagittal view → update Y (col) and Z (row) positions."""
        if self._ct is None:
            return
        Z, Y, _ = self._ct.shape
        self._iy = int(np.clip(xf * Y, 0, Y - 1))
        self._iz = int(np.clip(yf * Z, 0, Z - 1))
        self._y_slider.setValue(self._iy)
        self._z_slider.setValue(self._iz)
        self._refresh_axial()
        self._refresh_coronal()
        self._update_crosshairs()

    def _on_window_changed(self) -> None:
        self._ww = float(self._ww_spin.value())
        self._wl = float(self._wl_spin.value())
        if self._ct_raw is not None:
            self._ct = self._apply_window(self._ct_raw)
        self._refresh_all()

    def _on_opacity_changed(self, val: int) -> None:
        alpha = val / 100.0
        self._opacity_label.setText(f"{val}%")
        for canvas in (self._axial, self._coronal, self._sagittal):
            canvas.set_overlay_alpha(alpha)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _apply_window(self, hu: np.ndarray) -> np.ndarray:
        """Apply HU window/level → float32 [0, 1]."""
        lo = self._wl - self._ww / 2.0
        hi = self._wl + self._ww / 2.0
        return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _refresh_axial(self) -> None:
        if self._ct is None:
            return
        ct_sl = self._ct[self._iz]
        mask_sl = self._mask[self._iz] if self._mask is not None else None
        self._axial.set_slices(ct_sl, mask_sl)

    def _refresh_coronal(self) -> None:
        if self._ct is None:
            return
        ct_sl = self._ct[:, self._iy, :]   # (Z, X)
        mask_sl = (
            self._mask[:, self._iy, :] if self._mask is not None else None
        )
        self._coronal.set_slices(ct_sl, mask_sl)

    def _refresh_sagittal(self) -> None:
        if self._ct is None:
            return
        ct_sl = self._ct[:, :, self._ix]   # (Z, Y)
        mask_sl = (
            self._mask[:, :, self._ix] if self._mask is not None else None
        )
        self._sagittal.set_slices(ct_sl, mask_sl)

    def _refresh_all(self) -> None:
        self._refresh_axial()
        self._refresh_coronal()
        self._refresh_sagittal()
        self._update_crosshairs()

    def _update_crosshairs(self) -> None:
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        # Axial: crosshair at (ix/X, iy/Y)
        self._axial.set_crosshair(self._ix / max(X - 1, 1), self._iy / max(Y - 1, 1))
        # Coronal: crosshair at (ix/X, iz/Z)
        self._coronal.set_crosshair(self._ix / max(X - 1, 1), self._iz / max(Z - 1, 1))
        # Sagittal: crosshair at (iy/Y, iz/Z)
        self._sagittal.set_crosshair(self._iy / max(Y - 1, 1), self._iz / max(Z - 1, 1))

    def _update_sliders(self) -> None:
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        for sl, maxv, cur in (
            (self._z_slider, Z - 1, self._iz),
            (self._y_slider, Y - 1, self._iy),
            (self._x_slider, X - 1, self._ix),
        ):
            sl.blockSignals(True)
            sl.setRange(0, max(maxv, 0))
            sl.setValue(cur)
            sl.blockSignals(False)

    def _update_info(self) -> None:
        if self._ct is None:
            self._info_label.setText("<span style='color:#666;'>No data loaded.</span>")
            return
        Z, Y, X = self._ct.shape
        sz, sy, sx = self._spacing
        lesion_vox = int(self._mask.sum()) if self._mask is not None else 0
        lesion_mm3 = lesion_vox * sz * sy * sx
        lesion_ml = lesion_mm3 / 1000.0

        has_mask = self._mask is not None
        html = (
            "<b style='color:#e94560;'>Volume Info</b><br>"
            f"<b>Shape:</b> {Z} × {Y} × {X} (Z×Y×X)<br>"
            f"<b>Spacing:</b> {sz:.3f} × {sy:.3f} × {sx:.3f} mm<br>"
            f"<b>FOV:</b> {Z*sz:.1f} × {Y*sy:.1f} × {X*sx:.1f} mm<br>"
            "<br>"
            "<b style='color:#e94560;'>Lesion</b><br>"
        )
        if has_mask and lesion_vox > 0:
            html += (
                f"<b>Voxels:</b> {lesion_vox:,}<br>"
                f"<b>Volume:</b> {lesion_mm3:.1f} mm³<br>"
                f"<b>Volume:</b> {lesion_ml:.4f} mL<br>"
            )
        elif has_mask:
            html += "<span style='color:#888;'>No lesion detected.</span><br>"
        else:
            html += "<span style='color:#888;'>No mask loaded.</span><br>"

        html += (
            "<br><b style='color:#e94560;'>Navigation</b><br>"
            "• Drag sliders to scroll slices<br>"
            "• Click any view to jump<br>"
            "• Adjust W/L for contrast<br>"
        )
        self._info_label.setText(html)
