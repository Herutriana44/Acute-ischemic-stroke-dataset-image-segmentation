"""Native DICOM slice viewer — multi-planar reconstruction (MPR).

Visualises a 3-D DICOM volume (Z, Y, X) as three linked orthogonal planes:
  • AXIAL    — horizontal cross-section, scrolls through Z (superior→inferior)
  • CORONAL  — front-to-back cross-section, scrolls through Y (anterior→posterior)
  • SAGITTAL — left-to-right cross-section, scrolls through X (left→right)

Each plane has its own slider directly beneath it, a slice counter label,
and responds to the mouse scroll wheel.  Clicking any plane moves the
crosshair and updates the other two planes accordingly.

Controls
--------
  Slider / scroll wheel  — navigate slices in that plane
  Click on image         — jump crosshair to that position
  Window presets         — Brain / Stroke / Bone / Soft-Tissue one-click
  W / L spinboxes        — fine-tune HU window width & level
  Overlay opacity        — adjust lesion mask transparency
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Window / Level presets  (width, level)  in HU
# ---------------------------------------------------------------------------
_WL_PRESETS: dict[str, tuple[int, int]] = {
    "Brain":       (80,   40),
    "Stroke":      (40,   40),
    "Bone":        (2000, 400),
    "Soft Tissue": (400,  40),
}


# ---------------------------------------------------------------------------
# _SliceCanvas — one orthogonal plane
# ---------------------------------------------------------------------------

class _SliceCanvas(QWidget):
    """Renders one CT slice + mask overlay + crosshair.

    Signals
    -------
    clicked(x_frac, y_frac)
        Normalised [0, 1] image coordinates when the user clicks.
    scroll_delta(int)
        +1 or -1 when the user scrolls the mouse wheel.
    """

    clicked     = pyqtSignal(float, float)
    scroll_delta = pyqtSignal(int)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title

        # Data
        self._ct_slice:   Optional[np.ndarray] = None   # float32 [0,1]  (H, W)
        self._mask_slice: Optional[np.ndarray] = None   # uint8   {0,1}  (H, W)

        # Physical pixel size ratio  height_mm / width_mm  for this plane.
        # Used to pre-scale the raw array so anatomy looks correct.
        self._aspect: float = 1.0   # row_spacing / col_spacing

        # Crosshair (normalised 0-1 in *image* space after aspect correction)
        self._cx: float = 0.5
        self._cy: float = 0.5

        self._overlay_alpha: float = 0.45

        # ── image label ──────────────────────────────────────────────
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._img_label.setStyleSheet("background: #000;")
        self._img_label.setMinimumSize(180, 180)

        # ── slice counter ─────────────────────────────────────────────
        self._slice_label = QLabel("—")
        self._slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slice_label.setStyleSheet("color:#aaa; font-size:10px;")

        # ── slider ───────────────────────────────────────────────────
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)

        # ── layout ───────────────────────────────────────────────────
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        lay.addWidget(self._img_label, stretch=1)

        bottom = QHBoxLayout()
        bottom.addWidget(self._slice_label)
        bottom.addWidget(self._slider, stretch=1)
        lay.addLayout(bottom)

        self.setStyleSheet("background: #0a0a1a; border: 1px solid #2a2a3e;")

        # Forward scroll events from the image label
        self._img_label.installEventFilter(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def slider(self) -> QSlider:
        return self._slider

    def set_slices(
        self,
        ct: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        aspect: float = 1.0,
        slice_idx: int = 0,
        slice_total: int = 1,
    ) -> None:
        self._ct_slice   = ct
        self._mask_slice = mask
        self._aspect     = max(aspect, 0.01)
        self._slice_label.setText(f"{slice_idx + 1} / {slice_total}")
        self._render()

    def set_crosshair(self, cx: float, cy: float) -> None:
        self._cx = float(np.clip(cx, 0.0, 1.0))
        self._cy = float(np.clip(cy, 0.0, 1.0))
        self._render()

    def set_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = float(np.clip(alpha, 0.0, 1.0))
        self._render()

    def clear(self) -> None:
        self._ct_slice   = None
        self._mask_slice = None
        self._slice_label.setText("—")
        self._img_label.clear()
        self._img_label.setStyleSheet("background: #000;")

    # ------------------------------------------------------------------
    # Mouse / wheel
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):  # type: ignore[override]
        from PyQt6.QtCore import QEvent
        if obj is self._img_label and event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            self.scroll_delta.emit(1 if delta > 0 else -1)
            return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._ct_slice is None:
            return
        # Map click to image-label coordinates
        lw = self._img_label.width()
        lh = self._img_label.height()
        if lw == 0 or lh == 0:
            return
        pm = self._img_label.pixmap()
        if pm is None:
            return
        pw, ph = pm.width(), pm.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        # Position relative to this widget, then offset to label
        lpos = self._img_label.mapFrom(self, event.pos())
        px = lpos.x() - ox
        py = lpos.y() - oy
        if px < 0 or py < 0 or px >= pw or py >= ph:
            return
        self.clicked.emit(float(px / pw), float(py / ph))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if self._ct_slice is None:
            self._img_label.clear()
            return

        ct = self._ct_slice          # float32 [0,1]  (H, W)
        H, W = ct.shape

        # ── build RGB array ──────────────────────────────────────────
        ct_u8 = np.clip(ct * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1).copy()

        if self._mask_slice is not None and self._overlay_alpha > 0:
            mask = self._mask_slice.astype(bool)
            if mask.any():
                a = self._overlay_alpha
                rgb_f = rgb.astype(np.float32)
                rgb_f[mask, 0] = (1 - a) * rgb_f[mask, 0] + a * 234
                rgb_f[mask, 1] = (1 - a) * rgb_f[mask, 1] + a * 88
                rgb_f[mask, 2] = (1 - a) * rgb_f[mask, 2] + a * 12
                rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

        # ── QImage → QPixmap ─────────────────────────────────────────
        rgb_c = np.ascontiguousarray(rgb)
        qimg  = QImage(rgb_c.data, W, H, W * 3, QImage.Format.Format_RGB888)
        pm    = QPixmap.fromImage(qimg)

        # ── correct physical aspect ratio ────────────────────────────
        # self._aspect = row_spacing / col_spacing
        # The raw array has H rows and W cols.
        # Physical height = H * row_spacing,  physical width = W * col_spacing
        # We want the displayed image to reflect real-world proportions.
        phys_h = H * self._aspect   # in col_spacing units
        phys_w = float(W)

        lw = self._img_label.width()
        lh = self._img_label.height()
        if lw > 0 and lh > 0:
            # Scale so the physical rectangle fits inside (lw × lh)
            scale = min(lw / phys_w, lh / phys_h)
            disp_w = int(phys_w * scale)
            disp_h = int(phys_h * scale)
            pm = pm.scaled(
                disp_w, disp_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # ── crosshair ────────────────────────────────────────────────
        painter = QPainter(pm)
        pen = QPen(QColor(0, 255, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)
        cx_px = int(self._cx * pm.width())
        cy_px = int(self._cy * pm.height())
        painter.drawLine(cx_px, 0, cx_px, pm.height())
        painter.drawLine(0, cy_px, pm.width(), cy_px)

        # ── title ─────────────────────────────────────────────────────
        painter.setPen(QColor(220, 220, 220, 210))
        painter.drawText(6, 16, self._title)
        painter.end()

        self._img_label.setPixmap(pm)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render()


# ---------------------------------------------------------------------------
# DicomViewer — main widget
# ---------------------------------------------------------------------------

class DicomViewer(QWidget):
    """Multi-planar DICOM viewer (Axial / Coronal / Sagittal).

    The volume is expected in (Z, Y, X) order where:
      Z — axial slices (superior → inferior)
      Y — rows        (anterior → posterior)
      X — columns     (left → right)

    Usage::

        viewer = DicomViewer()
        viewer.load_volumes(hu_volume, mask_volume, spacing=(sz, sy, sx), use_hu=True)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._ct:     Optional[np.ndarray] = None   # float32 [0,1]  (Z,Y,X)
        self._ct_raw: Optional[np.ndarray] = None   # raw HU         (Z,Y,X)
        self._mask:   Optional[np.ndarray] = None   # uint8  {0,1}   (Z,Y,X)

        # spacing in mm: (sz, sy, sx)
        self._spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

        # current slice indices
        self._iz: int = 0
        self._iy: int = 0
        self._ix: int = 0

        # HU window
        self._ww: float = 80.0
        self._wl: float = 40.0
        self._use_hu: bool = False

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
        """Load a 3-D CT volume and optional segmentation mask.

        Parameters
        ----------
        ct_volume : ndarray (Z, Y, X)
            Raw HU values when ``use_hu=True``, otherwise [0,255] or [0,1].
        mask_volume : ndarray (Z, Y, X) or None
            Binary mask (0/1 or 0/255).
        spacing : (sz, sy, sx) in mm
        use_hu : bool
            Apply HU windowing via the W/L controls.
        """
        self._use_hu  = use_hu
        self._spacing = spacing

        ct = ct_volume.astype(np.float32)
        if use_hu:
            self._ct_raw = ct
            self._ct     = self._apply_window(ct)
        else:
            self._ct_raw = None
            if ct.max() > 1.5:
                ct = ct / 255.0
            self._ct = np.clip(ct, 0.0, 1.0)

        if mask_volume is not None:
            m = mask_volume.astype(np.float32)
            if m.max() > 1.5:
                m = m / 255.0
            self._mask = (m > 0.5).astype(np.uint8)
        else:
            self._mask = None

        Z, Y, X = self._ct.shape
        self._iz = Z // 2
        self._iy = Y // 2
        self._ix = X // 2

        self._sync_sliders()
        self._update_info()
        self._refresh_all()

    def clear(self) -> None:
        self._ct = self._ct_raw = self._mask = None
        for c in (self._axial, self._coronal, self._sagittal):
            c.clear()
            c.slider.setRange(0, 0)
        self._info_label.setText(
            "<span style='color:#666;'>No DICOM series loaded.</span>"
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── top control bar ───────────────────────────────────────────
        root.addLayout(self._build_controls())

        # ── 2×2 grid ─────────────────────────────────────────────────
        grid_w = QWidget()
        grid   = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        self._axial    = _SliceCanvas("AXIAL  (Z)")
        self._coronal  = _SliceCanvas("CORONAL  (Y)")
        self._sagittal = _SliceCanvas("SAGITTAL  (X)")

        # connect sliders
        self._axial.slider.valueChanged.connect(self._on_z_changed)
        self._coronal.slider.valueChanged.connect(self._on_y_changed)
        self._sagittal.slider.valueChanged.connect(self._on_x_changed)

        # connect clicks
        self._axial.clicked.connect(self._on_axial_click)
        self._coronal.clicked.connect(self._on_coronal_click)
        self._sagittal.clicked.connect(self._on_sagittal_click)

        # connect scroll wheel
        self._axial.scroll_delta.connect(
            lambda d: self._axial.slider.setValue(self._axial.slider.value() + d)
        )
        self._coronal.scroll_delta.connect(
            lambda d: self._coronal.slider.setValue(self._coronal.slider.value() + d)
        )
        self._sagittal.scroll_delta.connect(
            lambda d: self._sagittal.slider.setValue(self._sagittal.slider.value() + d)
        )

        grid.addWidget(self._axial,    0, 0)
        grid.addWidget(self._coronal,  0, 1)
        grid.addWidget(self._sagittal, 1, 0)

        # info panel (bottom-right)
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_frame.setStyleSheet("background:#0f1a2e; border:1px solid #2a2a3e;")
        info_lay = QVBoxLayout(info_frame)
        info_lay.setContentsMargins(8, 8, 8, 8)
        self._info_label = QLabel(
            "<span style='color:#666;'>No DICOM series loaded.</span>"
        )
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._info_label.setStyleSheet("color:#ccc; font-size:11px;")
        info_lay.addWidget(self._info_label)
        info_lay.addStretch()
        grid.addWidget(info_frame, 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        root.addWidget(grid_w, stretch=1)

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        # Window presets
        preset_box = QGroupBox("Window Preset")
        preset_lay = QHBoxLayout(preset_box)
        preset_lay.setContentsMargins(4, 2, 4, 2)
        preset_lay.setSpacing(4)
        for name, (ww, wl) in _WL_PRESETS.items():
            btn = QPushButton(name)
            btn.setFixedHeight(24)
            btn.clicked.connect(
                lambda _checked, w=ww, l=wl: self._apply_preset(w, l)
            )
            preset_lay.addWidget(btn)
        bar.addWidget(preset_box)

        # W / L spinboxes
        wl_box = QGroupBox("W / L  (HU)")
        wl_lay = QHBoxLayout(wl_box)
        wl_lay.setContentsMargins(4, 2, 4, 2)
        wl_lay.setSpacing(4)

        wl_lay.addWidget(QLabel("W:"))
        self._ww_spin = QSpinBox()
        self._ww_spin.setRange(1, 4000)
        self._ww_spin.setValue(int(self._ww))
        self._ww_spin.setSuffix(" HU")
        self._ww_spin.setFixedWidth(90)
        self._ww_spin.valueChanged.connect(self._on_window_changed)
        wl_lay.addWidget(self._ww_spin)

        wl_lay.addWidget(QLabel("L:"))
        self._wl_spin = QSpinBox()
        self._wl_spin.setRange(-2000, 2000)
        self._wl_spin.setValue(int(self._wl))
        self._wl_spin.setSuffix(" HU")
        self._wl_spin.setFixedWidth(90)
        self._wl_spin.valueChanged.connect(self._on_window_changed)
        wl_lay.addWidget(self._wl_spin)
        bar.addWidget(wl_box)

        # Overlay opacity
        ov_box = QGroupBox("Lesion Overlay")
        ov_lay = QHBoxLayout(ov_box)
        ov_lay.setContentsMargins(4, 2, 4, 2)
        ov_lay.setSpacing(4)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.setFixedWidth(100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self._opacity_label = QLabel("45%")
        self._opacity_label.setFixedWidth(32)
        ov_lay.addWidget(self._opacity_slider)
        ov_lay.addWidget(self._opacity_label)
        bar.addWidget(ov_box)

        bar.addStretch()
        return bar

    # ------------------------------------------------------------------
    # Slots
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
        """Axial click → update ix (col) and iy (row)."""
        if self._ct is None:
            return
        _, Y, X = self._ct.shape
        self._ix = int(np.clip(xf * X, 0, X - 1))
        self._iy = int(np.clip(yf * Y, 0, Y - 1))
        self._sagittal.slider.setValue(self._ix)
        self._coronal.slider.setValue(self._iy)
        self._refresh_coronal()
        self._refresh_sagittal()
        self._update_crosshairs()

    def _on_coronal_click(self, xf: float, yf: float) -> None:
        """Coronal click → update ix (col) and iz (row)."""
        if self._ct is None:
            return
        Z, _, X = self._ct.shape
        self._ix = int(np.clip(xf * X, 0, X - 1))
        self._iz = int(np.clip(yf * Z, 0, Z - 1))
        self._sagittal.slider.setValue(self._ix)
        self._axial.slider.setValue(self._iz)
        self._refresh_axial()
        self._refresh_sagittal()
        self._update_crosshairs()

    def _on_sagittal_click(self, xf: float, yf: float) -> None:
        """Sagittal click → update iy (col) and iz (row)."""
        if self._ct is None:
            return
        Z, Y, _ = self._ct.shape
        self._iy = int(np.clip(xf * Y, 0, Y - 1))
        self._iz = int(np.clip(yf * Z, 0, Z - 1))
        self._coronal.slider.setValue(self._iy)
        self._axial.slider.setValue(self._iz)
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
        self._opacity_label.setText(f"{val}%")
        alpha = val / 100.0
        for c in (self._axial, self._coronal, self._sagittal):
            c.set_overlay_alpha(alpha)

    def _apply_preset(self, ww: int, wl: int) -> None:
        self._ww_spin.blockSignals(True)
        self._wl_spin.blockSignals(True)
        self._ww_spin.setValue(ww)
        self._wl_spin.setValue(wl)
        self._ww_spin.blockSignals(False)
        self._wl_spin.blockSignals(False)
        self._ww = float(ww)
        self._wl = float(wl)
        if self._ct_raw is not None:
            self._ct = self._apply_window(self._ct_raw)
        self._refresh_all()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _apply_window(self, hu: np.ndarray) -> np.ndarray:
        lo = self._wl - self._ww / 2.0
        hi = self._wl + self._ww / 2.0
        return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _refresh_axial(self) -> None:
        """Axial plane: slice through Z → shows (Y, X) with spacing (sy, sx)."""
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        ct_sl   = self._ct[self._iz]                                    # (Y, X)
        mask_sl = self._mask[self._iz] if self._mask is not None else None
        sz, sy, sx = self._spacing
        # aspect = row_spacing / col_spacing = sy / sx
        self._axial.set_slices(
            ct_sl, mask_sl,
            aspect=sy / sx,
            slice_idx=self._iz, slice_total=Z,
        )

    def _refresh_coronal(self) -> None:
        """Coronal plane: slice through Y → shows (Z, X) with spacing (sz, sx)."""
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        ct_sl   = self._ct[:, self._iy, :]                              # (Z, X)
        mask_sl = self._mask[:, self._iy, :] if self._mask is not None else None
        sz, sy, sx = self._spacing
        # aspect = row_spacing / col_spacing = sz / sx
        self._coronal.set_slices(
            ct_sl, mask_sl,
            aspect=sz / sx,
            slice_idx=self._iy, slice_total=Y,
        )

    def _refresh_sagittal(self) -> None:
        """Sagittal plane: slice through X → shows (Z, Y) with spacing (sz, sy)."""
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        ct_sl   = self._ct[:, :, self._ix]                              # (Z, Y)
        mask_sl = self._mask[:, :, self._ix] if self._mask is not None else None
        sz, sy, sx = self._spacing
        # aspect = row_spacing / col_spacing = sz / sy
        self._sagittal.set_slices(
            ct_sl, mask_sl,
            aspect=sz / sy,
            slice_idx=self._ix, slice_total=X,
        )

    def _refresh_all(self) -> None:
        self._refresh_axial()
        self._refresh_coronal()
        self._refresh_sagittal()
        self._update_crosshairs()

    def _update_crosshairs(self) -> None:
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        # Axial view shows (Y rows, X cols) → crosshair at (ix/X, iy/Y)
        self._axial.set_crosshair(
            self._ix / max(X - 1, 1),
            self._iy / max(Y - 1, 1),
        )
        # Coronal view shows (Z rows, X cols) → crosshair at (ix/X, iz/Z)
        self._coronal.set_crosshair(
            self._ix / max(X - 1, 1),
            self._iz / max(Z - 1, 1),
        )
        # Sagittal view shows (Z rows, Y cols) → crosshair at (iy/Y, iz/Z)
        self._sagittal.set_crosshair(
            self._iy / max(Y - 1, 1),
            self._iz / max(Z - 1, 1),
        )

    def _sync_sliders(self) -> None:
        if self._ct is None:
            return
        Z, Y, X = self._ct.shape
        for slider, maxv, cur in (
            (self._axial.slider,    Z - 1, self._iz),
            (self._coronal.slider,  Y - 1, self._iy),
            (self._sagittal.slider, X - 1, self._ix),
        ):
            slider.blockSignals(True)
            slider.setRange(0, max(maxv, 0))
            slider.setValue(cur)
            slider.blockSignals(False)

    def _update_info(self) -> None:
        if self._ct is None:
            self._info_label.setText(
                "<span style='color:#666;'>No DICOM series loaded.</span>"
            )
            return
        Z, Y, X = self._ct.shape
        sz, sy, sx = self._spacing
        lesion_vox = int(self._mask.sum()) if self._mask is not None else 0
        lesion_mm3 = lesion_vox * sz * sy * sx
        lesion_ml  = lesion_mm3 / 1000.0

        html = (
            "<b style='color:#e94560;'>Volume</b><br>"
            f"<b>Slices (Z):</b> {Z}<br>"
            f"<b>Rows   (Y):</b> {Y}<br>"
            f"<b>Cols   (X):</b> {X}<br>"
            f"<b>Spacing:</b> {sz:.3f} × {sy:.3f} × {sx:.3f} mm<br>"
            f"<b>FOV:</b> {Z*sz:.1f} × {Y*sy:.1f} × {X*sx:.1f} mm<br>"
            "<br>"
            "<b style='color:#e94560;'>Lesion Mask</b><br>"
        )
        if self._mask is not None and lesion_vox > 0:
            html += (
                f"<b>Voxels:</b> {lesion_vox:,}<br>"
                f"<b>Volume:</b> {lesion_mm3:.1f} mm³<br>"
                f"<b>         </b> {lesion_ml:.4f} mL<br>"
            )
        elif self._mask is not None:
            html += "<span style='color:#888;'>No lesion detected.</span><br>"
        else:
            html += "<span style='color:#888;'>No mask loaded.</span><br>"

        html += (
            "<br><b style='color:#e94560;'>Navigation</b><br>"
            "• Slider or scroll wheel per plane<br>"
            "• Click image to move crosshair<br>"
            "• Crosshair links all three planes<br>"
            "• Presets for quick W/L adjustment<br>"
        )
        self._info_label.setText(html)
