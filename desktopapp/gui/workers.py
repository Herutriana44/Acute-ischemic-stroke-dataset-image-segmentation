"""QThread workers for running inference without blocking the GUI."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QThread, pyqtSignal

if TYPE_CHECKING:
    from desktopapp.backend import inference


class InferenceWorker(QThread):
    """Run DICOM or image inference in a background thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, dict)
    error = pyqtSignal(str)

    def __init__(self, mode: str, input_path: Path) -> None:
        super().__init__()
        self._mode = mode  # "dicom" or "image"
        self._input_path = input_path
        self._run_id: str = uuid.uuid4().hex[:12]

    def run(self) -> None:
        try:
            from backend.inference import run_dicom_inference, run_image_inference

            if self._mode == "dicom":
                self.progress.emit("Extracting archive…")
                run_dir, result = run_dicom_inference(self._input_path, self._run_id)
            else:
                self.progress.emit("Processing image…")
                run_dir, result = run_image_inference(self._input_path, self._run_id)
            self.finished.emit(run_dir, result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error.emit(str(exc))
