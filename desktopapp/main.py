#!/usr/bin/env python3
"""Acute Ischemic Stroke — Desktop App (PyQt6)

Entry point: starts the PyQt6 application and loads the main window.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import webapp services
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Acute Ischemic Stroke — Segmentation")
    app.setOrganizationName("MedicalAI")

    # Enable HiDPI pixmaps (scaling is auto-enabled in PyQt6)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
