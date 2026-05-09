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
    import logging
    from PyQt6.QtWidgets import QMessageBox
    # Configure logging to file and console
    logger = logging.getLogger('desktopapp')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # File handler
    file_handler = logging.FileHandler('desktopapp.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Global exception hook to show error dialog and log
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error('Uncaught exception', exc_info=(exc_type, exc_value, exc_traceback))
        err_msg = ''.join(logging.Formatter().formatException((exc_type, exc_value, exc_traceback)))
        QMessageBox.critical(None, 'Application Error', f'An unexpected error occurred:\n{err_msg}')
    import sys
    sys.excepthook = handle_exception
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
