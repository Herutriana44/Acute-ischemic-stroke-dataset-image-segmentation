import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtCore import QUrl


app = QApplication(sys.argv)

viewer = QQuickWidget()

qml_path = os.path.abspath("viewer.qml")

viewer.setSource(
    QUrl.fromLocalFile(qml_path)
)

viewer.resize(1200,800)
viewer.setWindowTitle(
    "3D Plastinated Human Brain"
)

viewer.show()

sys.exit(app.exec())