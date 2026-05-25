import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog)
import pyvista as pv
from pyvistaqt import QtInteractor

class STLViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        
        # Main Layout
        self.layout = QVBoxLayout()
        
        # Load Button (selalu ada)
        self.btn_load = QPushButton("Load STL File")
        self.btn_load.clicked.connect(self.load_stl)
        self.layout.addWidget(self.btn_load)

        self.setLayout(self.layout)

    def showEvent(self, event):
        super().showEvent(event)
        if self.plotter is None:
            # Lazy initialize plotter when widget is first shown
            self.plotter = QtInteractor(self)
            self.layout.addWidget(self.plotter.interactor)

    def load_stl(self):
        if self.plotter is None:
            return
            
        file_path, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
        if file_path:
            self.plotter.clear()
            mesh = pv.read(file_path)
            self.plotter.add_mesh(mesh, color="white", show_edges=True)
            self.plotter.reset_camera()

if __name__ == "__main__":
    from PySide6.QtWidgets import QMainWindow
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setCentralWidget(STLViewer())
    window.show()
    sys.exit(app.exec())
