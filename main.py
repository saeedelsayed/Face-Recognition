from PySide6.QtWidgets import *
from mainwindow import MainWindow
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(app.exec())