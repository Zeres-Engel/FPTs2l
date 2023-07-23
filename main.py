import os
import sys
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QApplication
from gui.core.json_settings import Settings
from gui.uis.windows.main_window import *
from gui.widgets import *
from gui.uis.windows.main_window.functions_main_window import *

os.environ["QT_FONT_DPI"] = "96"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create UI
        self.ui = UI_MainWindow()

        # Setup UI
        self.ui.setup_ui(self)

        # Create setting
        settings = Settings()
        self.settings = settings.items

        # Hide grips
        self.hide_grips = True
        
        # Setup UI on window
        SetupMainWindow.setup_gui(self)

        # Show
        self.show()

    def closeEvent(self, event):
        event.accept()

    def btn_clicked(self):
        btn = SetupMainWindow.setup_btns(self)

        if btn.objectName() == "btn_0":
            self.ui.left_menu.select_only_one(btn.objectName())
            MainFunctions.set_page(self, self.ui.load_pages.page_1)

        if btn.objectName() == "btn_1":
            self.ui.left_menu.select_only_one(btn.objectName())
            MainFunctions.set_page(self, self.ui.load_pages.page_4)

    def btn_released(self):
        btn = SetupMainWindow.setup_btns(self)

    def resizeEvent(self, event):
        SetupMainWindow.resize_grips(self)

    def mousePressEvent(self, event):
        self.dragPos = event.globalPosition()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())