from gui.core.functions import Functions
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtSvgWidgets import *
from gui.core.json_settings import Settings
from gui.core.json_themes import Themes
from gui.widgets import *
from . setup_main_window import *
from gui.uis.pages.ui_main_pages import Ui_MainPages
from gui.widgets.py_credits_bar.py_credits import PyCredits

class UI_MainWindow(object):
    
    def setup_ui(self, parent):
        if not parent.objectName():
            parent.setObjectName("MainWindow")
        settings = Settings()
        self.settings = settings.items
        themes = Themes()
        self.themes = themes.items
        parent.resize(self.settings["startup_size"][0], self.settings["startup_size"][1])
        parent.setMinimumSize(self.settings["minimum_size"][0], self.settings["minimum_size"][1])
        self.central_widget = QWidget() 
        self.central_widget.setStyleSheet(f'''
            font: {self.settings["font"]["text_size"]}pt "{self.settings["font"]["family"]}";
            color: {self.themes["app_color"]["text_foreground"]};
        ''')
        self.central_widget_layout = QVBoxLayout(self.central_widget)
        if self.settings["custom_title_bar"]:
            self.central_widget_layout.setContentsMargins(10,10,10,10)
        else:
            self.central_widget_layout.setContentsMargins(0,0,0,0)
        self.window = PyWindow(
            parent,
            bg_color = self.themes["app_color"]["bg_one"],
            border_color = self.themes["app_color"]["bg_two"],
            text_color = self.themes["app_color"]["text_foreground"]
        )
        
        if not self.settings["custom_title_bar"]:
            self.window.set_stylesheet(border_radius = 0, border_size = 0)
        self.central_widget_layout.addWidget(self.window)
        left_menu_margin = self.settings["left_menu_content_margins"]
        left_menu_minimum = self.settings["lef_menu_size"]["minimum"]
        self.left_menu_frame = QFrame()
        self.left_menu_frame.setMaximumSize(left_menu_minimum + (left_menu_margin * 2), 17280)
        self.left_menu_frame.setMinimumSize(left_menu_minimum + (left_menu_margin * 2), 0)
        self.left_menu_layout = QHBoxLayout(self.left_menu_frame)
        self.left_menu_layout.setContentsMargins(
            left_menu_margin,
            left_menu_margin,
            left_menu_margin,
            left_menu_margin  
        ) 
        
        self.left_menu = PyLeftMenu(
            parent = self.left_menu_frame,
            app_parent = self.central_widget,
            dark_one = self.themes["app_color"]["dark_one"],
            dark_three = self.themes["app_color"]["dark_three"],
            dark_four = self.themes["app_color"]["dark_four"],
            bg_one = self.themes["app_color"]["bg_one"],
            icon_color = self.themes["app_color"]["icon_color"],
            icon_color_hover = self.themes["app_color"]["icon_hover"],
            icon_color_pressed = self.themes["app_color"]["icon_pressed"],
            icon_color_active = self.themes["app_color"]["icon_active"],
            context_color = self.themes["app_color"]["context_color"],
            text_foreground = self.themes["app_color"]["text_foreground"],
            text_active = self.themes["app_color"]["text_active"]
        )
        
        self.left_menu_layout.addWidget(self.left_menu)
        
        self.left_column_frame = QFrame()
        self.left_column_frame.setMaximumWidth(self.settings["left_column_size"]["minimum"])
        self.left_column_frame.setMinimumWidth(self.settings["left_column_size"]["minimum"])
        self.left_column_frame.setStyleSheet(f"background: {self.themes['app_color']['bg_two']}")
        self.left_column_layout = QVBoxLayout(self.left_column_frame)
        self.left_column_layout.setContentsMargins(0,0,0,0)
        self.left_column = PyLeftColumn(
            parent,
            app_parent = self.central_widget,
            text_title = "Settings Left Frame",
            text_title_size = self.settings["font"]["title_size"],
            text_title_color = self.themes['app_color']['text_foreground'],
            icon_path = Functions.set_svg_icon("icon_settings.svg"),
            dark_one = self.themes['app_color']['dark_one'],
            bg_color = self.themes['app_color']['bg_three'],
            btn_color = self.themes['app_color']['bg_three'],
            btn_color_hover = self.themes['app_color']['bg_two'],
            btn_color_pressed = self.themes['app_color']['bg_one'],
            icon_color = self.themes['app_color']['icon_color'],
            icon_color_hover = self.themes['app_color']['icon_hover'],
            context_color = self.themes['app_color']['context_color'],
            icon_color_pressed = self.themes['app_color']['icon_pressed'],
            icon_close_path = Functions.set_svg_icon("icon_close.svg")
        )
        self.left_column_layout.addWidget(self.left_column)
        
        self.right_app_frame = QFrame()
        self.right_app_layout = QVBoxLayout(self.right_app_frame)
        self.right_app_layout.setContentsMargins(3,3,3,3)
        self.right_app_layout.setSpacing(6)
        self.title_bar_frame = QFrame()
        self.title_bar_frame.setMinimumHeight(40)
        self.title_bar_frame.setMaximumHeight(40)
        self.title_bar_layout = QVBoxLayout(self.title_bar_frame)
        self.title_bar_layout.setContentsMargins(0,0,0,0)
        self.title_bar = PyTitleBar(
            parent,
            logo_width = 170,
            app_parent = self.central_widget,
            logo_image = "logo_top_100x22.svg",
            bg_color = self.themes["app_color"]["bg_two"],
            div_color = self.themes["app_color"]["bg_three"],
            btn_bg_color = self.themes["app_color"]["bg_two"],
            btn_bg_color_hover = self.themes["app_color"]["bg_three"],
            btn_bg_color_pressed = self.themes["app_color"]["bg_one"],
            icon_color = self.themes["app_color"]["icon_color"],
            icon_color_hover = self.themes["app_color"]["icon_hover"],
            icon_color_pressed = self.themes["app_color"]["icon_pressed"],
            icon_color_active = self.themes["app_color"]["icon_active"],
            context_color = self.themes["app_color"]["context_color"],
            dark_one = self.themes["app_color"]["dark_one"],
            text_foreground = self.themes["app_color"]["text_foreground"],
            radius = 8,
            font_family = self.settings["font"]["family"],
            title_size = self.settings["font"]["title_size"],
            is_custom_title_bar = self.settings["custom_title_bar"]
        )
        
        self.title_bar_layout.addWidget(self.title_bar)
        
        self.content_area_frame = QFrame()
        self.content_area_layout = QHBoxLayout(self.content_area_frame)
        self.content_area_layout.setContentsMargins(0,0,0,0)
        self.content_area_layout.setSpacing(0)
        self.content_area_left_frame = QFrame()
        
        self.load_pages = Ui_MainPages()
        self.load_pages.setupUi(self.content_area_left_frame)
        
        self.right_column_frame = QFrame()
        self.right_column_frame.setMinimumWidth(self.settings["right_column_size"]["minimum"])
        self.right_column_frame.setMaximumWidth(self.settings["right_column_size"]["minimum"])

        self.content_area_right_layout = QVBoxLayout(self.right_column_frame)
        self.content_area_right_layout.setContentsMargins(5,5,5,5)
        self.content_area_right_layout.setSpacing(0)
        self.content_area_right_bg_frame = QFrame()
        self.content_area_right_bg_frame.setObjectName("content_area_right_bg_frame")
        self.content_area_right_bg_frame.setStyleSheet(f'''
        #content_area_right_bg_frame {{
            border-radius: 8px;
            background-color: {self.themes["app_color"]["bg_two"]};
        }}
        ''')
        self.content_area_right_layout.addWidget(self.content_area_right_bg_frame)
        self.content_area_layout.addWidget(self.content_area_left_frame)
        self.content_area_layout.addWidget(self.right_column_frame)
        
        self.credits_frame = QFrame()
        self.credits_frame.setMinimumHeight(26)
        self.credits_frame.setMaximumHeight(26)
        self.credits_layout = QVBoxLayout(self.credits_frame)
        self.credits_layout.setContentsMargins(0,0,0,0)
        self.credits = PyCredits(
            bg_two = self.themes["app_color"]["bg_two"],
            copyright = self.settings["copyright"], 
            version = self.settings["version"],
            font_family = self.settings["font"]["family"],
            text_size = self.settings["font"]["text_size"],
            text_description_color = self.themes["app_color"]["text_description"]
        )
        self.credits_layout.addWidget(self.credits)
        
        self.right_app_layout.addWidget(self.title_bar_frame)
        self.right_app_layout.addWidget(self.content_area_frame)
        self.right_app_layout.addWidget(self.credits_frame)
        
        self.window.layout.addWidget(self.left_menu_frame)
        self.window.layout.addWidget(self.left_column_frame)
        self.window.layout.addWidget(self.right_app_frame)
        
        parent.setCentralWidget(self.central_widget)