from PySide6.QtCore import QSize, QMetaObject, QCoreApplication, QRect
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QVBoxLayout, QStackedWidget, QWidget, QFrame, QLabel, QScrollArea, QHBoxLayout

class Ui_MainPages(object): 
    def setupUi(self, MainPages):
        if not MainPages.objectName():
            MainPages.setObjectName(u"MainPages")
        MainPages.resize(860, 600)
        self.main_pages_layout = QVBoxLayout(MainPages)
        self.main_pages_layout.setSpacing(0)
        self.main_pages_layout.setObjectName(u"main_pages_layout")
        self.main_pages_layout.setContentsMargins(5, 5, 5, 5)
        self.pages = QStackedWidget(MainPages)
        self.pages.setObjectName(u"pages")
        self.page_1 = QWidget()
        self.page_1.setObjectName(u"page_1")
        self.page_1_layout = QVBoxLayout(self.page_1)
        self.page_1_layout.setSpacing(5)
        self.page_1_layout.setObjectName(u"page_1_layout")
        self.page_1_layout.setContentsMargins(5, 5, 5, 5)
        self.welcome_base = QFrame(self.page_1)
        self.welcome_base.setObjectName(u"welcome_base")
        self.welcome_base.setMinimumSize(QSize(300, 150))
        self.welcome_base.setMaximumSize(QSize(300, 150))
        self.welcome_base.setFrameShape(QFrame.NoFrame)
        self.welcome_base.setFrameShadow(QFrame.Raised)
        self.center_page_layout = QVBoxLayout(self.welcome_base)
        self.center_page_layout.setSpacing(10)
        self.center_page_layout.setObjectName(u"center_page_layout")
        self.center_page_layout.setContentsMargins(0, 0, 0, 0)
        self.logo = QFrame(self.welcome_base)
        self.logo.setObjectName(u"logo")
        self.logo.setMinimumSize(QSize(300, 120))
        self.logo.setMaximumSize(QSize(300, 120))
        self.logo.setFrameShape(QFrame.NoFrame)
        self.logo.setFrameShadow(QFrame.Raised)
        self.logo_layout = QVBoxLayout(self.logo)
        self.logo_layout.setSpacing(0)
        self.logo_layout.setObjectName(u"logo_layout")
        self.logo_layout.setContentsMargins(0, 0, 0, 0)
        self.center_page_layout.addWidget(self.logo)
        self.label = QLabel(self.welcome_base)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)
        self.center_page_layout.addWidget(self.label)
        
        # Page 1
        #scroll
        self.scroll_area = QScrollArea(self.page_1)
        self.scroll_area.setObjectName(u"scroll_area")
        self.scroll_area.setStyleSheet(u"background: transparent;")
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        
        # Content
        self.contents = QWidget()
        self.contents.setObjectName(u"contents")
        self.contents.setGeometry(QRect(0, 0, 840, 580))
        self.contents.setStyleSheet(u"background: transparent;")
        
        # Vertical inherit (Contents)
        self.verticalLayout = QVBoxLayout(self.contents)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        
        # Row 1
        self.row_1_1layout = QHBoxLayout()
        self.row_1_1layout.setObjectName(u"row_1_1layout")
        self.verticalLayout.addLayout(self.row_1_1layout)
        
        # add contents to scroll
        self.scroll_area.setWidget(self.contents)
        
        # add scroll to page_1_layout
        self.page_1_layout.addWidget(self.scroll_area)
        
        # add page_1 to pages
        self.pages.addWidget(self.page_1)

        self.page_4 = QWidget()
        self.page_4.setObjectName(u"page_4")
        self.page_4_layout = QVBoxLayout(self.page_4)
        self.page_4_layout.setSpacing(5)
        self.page_4_layout.setObjectName(u"page_4_layout")
        self.page_4_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_area = QScrollArea(self.page_4)
        self.scroll_area.setObjectName(u"scroll_area")
        self.scroll_area.setStyleSheet(u"background: transparent;")
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        # Contents
        self.contents = QWidget()
        self.contents.setObjectName(u"contents")
        self.contents.setGeometry(QRect(0, 0, 840, 580))
        self.contents.setStyleSheet(u"background: transparent;")
        # Vertical inherit (contents)
        self.verticalLayout = QVBoxLayout(self.contents)
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        # row 1
        self.row_1_4layout = QHBoxLayout()
        self.row_1_4layout.setObjectName(u"row_1_4layout")
        self.verticalLayout.addLayout(self.row_1_4layout)
        # row 2
        self.row_2_4layout = QHBoxLayout()
        self.row_2_4layout.setObjectName(u"row_2_4layout")
        self.verticalLayout.addLayout(self.row_2_4layout)
        # add contents to scroll
        self.scroll_area.setWidget(self.contents)
        # add scroll to page_1_layout
        self.page_4_layout.addWidget(self.scroll_area)
        # add page to pages
        self.pages.addWidget(self.page_4)
        
        #Main Page
        self.main_pages_layout.addWidget(self.pages)
        self.retranslateUi(MainPages)
        self.pages.setCurrentIndex(0)
        QMetaObject.connectSlotsByName(MainPages)

    def retranslateUi(self, MainPages):
        MainPages.setWindowTitle(QCoreApplication.translate("MainPages", u"Form", None))
