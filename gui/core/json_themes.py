import json
import os
from gui.core.json_settings import Settings
class Themes(object):
    setup_settings = Settings()
    _settings = setup_settings.items
    def __init__(self):
        super(Themes, self).__init__()
        '#f26f21, #faae2d'
        self.items = {'theme_name': 'Default', 'app_color': 
            {'dark_one': '#f26f21', 'dark_two': '#faae2d', 'dark_three': '#faae2d', 
                                                             'dark_four': '#f26f21', 'bg_one': '#fff', 'bg_two': '#faae2d', 'bg_three': '#fff', 
                                                             'icon_color': '#fff', 'icon_hover': '#dce1ec', 'icon_pressed': '#f26f21', 
                                                             'icon_active': '#f26f21', 'context_color': '#f26f21', 'context_hover': '#faae2d', 
                                                             'context_pressed': '#faae2d', 'text_title': '#fff', 'text_foreground': '#fff', 'text_description': 
                                                             '#fff', 'text_active': '#f26f21', 'white': '#f5f6f9', 'pink': '#ff007f', 'green': '#00ff7f', 
                                                             'red': '#ff5555', 'yellow': '#f1fa8c'}} 