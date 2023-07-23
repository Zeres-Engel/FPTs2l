import os

class Functions:
    def __init__(self):
        pass
        
    def set_svg_icon(icon_name):
        app_path = os.path.abspath(os.getcwd())
        folder = "./gui/images/svg_icons/"
        path = os.path.join(app_path, folder)
        icon = os.path.normpath(os.path.join(path, icon_name))
        return icon
    
    def set_svg_image(icon_name):
        app_path = os.path.abspath(os.getcwd())
        folder = "./gui/images/svg_images/"
        path = os.path.join(app_path, folder)
        image = os.path.normpath(os.path.join(path, icon_name))
        return image
