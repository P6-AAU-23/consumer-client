from CornerProvider import CornerProvider
from helper import quadrilateral_to_rectangle


class Pipeline:
    def __init__(self, args):
        self.corner_provider = CornerProvider("Corner Selection Preview")
        self.args = args

    def process(self, image):
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = quadrilateral_to_rectangle(image, corners)
        return whiteboard
