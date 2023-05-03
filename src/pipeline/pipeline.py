import cv2
import torch
import src.pipeline.pipeline_functions as pipe_func
from src.pipeline.corner_provider import CornerProvider
from src.pipeline.pipeline_functions import IdealizeColorsMode


class Pipeline:
    def __init__(self):
        self.start_handler = StartHandler()
        self.corner_provider_handler = CornerProviderHandler()
        self.foreground_remover_handler = ForegroundRemoverHandler()
        self.idealize_colors_handler = IdealizeColorsHandler()
        self.inpainter_handler = InpainterHandler()
        self.final_handler = FinalHandler()

        self.start_handler.set_successor(self.corner_provider_handler)
        self.corner_provider_handler.set_successor(self.foreground_remover_handler)
        self.foreground_remover_handler.set_successor(self.idealize_colors_handler)
        self.idealize_colors_handler.set_successor(self.inpainter_handler)
        self.inpainter_handler.set_successor(self.final_handler)

    def process(self, image: cv2.Mat) -> cv2.Mat:
        result = self.start_handler.handle(image)
        return result


class ImageHandler:
    def __init__(self, successor: 'ImageHandler' = None):
        self._successor = successor

    def set_successor(self, successor: 'ImageHandler') -> None:
        self._successor = successor

    def handle(self, image: cv2.Mat) -> cv2.Mat:
        raise NotImplementedError()


class SkipStepHandler(ImageHandler):
    def __init__(self, skip_condition: bool, successor: ImageHandler = None):
        super().__init__(successor)
        self._skip_condition = skip_condition

    def handle(self, image: cv2.Mat) -> cv2.Mat:
        if self._skip_condition:
            return image
        else:
            return self._successor.handle(image)


class StartHandler(ImageHandler):
    def __init__(self, successor: ImageHandler = None):
        super().__init__(successor)

    def handle(self, image: cv2.Mat) -> cv2.Mat:
        return self._successor.handle(image)


class CornerProviderHandler(ImageHandler):
    def __init__(self):
        self.corner_provider = CornerProvider(gui_window_name="Corner Selection Preview")

    def handle(self, image: cv2.Mat) -> cv2.Mat:
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = pipe_func.quadrilateral_to_rectangle(image, corners)
        return self._successor.handle(whiteboard)


class ForegroundRemoverHandler(ImageHandler):
    def __init__(self):
        # Alternative models
        # torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.torch_model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "deeplabv3_mobilenet_v3_large",
            weights="DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT"
        )
        self.torch_model.eval()

    def handle(self, image: cv2.Mat) -> cv2.Mat:
        foreground_mask = pipe_func.segment(self.torch_model, image)
        return self._successor.handle((image, foreground_mask))


class IdealizeColorsHandler(ImageHandler):
    def __init__(self, successor: ImageHandler = None):
        super().__init__(successor)

    def handle(self, data: tuple[cv2.Mat, cv2.Mat]) -> cv2.Mat:
        whiteboard, foreground_mask = data
        whiteboard = pipe_func.idealize_colors(whiteboard, IdealizeColorsMode.MASKING)
        return self._successor.handle((whiteboard, foreground_mask))


class InpainterHandler(ImageHandler):
    def __init__(self):
        self._last_image = None

    def handle(self, data: tuple[cv2.Mat, cv2.Mat]) -> cv2.Mat:
        whiteboard, foreground_mask = data
        whiteboard = pipe_func.inpaint_missing(whiteboard, foreground_mask, self._last_image)
        self._last_image = whiteboard
        return self._successor.handle(whiteboard)


class FinalHandler (ImageHandler):
    def __init__(self, successor: ImageHandler = None):
        super().__init__(successor)

    def handle(self, image: cv2.Mat) -> cv2.Mat:
        return image
