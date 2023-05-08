import argparse
from .pipeline_modules import (
    ColorIdealizer,
    ForegroundRemover,
    IdealizeColorsMode,
    IdentityProcessor,
    ImageProcessor,
    Inpainter,
    PerspectiveTransformer,
    WipeSaver,
)


def pipeline_builder(args: argparse.Namespace) -> ImageProcessor:
    start = IdentityProcessor()
    perspective_transformer = PerspectiveTransformer()
    foreground_remover_handler = ForegroundRemover()
    idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)
    inpainter_handler = Inpainter()
    wipe_saver = WipeSaver(args.saved_path)

    head = start

<<<<<<< HEAD
    def process(self, image: np.ndarray, avg_color: "Avg_bgr", sat: float, bright: int) -> np.ndarray:
        self.corner_provider.update(image)
        corners = self.corner_provider.get_corners()
        whiteboard = quadrilateral_to_rectangle(image, corners)
        foreground_mask = self.foreground_remover.segment(whiteboard)
        whiteboard = color_adjust(whiteboard, avg_color, sat, bright)
        whiteboard = idealize_colors(whiteboard, IdealizeColorsMode.MASKING)
        whiteboard = self.inpainter.inpaint_missing(whiteboard, foreground_mask)
        return whiteboard
=======
    if not args.disable_transform_perspective:
        head = head.set_next(perspective_transformer)
>>>>>>> main

    if not args.disable_remove_foreground:
        head = head.set_next(foreground_remover_handler)

    if not args.disable_idealize_colors:
        head = head.set_next(idealize_colors_handler)

    if not args.disable_remove_foreground:
        head = head.set_next(inpainter_handler)

    if args.save_on_wipe:
        head = head.set_next(wipe_saver)

<<<<<<< HEAD
class Avg_bgr:
    def __init__(self, avg_b: float, avg_g: float, avg_r: float):
        self.b = avg_b
        self.g = avg_g
        self.r = avg_r

    def white_balance(self, image: np.ndarray) -> np.ndarray:
        # Split channels
        blue, green, red = cv2.split(image)

        # Calculate scaling factors for each channel
        scale_b = self.g / self.b
        scale_r = self.g / self.r

        # Apply scaling factors to each channel
        blue = cv2.convertScaleAbs(blue, alpha=scale_b)
        red = cv2.convertScaleAbs(red, alpha=scale_r)

        # Merge channels
        result = cv2.merge((blue, green, red))
        return result


def color_adjust(image: cv2.Mat, avg_color: Avg_bgr, saturate_input: float, bright_input: int) -> cv2.Mat:
    """
    Apply white balancing to an input image using a pre-calculated average of B, G, R channels.
    Also Applying saturation, brightness, and normalization.

    :param image: Input image as a numpy array.
    :type image: numpy.ndarray
    :return: Color adjusted image as a numpy array.
    :rtype: numpy.ndarray
    """

    # Applying white balancing
    result = avg_color.white_balance(image)

    # Up saturation & brightness
    saturation_boost = saturate_input
    brightness = bright_input

    if saturate_input != 1 or bright_input != 0:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        if saturate_input != 1:
            result[:, :, 1] = cv2.convertScaleAbs(result[:, :, 1], alpha=saturation_boost)
        if bright_input != 0:
            result[:, :, 2] = cv2.add(result[:, :, 2], brightness)
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    # Normalize image
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    return result


=======
    return start
>>>>>>> main
