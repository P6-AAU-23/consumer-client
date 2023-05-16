import argparse
from ..helper import AvgBgr
from .pipeline_modules import *  # noqa: F403


def pipeline_builder(args: argparse.Namespace, avg_bgr: AvgBgr) -> ImageProcessor:  # noqa: F405
    start = IdentityProcessor()  # noqa: F405
    perspective_transformer = PerspectiveTransformer()  # noqa: F405
    foreground_remover_handler = ForegroundRemover()  # noqa: F405
    color_adjuster_handler = ColorAdjuster(avg_bgr, args.saturation, args.brightness)  # noqa: F405
    idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)  # noqa: F405
    inpainter_handler = Inpainter()  # noqa: F405
    wipe_saver = WipeSaver(args.saved_path)  # noqa: F405

    head = start

    if not args.disable_transform_perspective:
        head = head.set_next(perspective_transformer)

    if not args.disable_remove_foreground:
        head = head.set_next(foreground_remover_handler)

    if not args.disable_color_adjuster:
        head = head.set_next(color_adjuster_handler)

    if not args.disable_idealize_colors:
        head = head.set_next(idealize_colors_handler)

    if not args.disable_remove_foreground:
        head = head.set_next(inpainter_handler)

    if args.save_on_wipe:
        head = head.set_next(wipe_saver)

    return start
