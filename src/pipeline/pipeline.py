import argparse
from .helper import AvgBgr
from .pipeline_modules import (
    ColorIdealizer,
    ColorAdjuster,
    ForegroundRemover,
    IdealizeColorsMode,
    IdentityProcessor,
    ImageProcessor,
    Inpainter,
    PerspectiveTransformer,
    WipeSaver,
)


def pipeline_builder(args: argparse.Namespace, avg_bgr: AvgBgr) -> ImageProcessor:
    start = IdentityProcessor()
    perspective_transformer = PerspectiveTransformer()
    foreground_remover_handler = ForegroundRemover()
    color_adjuster_handler = ColorAdjuster(avg_bgr, args.saturation, args.brightness)
    idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)
    inpainter_handler = Inpainter()
    wipe_saver = WipeSaver(args.saved_path)

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
