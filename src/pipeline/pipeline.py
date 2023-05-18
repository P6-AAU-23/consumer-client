import argparse
from ..helper import AvgBgr
from .pipeline_modules import *  # noqa: F403


def pipeline_builder(args: argparse.Namespace, avg_bgr: AvgBgr) -> ImageProcessor:  # noqa: F405
    start = IdentityProcessor()  # noqa: F405
    perspective_transformer = PerspectiveTransformer()  # noqa: F405
    fast_foreground_remover = FastForegroundRemover()  # noqa: F405
    medium_foreground_remover = MediumForegroundRemover()  # noqa: F405
    slow_foreground_remover = SlowForegroundRemover()  # noqa: F405
    color_adjuster_handler = ColorAdjuster(avg_bgr, args.saturation, args.brightness)  # noqa: F405
    idealize_colors_handler = ColorIdealizer(IdealizeColorsMode.MASKING)  # noqa: F405
    inpainter_handler = Inpainter()  # noqa: F405
    wipe_saver = WipeSaver(args.saved_path)  # noqa: F405

    head = start

    if not args.disable_transform_perspective:
        head = head.set_next(perspective_transformer)

    if not args.disable_remove_foreground:
        if args.fast and not args.slow:
            head = head.set_next(fast_foreground_remover)
        elif args.slow and not args.fast:
            head = head.set_next(slow_foreground_remover)
        else:
            head = head.set_next(medium_foreground_remover)

    if not args.disable_color_adjuster:
        head = head.set_next(color_adjuster_handler)

    if not args.disable_idealize_colors:
        head = head.set_next(idealize_colors_handler)

    if not args.disable_remove_foreground:
        head = head.set_next(inpainter_handler)

    if args.save_on_wipe:
        head = head.set_next(wipe_saver)

    return start
