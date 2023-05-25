import argparse
from ..helper import AvgBgr
from .pipeline_modules import *  # noqa: F403


def pipeline_builder(args: argparse.Namespace, avg_bgr: AvgBgr) -> ImageProcessor:  # noqa: F405
    start = IdentityProcessor()  # noqa: F405
    perspective_transformer = PerspectiveTransformer()  # noqa: F405
    fast_foreground_masker = FastForegroundMasker()  # noqa: F405
    medium_foreground_masker = MediumForegroundMasker()  # noqa: F405
    slow_foreground_masker = SlowForegroundMasker()  # noqa: F405
    color_adjuster = ColorAdjuster(avg_bgr, args.saturation, args.brightness)  # noqa: F405
    color_idealizer = ColorIdealizer(IdealizeColorsMode.MASKING)  # noqa: F405
    inpainter = Inpainter()  # noqa: F405
    wipe_saver = WipeSaver(args.saved_path)  # noqa: F405

    head = start

    if not args.disable_transform_perspective:
        head = head.set_next(perspective_transformer)

    if not args.disable_remove_foreground:
        if args.fast and not args.slow:
            head = head.set_next(fast_foreground_masker)
        elif args.slow and not args.fast:
            head = head.set_next(slow_foreground_masker)
        else:
            head = head.set_next(medium_foreground_masker)

    if not args.disable_color_adjuster:
        head = head.set_next(color_adjuster)

    if not args.disable_idealize_colors:
        head = head.set_next(color_idealizer)

    if not args.disable_remove_foreground:
        head = head.set_next(inpainter)

    if args.save_on_wipe:
        head = head.set_next(wipe_saver)

    return start
