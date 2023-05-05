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

    if not args.disable_transform_perspective:
        head = head.set_next(perspective_transformer)

    if not args.disable_remove_foreground:
        head = head.set_next(foreground_remover_handler)

    if not args.disable_idealize_colors:
        head = head.set_next(idealize_colors_handler)

    if not args.disable_remove_foreground:
        head = head.set_next(inpainter_handler)

    if args.save_on_wipe:
        head = head.set_next(wipe_saver)

    return start
