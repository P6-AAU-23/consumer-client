from .pipeline_modules import *


def pipeline_builder(args):
    start_handler = StartHandler()
    corner_provider_handler = CornerProviderHandler()
    foreground_remover_handler = ForegroundRemoverHandler()
    idealize_colors_handler = IdealizeColorsHandler(IdealizeColorsMode.MASKING)
    inpainter_handler = InpainterHandler()
    final_handler = FinalHandler()

    head = start_handler

    if not args.disable_transform_perspective:
        head.set_successor(corner_provider_handler)
        head = corner_provider_handler

    if not args.disable_remove_foreground:
        head.set_successor(foreground_remover_handler)
        head = foreground_remover_handler

    
    if not args.disable_idealize_colors:
        head.set_successor(idealize_colors_handler)
        head = idealize_colors_handler

    if not args.disable_remove_foreground:
        head.set_successor(inpainter_handler)
        head = inpainter_handler


    head.set_successor(final_handler)

    return start_handler
