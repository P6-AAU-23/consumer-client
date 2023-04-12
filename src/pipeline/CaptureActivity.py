import numpy as np
import time


from MaskingPipeline.Pipeline import (
    Binarization,
    Changes,
    Colourization,
    Segmentation,
    UpdateWhiteboard,
    SegmentationRemoval,
)


class CaptureActivity:
    VirtualWhiteboard = None
    Seg = Segmentation.Segmentator()

    def __init__(self, img):
        self.VirtualWhiteboard = np.full(img.shape, 255, dtype=np.uint8)

    def CaptureActivityAct(self, img):
        timestamp = time.time()

        origImg = img

        binImg = Binarization.BinarizeAct(img)
        segImg = self.Seg.SegmentAct(img)

        removedSegBinImg = SegmentationRemoval.RemoveSegmentAct(binImg, segImg) # noqa: 841
        TrackedChangesMask = Changes.ChangesAct(binImg, segImg)
        ColouredChanges = Colourization.ColouringAct(TrackedChangesMask, origImg)
        self.VirtualWhiteboard = UpdateWhiteboard.UpdateWhiteboardAct(
            ColouredChanges, self.VirtualWhiteboard
        )

        print((time.time() - timestamp))
