from Pipeline import Binarization
from Pipeline import Changes
from Pipeline import Colourization
from Pipeline import Segmentation
from Pipeline import UpdateWhiteboard



def CaptureActivity():
    Segmentation.SegmentAct()
    Binarization.BinerizeAct()
    Changes.ChangesAct()
    Colourization.ColouringAct()
    UpdateWhiteboard.UpdateWhiteboardAct()

CaptureActivity()