import cv2 as cv
import os
from pathlib import Path
from MaskingPipeline.CaptureActivity import CaptureActivity

ROOT_DIR = os.path.realpath(os.path.dirname(__file__))+'\\'

relativePath = Path('Tests/Images')
#fullPath = relativePath / 'whiteboard1.png'
fullPath = relativePath / 'hellowfellow.jpg'
#relativePath = 'Tests\Images\whiteboard1.png'
#fullPath = os.path.join(ROOT_DIR, relativePath)
image = cv.imread(str(fullPath))


cv.waitKey(0)
CA = CaptureActivity(image)
CA.CaptureActivity(image)

#IMG_DIR = "Tests/Images"
#images = Path(IMG_DIR).glob('*') 
#image = cv.imread(str(fullPath))

#def thisShouldWorkForIMGSTREAMbutSinceThomasWantMeToFinishSegmenTationIllgoAndDoThatBigSad
#for image in images:
#    image = cv.imread(str(fullPath))
#    cv.waitKey(0)
#    CA.CaptureActivity(image)