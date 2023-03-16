import cv2 as cv
import os
from pathlib import Path
from MaskingPipeline.CaptureActivity import CaptureActivity

#thomas old stuff
#ROOT_DIR = os.path.realpath(os.path.dirname(__file__))+'\\'
#relativePath = 'Tests\Images\whiteboard1.png'
#fullPath = os.path.join(ROOT_DIR, relativePath)

#my cooler and newer fix
ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
relativePath = Path('Tests/Images')
fullPath = ROOT_DIR / relativePath / 'whiteboard1.png'

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