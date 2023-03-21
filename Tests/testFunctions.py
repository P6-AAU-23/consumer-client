from pathlib import Path
import os

def CheckIfImageIsPassed(image, pipelineStep):
    if image is None:
        result = "Image is empty!!"
    else:
        result = "Image is not empty!!"
  
    print(pipelineStep+': '+result)

def GetPath():
    ROOT_DIR = os.path.realpath(os.path.dirname(__file__)+'/..')
    relativePath = Path('Tests/Images')
    return ROOT_DIR / relativePath 