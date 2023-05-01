import timeit
import cv2
import os
import sys

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
#from src.pipeline.pipeline import Pipeline
print(sys.path)
from src.pipeline.pipeline import Pipeline



print(sys.path)
project_root = os.path.dirname(os.path.abspath(__file__))  # + "/.."
#test_root = path_root + os.path.abspath(__file__) + "/.."
#sys.path.append(test_root)
print("ignore next ///////")
SCALE = 1000
print("this is {project_root}")
print(project_root)
print("ignore next ///////")
#print(test_root)
path2= project_root + "/benchmark11.jpg"
print(path2)
mockImages = cv2.imread("path2")

print(mockImages.size)
cv2.imshow('meh', mockImages)
pipeline = Pipeline()
    # timeit.repeat(stmt='pass', setup='pass', timer=<default timer>, repeat=5, number=1000000, globals=None)

def mockImaages():
    image = {}
    
    
    for i in range(3):
       image = cv2.imread("{project_root}/resources/benchmark{i}.jpg")
    return image

def bench1(img):
    return img

#mimg = mockImaages()
def otherBench():
    whiteboard = pipeline.process(mockImages)
    cv2.imshow("benchmark", whiteboard)
    cv2.waitKey(1)
    return whiteboard

bench1(mockImages)
otherBench()

timeit.repeat(otherBench, setup="pass", repeat=5, number=1000)