import timeit
import cv2
import os
import sys


from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from src.pipeline.pipeline import Pipeline
from src.pipeline.pipeline import Pipeline

SCALE = 1

pipeline = Pipeline()
    # timeit.repeat(stmt='pass', setup='pass', timer=<default timer>, repeat=5, number=1000000, globals=None)

def mockImages():
    image = []
    
    for i in range(0,3):
       image.append(cv2.imread(f"resources/benchmark{i}.jpg"))
    return image


def otherBench():
    for img in mockImages():
        whiteboard = pipeline.process(img)
        cv2.imshow("benchmark", whiteboard)
        cv2.waitKey(1)
    return whiteboard

otherBench()

timeit.repeat(otherBench, setup="pass", repeat=1, number=SCALE)