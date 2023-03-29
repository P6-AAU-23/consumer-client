import sys
from Tests.WholePipelineTest import WholePipelineTest
from Tests.segmentationTest import segmentationTest
from Tests.RemoveSegmentFromBinaryTest import RemoveSegmentFromBinaryTest

def main():
    arg = sys.argv[1:]
    TestMain(arg[0])

def TestMain(imgName):
    RemoveSegmentFromBinaryTest()
    segmentationTest(imgName)
    WholePipelineTest(imgName)


if __name__ == '__main__':
    main()