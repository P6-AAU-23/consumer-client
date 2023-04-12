import time


def RemoveSegmentAct(binImg, segImg):
    timeStamp = time.time()

    sizeOfSegImg = segImg.shape

    for colPx in range(sizeOfSegImg[0]):
        for rowPx in range(sizeOfSegImg[1]):
            if segImg[colPx, rowPx] == 0:
                binImg[colPx, rowPx] = 0

    print("Remove Segmentation:" + str(time.time() - timeStamp))

    return binImg
