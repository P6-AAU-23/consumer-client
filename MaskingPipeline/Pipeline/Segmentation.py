import torch
import cv2 as cv
from torchvision import transforms
import time


class Segmentor:
    # torchModel = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # torchModel = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    torchModel = None

    def __init__(self):
        self.torchModel = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_mobilenet_v3_large", pretrained=True
        )
        self.torchModel.eval()

    def SegmentAct(self, img):
        timeStamp = time.time()

        inputImage = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        inputTensor = preprocess(inputImage)
        inputBatch = inputTensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            inputBatch = inputBatch.to("cuda")
            self.torchModel.to("cuda")

        with torch.no_grad():
            output = self.torchModel(inputBatch)["out"][0]
        outputPredictions = output.argmax(0)

        predictionInNumpy = outputPredictions.byte().cpu().numpy()

        mask = cv.inRange(predictionInNumpy, 0, 0)

        print("Segmentation:" + str((time.time() - timeStamp)))

        return mask
