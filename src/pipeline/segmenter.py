import torch
import cv2 as cv
from torchvision import transforms
from ..helper import dilate_black_regions
import numpy as np


class Segmentor:
    torch_model = None

    def __init__(self):
        # Alternative models
        # torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        self.torch_model = torch.hub.load(
            "pytorch/vision:v0.10.0", 
            "deeplabv3_mobilenet_v3_large", 
            weights="DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT"
        )
        self.torch_model.eval()

    def segment(self, img: np.ndarray) -> np.ndarray:
        input_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            self.torch_model.to("cuda")

        with torch.no_grad():
            output = self.torch_model(input_batch)["out"][0]
        output_predictions = output.argmax(0)

        prediction_in_numpy = output_predictions.byte().cpu().numpy()

        mask = cv.inRange(prediction_in_numpy, 0, 0)
        mask = dilate_black_regions(mask, iterations=11)

        return mask
