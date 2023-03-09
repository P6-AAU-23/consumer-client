def CheckIfImageIsPassed(image, pipelineStep):
    if image is None:
        result = "Image is empty!!"
    else:
        result = "Image is not empty!!"
  
    print(pipelineStep+': '+result)