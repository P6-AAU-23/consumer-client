import pykka

from CornerProvider import CornerProvider
from helper import quadrilateral_to_rectangle

class Pipeline(pykka.ThreadingActor):
    def __init__(self, args, video_capture):
        super().__init__()
        self.corner_provider = CornerProvider.start('Corner Selection Preview').proxy()
        self.args = args
        self.video_capture = video_capture
        ret, image = self.video_capture.read()
        self.corner_provider.update(image)
        while True:
            ret, image = self.video_capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            corners = self.corner_provider.get_corners().get()
            self.corner_provider.update(self, image)
            whiteboard = quadrilateral_to_rectangle(image, corners)
            cv2.imshow('preview', whiteboard) # type: ignore
            if cv2.waitKey(1) == ord('q'): # type: ignore
                break
            self.stop()

    def stop(self):
        self.video_capture.release()
        cv2.destroyAllWindows() # type: ignore
        self.corner_provider.stop()
        super().stop()
        
