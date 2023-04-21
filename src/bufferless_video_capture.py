import cv2
import queue
import threading
import numpy as np
from typing import Any


class BufferlessVideoCapture:
    def __init__(self, name: str):
        self.cap = cv2.VideoCapture(name)  # type: ignore
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self._reader, args=(self.stop_event,))
        self.t.daemon = True
        self.t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self, stop_event: Any) -> None:
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self) -> np.ndarray:
        return self.q.get()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.stop_event.set()
        self.t.join()
        self.cap.release()

    def __del__(self):
        self.release()
