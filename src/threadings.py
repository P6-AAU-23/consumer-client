from typing import Any
import pykka


class Threadpool:
    def __init__(self, numberOfWorkers):
        self.numberOfWorkers = numberOfWorkers
        self.actors = []

    def start():
        pass
    def stop(self):
        for actor in self.actors:
            actor.stop


class Slave(pykka.ThreadingActor):
    def __init__(self):
        super().__init__()

    def on_receive(self, message: Any) -> Any:
        return message
        #return super().on_receive(Any)
        #if callable(message):
         #   try:
          #      result = message()
           # except Exception as e:
            #    result = e
          #  return result
       # return None
    def stop(self):
        self.stop
