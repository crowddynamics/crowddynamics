import logging
from multiprocessing import Process, Event, Queue
from moviepy.editor import *


class WriteMovie(Process):
    def __init__(self, queue: Queue):
        super(WriteMovie, self).__init__()
        self.queue = queue
        self.exit = Event()

    def stop(self):
        logging.info("")
        self.exit.set()

    def run(self):
        logging.info("Start")
        while not self.exit.is_set():
            self.update()
        logging.info("End")

    def update(self):
        image = self.queue.get()
