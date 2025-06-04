from collections import deque


class LabelingProcess:
    """Class that simulates the delay in labeling process with priority queue"""

    def __init__(self, delay):
        self.delay = delay
        self.current_counter = -1
        self.buffer = deque([], maxlen=delay)

    def request_annotation(self, X, y):
        if self.current_counter <= 0 and len(self.buffer) == 0:
            self.current_counter = self.delay

        if len(self.buffer) == self.delay:
            raise OverflowError

        self.buffer.append((X, y))

    def update_time(self):
        self.current_counter = max(self.current_counter - 1, -1)

    def retrive_annotated(self):
        if len(self.buffer) == 0:
            return

        if self.current_counter <= 0:
            X, y = self.buffer.popleft()
            return X, y

    def labels_avaliable(self):
        return len(self.buffer) > 0 and self.current_counter <= 0

    def peding_labeling(self):
        return len(self.buffer) > 0
