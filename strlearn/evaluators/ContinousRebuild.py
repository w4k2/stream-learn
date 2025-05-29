import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

import abc

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


class Evaluator(abc.ABC):
    def __init__(self, metrics=(balanced_accuracy_score,), labeling_delay=10, partial=True, verbose=False):
        self.metrics = metrics
        self.labeling_delay = labeling_delay
        self.labeling_process = LabelingProcess(labeling_delay)
        self.partial = partial
        self.verbose = verbose

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError

    def train_model(self, clf, X, y):
        if self.partial:
            clf.partial_fit(X, y, np.unique(y))
        else:
            clf.fit(X, y)


class ContinousRebuild(Evaluator):
    """
    Continous rebuild of classified with all data chunks in the stream

    :param metrics: Set of metrics used for evaluation
    :type metrics: tuple or list
    :param labeling_delay: Number of chunks for the labels to arrive since explicit request
    :type labeling_delay: int
    :param partial: Wheather partial fit or fit
    :type partial: bool
    :param verbose: Flag to turn on verbose mode.
    :type verbose: boolean
    """

    def process(self, stream, clf):
        self.scores = []
        # self.label_request_chunks = []
        # self.training_chunks = []

        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        for chunk_id, (X, y) in enumerate(stream):
            if self.verbose:
                pbar.update(1)

            if chunk_id == 0:
                self.train_model(clf, X, y)
                continue

            self.labeling_process.update_time()
            if self.labeling_process.labels_avaliable():
                past_X, past_y = self.labeling_process.retrive_annotated()
                self.train_model(clf, past_X, past_y)
                # self.training_chunks.append(chunk_id)

            self.labeling_process.request_annotation(X, y)
            # self.label_request_chunks.append(chunk_id)

            preds = clf.predict(X)
            self.scores.append([metric(y, preds) for metric in self.metrics])

        self.scores = np.array(self.scores)
