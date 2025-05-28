import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


class ContinousRebuild:
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

    def __init__(self, metrics=(balanced_accuracy_score,), labeling_delay=10, partial=True, verbose=False):
        self.metrics = metrics
        self.labeling_delay = labeling_delay
        self.partial = partial
        self.verbose = verbose

    def process(self, stream, clf):
        self.scores = []
        self.label_request_chunks = []
        self.training_chunks = []

        pending_labeling_requests = []

        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        for chunk_id, (X, y) in enumerate(stream):
            if self.verbose:
                pbar.update(1)

            if chunk_id == 0:
                if self.partial == True:
                    clf.partial_fit(X, y, np.unique(y))
                else:
                    clf.fit(X, y)
                continue

            if chunk_id-self.labeling_delay in pending_labeling_requests:
                start = stream.chunk_size * (chunk_id-self.labeling_delay)
                end = stream.chunk_size * (chunk_id-self.labeling_delay) + stream.chunk_size

                past_X = stream.X[start:end]
                past_y = stream.y[start:end]

                if self.partial:
                    clf.partial_fit(past_X, past_y, np.unique(past_y))
                else:
                    clf.fit(past_X, past_y)
                self.training_chunks.append(chunk_id)

                pending_labeling_requests.remove(chunk_id-self.labeling_delay)

            pending_labeling_requests.append(chunk_id)
            self.label_request_chunks.append(chunk_id)

            preds = clf.predict(X)
            self.scores.append(metric(y, preds) for metric in self.metrics)
