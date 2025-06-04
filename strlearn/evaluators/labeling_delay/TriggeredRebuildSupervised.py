import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from .Evaluator import Evaluator


class TriggeredRebuildSupervised(Evaluator):
    """
    Triggered Rebuild with Supervised Drift Detection

    :param metrics: Set of metrics used for evaluation
    :type metrics: tuple or list
    :param labeling_delay: Number of chunks for the labels to arrive since explicit request
    :type labeling_delay: int
    :param partial: Wheather partial fit or fit
    :type partial: bool
    :param verbose: Flag to turn on verbose mode.
    :type verbose: boolean
    """

    def process(self, stream, clf, drift_detector):
        self.scores = []
        past_preds = []

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
                pp = past_preds.pop(0)
                drift_detector.feed(past_X, past_y, pp)

                if drift_detector._is_drift:
                    self.train_model(clf, X, y)

            preds = clf.predict(X)
            self.scores.append([metric(y, preds) for metric in self.metrics])

            self.labeling_process.request_annotation(X, y)
            past_preds.append(preds)

        self.scores = np.array(self.scores)
