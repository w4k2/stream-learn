"""Prequential evaluator."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from ..utils import bac, f_score, geometric_mean_score

METRICS = (accuracy_score, roc_auc_score, geometric_mean_score, bac, f_score)


class PrequentialEvaluator:
    """
    Prequential data stream evaluator.

    Implementation of prequential evaluation procedure, based
    on sliding windows instead of separate data chunks. Window
    moves by a fixed number of instances in order to preserve
    some of the already processed ones. After each step, samples
    that are currently in the window are used to test the classifier
    and then for training.

    Attributes
    ----------

    """
    def __init__(self):
        pass

    def process(self, clf, stream, interval=100):
        """
        Perform learning procedure on data stream.

        Parameters
        ----------
        clf : scikit-learn estimator
            Classifier implementing a `partial_fit()` method.
        stream : object
            Data stream as an object.
        interval: integer, optional (default=100)
            The number of instances by which the sliding window
            moves before the next evaluation and training steps.

        Attributes
        ----------

        """
        self.clf = clf
        self.stream = stream
        self.interval = interval
        self.classes = np.array(range(stream.n_classes))

        intervals_per_chunk = int(self.stream.chunk_size / self.interval)
        self.scores = np.zeros(
            ((stream.n_chunks - 1) * intervals_per_chunk, len(METRICS))
        )

        i = 0
        while True:
            stream.get_chunk()
            if stream.previous_chunk is not None:
                X_p, y_p = stream.previous_chunk
                X_c, y_c = stream.current_chunk

                X = np.concatenate((X_p, X_c), axis=0)
                y = np.concatenate((y_p, y_c), axis=0)

                for interval_id in range(intervals_per_chunk):
                    start = interval_id * interval
                    end = start + self.stream.chunk_size

                    y_pred = clf.predict(X[start:end])

                    self.scores[i] = [
                        metric(y[start:end], y_pred) for metric in METRICS
                    ]

                    clf.partial_fit(X[start:end], y[start:end])

                    i += 1
            else:
                X_train, y_train = stream.current_chunk
                clf.partial_fit(X_train, y_train, self.classes)

            if stream.is_dry():
                break
