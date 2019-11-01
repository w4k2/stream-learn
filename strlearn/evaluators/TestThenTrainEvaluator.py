"""Test Than Train evaluator."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from ..utils import bac, f_score, geometric_mean_score

METRICS = (accuracy_score, roc_auc_score, geometric_mean_score, bac, f_score)


class TestThenTrainEvaluator:
    """
    Test Than Train data stream evaluator.

    Implementation of test-then-train evaluation procedure,
    where each individual data chunk is first used to test
    the classifier and then it is used for training.

    Attributes
    ----------

    """

    def __init__(self, cut=0):
        self.cut = cut

    def process(self, clf, stream):
        """
        Perform learning procedure on data stream.

        Parameters
        ----------
        clf : scikit-learn estimator
            Classifier implementing a `partial_fit()` method.
        stream : object
            Data stream as an object.

        Attributes
        ----------

        """
        self.clf = clf
        self.stream = stream

        self.scores = np.zeros(
            (((stream.n_chunks - 1) if self.cut == 0 else self.cut), len(METRICS))
        )

        self.classes = np.array(range(stream.n_classes))

        while True:
            X, y = stream.get_chunk()
            print("CHUNK %i" % stream.chunk_id)

            if stream.previous_chunk is not None:
                y_pred = self.clf.predict(X)

                self.scores[stream.chunk_id - 1] = [
                    metric(y, y_pred) for metric in METRICS
                ]

            clf.partial_fit(X, y, self.classes)

            if self.cut > 0 and stream.chunk_id == self.cut:
                break

            if stream.is_dry():
                break
