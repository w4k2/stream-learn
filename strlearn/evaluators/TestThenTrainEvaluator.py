from importlib.machinery import SourceFileLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from ..utils import bac, f_score

METRICS = (accuracy_score, roc_auc_score, geometric_mean_score, bac, f_score)


class TestThenTrainEvaluator:
    def __init__(self):
        pass

    def process(self, clf, stream):
        self.clf = clf
        self.stream = stream

        self.scores = np.zeros((stream.n_chunks - 1, len(METRICS)))

        self.classes = np.array(range(stream.n_classes))

        while True:
            X_train, y_train = stream.get_chunk()

            if stream.previous_chunk is not None:
                X_test, y_test = stream.previous_chunk
                y_pred = clf.predict(X_test)

                self.scores[stream.chunk_id - 1] = [
                    metric(y_test, y_pred) for metric in METRICS
                ]

            clf.partial_fit(X_train, y_train, self.classes)

            if stream.is_dry():
                break
