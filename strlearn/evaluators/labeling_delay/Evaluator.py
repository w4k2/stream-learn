import abc
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from .utils import LabelingProcess


class Evaluator(abc.ABC):
    def __init__(self, metrics=(balanced_accuracy_score,), labeling_delay=10, partial=True, verbose=False):
        self.metrics = metrics
        self.labeling_delay = labeling_delay
        assert labeling_delay > 0
        self.labeling_process = LabelingProcess(labeling_delay)
        self.partial = partial
        self.verbose = verbose

    @abc.abstractmethod
    def process(self, stream, clf, detector=None):
        raise NotImplementedError

    def train_model(self, clf, X, y):
        if self.partial:
            clf.partial_fit(X, y, np.unique(y))
        else:
            clf.fit(X, y)
