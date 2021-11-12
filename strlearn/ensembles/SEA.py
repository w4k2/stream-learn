import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from ..ensembles.base import StreamingEnsemble


class SEA(StreamingEnsemble):
    """
    Streaming Ensemble Algorithm.

    Ensemble classifier composed of estimators trained on the fixed
    number of previously seen data chunks, prunning the worst one in the pool.

    :type n_estimators: integer, optional (default=10)
    :param n_estimators: The maximum number of estimators trained using consecutive data chunks and maintained in the ensemble.
    :type metric: function, optional (default=accuracy_score)
    :param metric: The metric used to prune the worst classifier in the pool.

    :vartype ensemble_: list of classifiers
    :var ensemble_: The collection of fitted sub-estimators.
    :vartype classes_: array-like, shape (n_classes, )
    :var classes_: The class labels.

    :Example:

    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.ensembles.SEA()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
    [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.925      0.92364329 0.92360881 0.92364329 0.91017964]
    ...
    [0.925      0.92427885 0.924103   0.92427885 0.92890995]
    [0.89       0.89016179 0.89015879 0.89016179 0.88297872]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(self, base_estimator=None, n_estimators=10, metric=accuracy_score):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.metric = metric

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Append new estimator
        self.ensemble_.append(clone(self.base_estimator).fit(self.X_, self.y_))

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            del self.ensemble_[
                np.argmin([self.metric(y, clf.predict(X)) for clf in self.ensemble_])
            ]
        return self
