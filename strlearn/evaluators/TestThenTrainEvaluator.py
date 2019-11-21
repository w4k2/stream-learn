"""Test Than Train evaluator."""

import numpy as np
from sklearn.metrics import accuracy_score
from ..utils import bac, geometric_mean_score, recall
from sklearn.base import ClassifierMixin


class TestThenTrainEvaluator:
    """
    Test Than Train data stream evaluator.

    Implementation of test-then-train evaluation procedure,
    where each individual data chunk is first used to test
    the classifier and then it is used for training.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes, )
        The class labels.
    scores_ : array-like, shape (stream.n_chunks, 5)
        Values of accuracy_score, roc_auc_score,
        geometric_mean_score, bac and f_score for
        each processed data chunk.

    Examples
    --------
    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.classifiers.AccumulatedSamplesClassifier()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
    [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.92       0.91936979 0.91936231 0.91936979 0.9047619 ]
    ...
    [0.92       0.91907051 0.91877671 0.91907051 0.9245283 ]
    [0.885      0.8854889  0.88546135 0.8854889  0.87830688]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(self, cut=0):
        self.cut = cut

    def process(self, stream, clfs, metrics=(accuracy_score, bac)):
        """
        Perform learning procedure on data stream.

        Parameters
        ----------
        clf : scikit-learn estimator
            Classifier implementing a `partial_fit()` method.
        stream : object
            Data stream as an object.
        """
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs = [clfs]
        else:
            self.clfs = clfs

        # Assign parameters
        self.stream = stream
        self.metrics = metrics

        # Prepare scores table
        self.scores_ = np.zeros(
            (
                len(self.clfs),
                ((stream.n_chunks - 1) if self.cut == 0 else self.cut),
                len(self.metrics),
            )
        )

        self.classes_ = stream.classes

        while True:
            X, y = stream.get_chunk()

            # Test
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs):
                    y_pred = clf.predict(X)

                    self.scores_[clfid, stream.chunk_id - 1] = [
                        metric(y, y_pred) for metric in self.metrics
                    ]

            # Train
            [clf.partial_fit(X, y, self.classes_) for clf in self.clfs]

            if self.cut > 0 and stream.chunk_id == self.cut:
                break

            if stream.is_dry():
                break
