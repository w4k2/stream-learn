import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from ..metrics import balanced_accuracy_score


class TestThenTrain:
    """
    Test Than Train data stream evaluator.

    Implementation of test-then-train evaluation procedure,
    where each individual data chunk is first used to test
    the classifier and then it is used for training.

    :type metrics: tuple or function
    :param metrics: Tuple of metric functions or single metric function.
    :type verbose: boolean
    :param verbose: Flag to turn on verbose mode.

    :var classes_: The class labels.
    :var scores_: Values of metrics for each processed data chunk.
    :vartype classes_: array-like, shape (n_classes, )
    :vartype scores_: array-like, shape (stream.n_chunks, len(metrics))

    :Example:

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

    def __init__(
        self, metrics=(accuracy_score, balanced_accuracy_score), verbose=False
    ):
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]
        self.verbose = verbose

    def process(self, stream, clfs):
        """
        Perform learning procedure on data stream.

        :param stream: Data stream as an object
        :type stream: object
        :param clfs: scikit-learn estimator of list of scikit-learn estimators.
        :type clfs: tuple or function
        """
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs_ = [clfs]
        else:
            self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        self.scores = np.zeros(
            (len(self.clfs_), ((self.stream_.n_chunks - 1)), len(self.metrics))
        )

        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        while True:
            X, y = stream.get_chunk()
            if self.verbose:
                pbar.update(1)

            # Test
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs_):
                    y_pred = clf.predict(X)

                    self.scores[clfid, stream.chunk_id - 1] = [
                        metric(y, y_pred) for metric in self.metrics
                    ]

            # Train
            [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]

            if stream.is_dry():
                break
