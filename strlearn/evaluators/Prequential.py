"""Prequential data stream evaluator."""

import numpy as np
from sklearn.metrics import accuracy_score
from ..metrics import balanced_accuracy_score
from sklearn.base import ClassifierMixin


class Prequential:
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
    >>> evaluator = sl.evaluators.PrequentialEvaluator()
     >>> evaluator.process(clf, stream, interval=50)
    >>> print(evaluator.scores_)
    ...
    [[0.95       0.9483469  0.94805282 0.9483469  0.95412844]
    [0.96       0.95728313 0.95696445 0.95728313 0.96460177]
    [0.96       0.95858586 0.95848154 0.95858586 0.96396396]
    ...
    [0.92       0.91987179 0.91986621 0.91987179 0.91666667]
    [0.91       0.91065705 0.91050889 0.91065705 0.90816327]
    [0.925      0.92567027 0.9250634  0.92567027 0.92610837]]
    """

    def __init__(self, metrics=(accuracy_score, balanced_accuracy_score)):
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]

    def process(self, stream, clfs, interval=100):
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
        """

        if isinstance(clfs, ClassifierMixin):
            self.clfs = [clfs]
        else:
            self.clfs = clfs

        # Assign parameters
        self.stream_ = stream
        self.interval_ = interval

        intervals_per_chunk = int(self.stream_.chunk_size / self.interval_)
        self.scores = np.zeros(
            (
                len(self.clfs),
                ((stream.n_chunks - 1) * intervals_per_chunk),
                len(self.metrics),
            )
        )

        i = 0
        while True:
            stream.get_chunk()
            a, _ = stream.current_chunk
            # break

            if stream.previous_chunk is not None:
                X_p, y_p = stream.previous_chunk
                X_c, y_c = stream.current_chunk

                X = np.concatenate((X_p, X_c), axis=0)
                y = np.concatenate((y_p, y_c), axis=0)

                for interval_id in range(1, intervals_per_chunk + 1):
                    start = interval_id * interval
                    end = start + self.stream_.chunk_size

                    for clfid, clf in enumerate(self.clfs):
                        y_pred = clf.predict(X[start:end])

                        self.scores[clfid, i] = [
                            metric(y[start:end], y_pred) for metric in self.metrics
                        ]

                    [clf.partial_fit(X[start:end], y[start:end]) for clf in self.clfs]

                    i += 1
            else:
                X_train, y_train = stream.current_chunk
                [
                    clf.partial_fit(X_train, y_train, self.stream_.classes_)
                    for clf in self.clfs
                ]

            if stream.is_dry():
                break
