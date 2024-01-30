import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from ..metrics import balanced_accuracy_score


class SparseTrainDenseTest:
    """
    Sparse Train Dense Test data stream evaluator.

    Implementation of sparse-train-dense-test evaluation procedure,
    where each individual data chunk is first used to test
    the classifier and then it is used for given number of chunks of training.

    :type metrics: tuple or function
    :param metrics: Tuple of metric functions or single metric function.
    :type verbose: boolean
    :param verbose: Flag to turn on verbose mode.

    :var classes_: The class labels.
    :var scores_: Values of metrics for each processed data chunk.
    :vartype classes_: array-like, shape (n_classes, )
    :vartype scores_: array-like, shape (stream.n_chunks, len(metrics))
    """

    def __init__(
        self, n_repeats=5, metrics=(accuracy_score, balanced_accuracy_score), verbose=False
    ):
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]
        self.n_repeats = n_repeats
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
        while chunk := stream.get_chunk():
            X, y = chunk
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
            if stream.chunk_id % self.n_repeats == 0:
                [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
