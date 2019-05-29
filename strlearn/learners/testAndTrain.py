"""Learnerr module."""
from sklearn import base
import time
import csv
from sklearn import neural_network
from strlearn import controllers
from sklearn.metrics import accuracy_score
import numpy as np


class TestAndTrain(object):
    """
    Perform learning procedure on stream.

    lorem ipsum of description

    Parameters
    ----------
    stream : object
        data stream as an object
    clf : sklearn estimator implementing a ``partial_fit()`` method
    chunk_size : int, optional (default=200)
        Number of samples included in each chunk.
    evaluate_interval : int, optional (default=1000)
        Interval of processed samples before every evaluation.
    controller : processing controller delegate object (default=
    ``controllers.Bare``)

    Examples
    --------
    >>> from strlearn import Learner, controllers
    >>> from sklearn import naive_bayes
    >>> base_classifier = naive_bayes.GaussianNB()
    >>> stream = open('datasets/toyset.arff', 'r')
    >>> controller = controllers.Bare()
    >>> learner = Learner(stream, base_classifier, controller = controller)
    >>> learner.run()

    """

    def __init__(
        self,
        stream,
        base_classifier=neural_network.MLPClassifier(),
        metric=accuracy_score,
    ):
        """Initializer."""
        self.base_classifier = base_classifier
        self.metric = metric

        # Loading dataset
        self.stream = stream
        self.chunk_size = stream.chunk_size

        # Prepare to classification
        self._reset()

    def _reset(self):
        self.clf = base.clone(self.base_classifier)
        self.evaluations = 0
        self.processed_chunks = 0
        self.processed_instances = 0

        self.scores = []
        self.score_points = []
        self.training_times = []
        self.evaluation_times = []

        self.previous_chunk = None
        self.chunk = None

    def run(self):
        """Start learning process."""
        self.training_time = time.time()
        while True:
            self._process_chunk()
            if self.stream.is_dry:
                self.stream.close()
                break

    def _process_chunk(self):
        # Copy the old chunk used in the previous repetition and take a new one
        # from the stream.
        self.previous_chunk = self.chunk
        self.chunk = self.stream.get_chunk()
        X, y = self.chunk

        # Test
        if self.processed_chunks > 0:
            self.test(X, y)

        # Train
        self.train(X, y)

        self.processed_chunks += 1

    def train(self, X, y):
        """Train model."""
        # TODO: Nowe podejście do kontrolerów
        # Iterate samples from chunk.
        self.processed_instances += X.shape[0]

        # Train.
        self.clf.partial_fit(X, y, self.stream.classes)

    def test(self, X, y):
        """Evaluate and return score."""
        self.training_time = time.time() - self.training_time
        evaluation_time = time.time()

        # Prepare evaluation chunk
        y_pred = self.clf.predict(X)
        score = self.metric(y, y_pred)
        # score = self.clf.score(X, y)
        # print(score, self.metric)
        evaluation_time = time.time() - evaluation_time

        # Collecting results
        self.score_points.append(self.processed_instances)
        self.scores.append(score)
        self.evaluation_times.append(evaluation_time)
        self.training_times.append(self.training_time)

        self.training_time = time.time()

        try:
            metrics = self.clf.metrics()
            if not hasattr(self, "metrics"):
                self.metrics = np.zeros((self.stream.n_chunks - 1, metrics.shape[0]))
            self.metrics[self.evaluations] = metrics
        except:
            pass
        self.evaluations += 1

        return score

    def serialize(self, filename):
        """
        Save obtained metrics in CSV file.

        Parameters
        ----------
        filename : name of resulting CSV file

        """
        with open(filename, "w") as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=",")
            for idx, point in enumerate(self.score_points):
                spamwriter.writerow(
                    [
                        "%i" % self.score_points[idx],
                        "%.3f" % self.scores[idx],
                        "%.0f" % (self.evaluation_times[idx] * 1000.0),
                        "%.0f" % (self.training_times[idx] * 1000.0),
                    ]
                )
