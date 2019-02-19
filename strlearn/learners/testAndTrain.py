"""Learnerr module."""
from sklearn import base
import time
import csv
from sklearn import neural_network
from strlearn import controllers


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
        controller=controllers.Bare(),
    ):
        """Initializer."""
        self.base_classifier = base_classifier
        self.controller = controller
        self.controller.learner = self

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
        self.controller_measures = []

        self.previous_chunk = None
        self.chunk = None

        self.controller.prepare()

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

        # Inform the processing controller about the analysis of the next
        # chunk.
        self.controller.next_chunk()

        # Train
        self.train(X, y)

        self.processed_chunks += 1

    def train(self, X, y):
        """Train model."""
        # Initialize a training set.
        X_, y_ = [], []
        # Iterate samples from chunk.
        for sid, x in enumerate(X):
            # Check if interruption case occured according to controller.
            if not self.controller.should_break_chunk(X_):
                # Get single object wit a label.
                label = y[sid]

                # Check if, according to controller, it is needed to include
                # current sample in training set.
                if self.controller.should_include(X_, x, label):
                    X_.append(x)
                    y_.append(label)

            # Verify if evaluation is needed.
            self.processed_instances += 1

        # Train.
        self.clf.partial_fit(X_, y_, self.stream.classes)

    def test(self, X, y):
        """Evaluate and return score."""
        self.training_time = time.time() - self.training_time
        evaluation_time = time.time()

        # Prepare evaluation chunk
        score = self.clf.score(X, y)
        evaluation_time = time.time() - evaluation_time

        controller_measure = self.controller.get_measures()

        # Collecting results
        self.score_points.append(self.processed_instances)
        self.scores.append(score)
        self.evaluation_times.append(evaluation_time)
        self.training_times.append(self.training_time)
        self.controller_measures.append(controller_measure)

        self.evaluations += 1

        self.training_time = time.time()

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
                        self.controller_measures[idx],
                    ]
                )
