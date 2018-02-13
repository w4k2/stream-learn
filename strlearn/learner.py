"""Learner module"""
import arff
from sklearn import preprocessing, base
import numpy as np
import time
import csv
from tqdm import tqdm

import controllers

class Learner(object):
    """Perform learning procedure on stream.

    lorem ipsum of description

    Parameters
    ----------
    stream : data stream as a binary arff file, loaded like ``toystream = open('datasets/toyset.arff', 'r')``
    clf : sklearn estimator implementing a ``partial_fit()`` method
    chunk_size : int, optional (default=200)
        Number of samples included in each chunk.
    evaluate_interval : int, optional (default=1000)
        Interval of processed samples before every evaluation.
    controller : processing controller delegate object (default= ``controllers.Bare``)

    Examples
    --------
    >>> from strlearn import Learner, controllers
    >>> from sklearn import naive_bayes
    >>> base_classifier = naive_bayes.GaussianNB()
    >>> stream = open('datasets/toyset.arff', 'r')
    >>> controller = controllers.Bare()
    >>> learner = Learner(stream = stream, base_classifier = base_classifier, controller = controller)
    >>> learner.run()
    """

    def __init__(self, stream, base_classifier, chunk_size=200, evaluate_interval=1000, controller=controllers.Bare()):
        self.base_classifier = base_classifier
        self.chunk_size = chunk_size
        self.evaluate_interval = evaluate_interval
        self.controller = controller
        self.controller.learner = self

        # Loading dataset
        dataset = arff.load(stream)
        data = np.array(dataset['data'])
        self.classes = dataset['attributes'][-1][-1]
        self.X = data[:,:-1].astype(np.float)
        self.y = data[:,-1]

        # Data analysis
        self.number_of_samples = len(self.y)
        self.number_of_classes = len(self.classes)

        # Prepare to classification
        self._reset()

    def __str__(self):
        return "stream_learner_c_%i_e_%i_clf_%s_ctrl_%s" % (
            self.chunk_size,
            self.evaluate_interval,
            self.base_classifier,
            self.controller
        )

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
        '''
        Start learning process.
        '''
        self.training_time = time.time()
        for i in tqdm(xrange(self.number_of_samples / self.chunk_size), desc='CHN'):
            self._process_chunk()

    def _process_chunk(self):
        # Copy the old chunk used in the previous repetition and take a new one from the stream.
        self.previous_chunk = self.chunk
        startpoint = self.processed_chunks * self.chunk_size
        self.chunk = (self.X[startpoint:startpoint + self.chunk_size], self.y[startpoint:startpoint + self.chunk_size])

        # Inform the processing controller about the analysis of the next chunk.
        self.controller.next_chunk()

        # Initialize a training set.
        X, y = [], []

        # Iterate samples from chunk.
        for sid, x in enumerate(self.chunk[0]):
            # Check if interruption case occured according to controller.
            if not self.controller.should_break_chunk(X):
                # Get single object wit a label.
                label = self.chunk[1][sid]

                # Check if, according to controller, it is needed to include current sample in training set.
                if self.controller.should_include(X, x, label):
                    X.append(x)
                    y.append(label)

            # Verify if evaluation is needed.
            self.processed_instances += 1
            if self.processed_instances % self.evaluate_interval == 0:
                self._evaluate()

        X = np.array(X)
        y = np.array(y)

        # Fit model with current training set.
        self._fit_with_chunk(X, y)
        self.processed_chunks += 1

    def _fit_with_chunk(self, X, y):
        self.clf.partial_fit(X, y, self.classes)

    def _evaluate(self):
        self.training_time = time.time() - self.training_time
        evaluation_time = time.time()

        # Prepare evaluation chunk
        startpoint = (self.evaluations - 1) * self.evaluate_interval

        if startpoint > 0:
            evaluation_chunk = (self.X[startpoint:startpoint + self.evaluate_interval], self.y[startpoint:startpoint + self.evaluate_interval])

            # Create empty training set
            X, y = evaluation_chunk

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

    def serialize(self, filename):
        """
        Save obtained metrics in CSV file.

        Parameters
        ----------
        filename : name of resulting CSV file
        """
        with open(filename, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for idx, point in enumerate(self.score_points):
                spamwriter.writerow([
                    '%i' % self.score_points[idx],
                    '%.3f' % self.scores[idx],
                    '%.0f' % (self.evaluation_times[idx] * 1000.),
                    '%.0f' % (self.training_times[idx] * 1000.),
                    self.controller_measures[idx]
                ])
