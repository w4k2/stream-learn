"""Basic tests."""
import sys

import strlearn as sl
import numpy as np

sys.path.insert(0, "../..")
from sklearn.neural_network import MLPClassifier


def test_generator_same():
    n_chunks = 10
    stream_one = sl.streams.StreamGenerator(random_state=5, n_chunks=n_chunks)
    stream_two = sl.streams.StreamGenerator(random_state=5, n_chunks=n_chunks)
    for i in range(n_chunks):
        X_one, y_one = stream_one.get_chunk()
        X_two, y_two = stream_two.get_chunk()

        assert np.array_equal(X_one, X_two)
        assert np.array_equal(y_one, y_two)


def test_generators_drying():
    stream = sl.streams.StreamGenerator()
    while stream.get_chunk():
        pass


def test_generator_drifted():
    stream = sl.streams.StreamGenerator(n_drifts=1)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_generator_stationary():
    stream = sl.streams.StreamGenerator(n_drifts=0)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


# def test_arff_parser():
#     stream = sl.streams.ARFFParser("strlearn/streams/Agrawal")
#     clf = MLPClassifier()
#     evaluator = sl.evaluators.TestThenTrainEvaluator()
#     evaluator.process(clf, stream)
