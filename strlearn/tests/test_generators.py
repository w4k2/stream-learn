"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")
from sklearn.neural_network import MLPClassifier


def test_generators_drying():
    stream = sl.generators.DriftedStream()
    while stream.get_chunk():
        pass
    stream = sl.generators.StationaryStream()
    while stream.get_chunk():
        pass


def test_generator_drifted():
    stream = sl.generators.DriftedStream()
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)


def test_generator_stationary():
    stream = sl.generators.StationaryStream()
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)
