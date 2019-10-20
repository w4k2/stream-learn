"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")
from sklearn.neural_network import MLPClassifier


def test_generators_drying():
    stream = sl.streams.StreamGenerator()
    while stream.get_chunk():
        pass


def test_generator_drifted():
    stream = sl.streams.StreamGenerator(n_drifts=1)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)


def test_generator_stationary():
    stream = sl.streams.StreamGenerator(n_drifts=0)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)

# def test_arff_parser():
#     stream = sl.streams.ARFFParser("strlearn/streams/Agrawal")
#     clf = MLPClassifier()
#     evaluator = sl.evaluators.TestThenTrainEvaluator()
#     evaluator.process(clf, stream)
