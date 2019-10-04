"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")
from sklearn.neural_network import MLPClassifier


def test_mlp_drifted():
    stream = sl.streams.StreamGenerator()
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)
