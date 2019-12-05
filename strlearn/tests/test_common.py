"""Basic tests."""

import sys
import strlearn as sl
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, "../..")


def test_mlp_drifted():
    stream = sl.streams.StreamGenerator(n_drifts=1)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
