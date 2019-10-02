"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")
from sklearn.neural_network import MLPClassifier


def test_mlp_drifted():
    stream = sl.generators.DriftedStream(sigmoid_spacing=999)
    clf = MLPClassifier(hidden_layer_sizes=(100,))
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)
