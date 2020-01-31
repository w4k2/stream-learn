"""Basic tests."""

import sys

from sklearn.neural_network import MLPClassifier

import strlearn as sl


def test_mlp_drifted():
    stream = sl.streams.StreamGenerator(n_drifts=1)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
