"""Classifier tests."""

import sys
import strlearn as sl

sys.path.insert(0, "../..")


def test_ACS_TestThanTrain():
    "Bare ACS for TTT"
    stream = sl.streams.StreamGenerator()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)


def test_ACS_Prequential():
    "Bare ACS for Prequential"
    stream = sl.streams.StreamGenerator()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.PrequentialEvaluator()
    evaluator.process(clf, stream)
