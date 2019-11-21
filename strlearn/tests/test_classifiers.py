"""Classifier tests."""

import sys
import strlearn as sl

sys.path.insert(0, "../..")


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)


def test_ACS_TestThanTrain():
    "Bare ACS for TTT"
    stream = get_stream()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)


def test_ACS_Prequential():
    "Bare ACS for Prequential"
    stream = get_stream()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.PrequentialEvaluator()
    evaluator.process(stream, clf)
