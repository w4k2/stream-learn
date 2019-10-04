"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")


def test_ACS_TestThanTrain():
    stream = sl.generators.DriftedStream(sigmoid_spacing=999)
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)

def test_ACS_Prequential():
    stream = sl.generators.DriftedStream(sigmoid_spacing=999)
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.PrequentialEvaluator()
    evaluator.process(clf, stream)