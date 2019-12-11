"""Classifier tests."""

import sys
import strlearn as sl
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, "../..")


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)

def test_ACS_Prequential():
    "Bare ACS for Prequential"
    stream = get_stream()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.Prequential()
    evaluator.process(stream, clf)

def test_MetaEstimator_TestThanTrain():
    "Bare ACS for TTT"
    stream = get_stream()
    base = sl.classifiers.SampleWeightedMetaEstimator(base_classifier=GaussianNB())
    clf = sl.ensembles.OOB(base_estimator=base)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)

def test_MetaEstimator_fit():
    "Bare ACS for TTT"
    stream = get_stream()
    X, y = stream.get_chunk()
    clf = sl.classifiers.SampleWeightedMetaEstimator(base_classifier=GaussianNB())
    clf.fit(X, y)
    clf.predict(X)
