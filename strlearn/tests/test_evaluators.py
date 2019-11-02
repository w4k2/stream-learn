"""Classifier tests."""

import sys
import strlearn as sl
from sklearn.metrics import accuracy_score, roc_auc_score
from ..utils import bac, f_score, geometric_mean_score

sys.path.insert(0, "../..")


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)


def test_TTT_single_clf():
    stream = get_stream()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf)

    assert evaluator.scores_.shape == (1, stream.n_chunks - 1, 2)


def test_TTT_custom_metrics():
    stream = get_stream()
    clf = sl.classifiers.AccumulatedSamplesClassifier()
    metrics = [accuracy_score, roc_auc_score, geometric_mean_score, bac, f_score]
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clf, metrics=metrics)

    assert evaluator.scores_.shape == (1, stream.n_chunks - 1, len(metrics))


def test_TTT_multiple_clfs():
    stream = get_stream()
    clfs = [
        sl.classifiers.AccumulatedSamplesClassifier(),
        sl.classifiers.AccumulatedSamplesClassifier(),
    ]
    metrics = [accuracy_score, roc_auc_score, geometric_mean_score, bac, f_score]
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(stream, clfs, metrics=metrics)

    assert evaluator.scores_.shape == (len(clfs), stream.n_chunks - 1, len(metrics))
