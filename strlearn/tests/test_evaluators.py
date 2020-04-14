"""Classifier tests."""

import sys

from sklearn.metrics import accuracy_score, roc_auc_score

import strlearn as sl
from sklearn.naive_bayes import GaussianNB

from ..metrics import balanced_accuracy_score, f1_score, geometric_mean_score_1


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)


def test_TTT_single_clf():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = sl.evaluators.TestThenTrain(verbose=True)
    evaluator.process(stream, clf)

    assert evaluator.scores.shape == (1, stream.n_chunks - 1, 2)


def test_TTT_custom_metrics():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    metrics = [
        accuracy_score,
        roc_auc_score,
        geometric_mean_score_1,
        balanced_accuracy_score,
        f1_score,
    ]
    evaluator = sl.evaluators.TestThenTrain(metrics=metrics)
    evaluator.process(stream, clf)

    assert evaluator.scores.shape == (1, stream.n_chunks - 1, len(metrics))


def test_TTT_one_metric():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = sl.evaluators.TestThenTrain(metrics=accuracy_score)
    evaluator.process(stream, clf)

    assert evaluator.scores.shape == (1, stream.n_chunks - 1, 1)


def test_TTT_multiple_clfs():
    stream = get_stream()
    clfs = [
        sl.classifiers.ASC(base_clf=GaussianNB()),
        sl.classifiers.ASC(base_clf=GaussianNB()),
    ]
    metrics = [
        accuracy_score,
        roc_auc_score,
        geometric_mean_score_1,
        balanced_accuracy_score,
        f1_score,
    ]
    evaluator = sl.evaluators.TestThenTrain(metrics=metrics)
    evaluator.process(stream, clfs)

    assert evaluator.scores.shape == (len(clfs), stream.n_chunks - 1, len(metrics))


def test_P_multiple_clfs():
    stream = get_stream()
    clfs = [
        sl.classifiers.ASC(base_clf=GaussianNB()),
        sl.classifiers.ASC(base_clf=GaussianNB()),
    ]
    metrics = [
        accuracy_score,
        roc_auc_score,
        geometric_mean_score_1,
        balanced_accuracy_score,
        f1_score,
    ]
    evaluator = sl.evaluators.Prequential(metrics=metrics)
    evaluator.process(stream, clfs)

    assert evaluator.scores.shape == (
        len(clfs),
        (stream.n_chunks - 1) * 2,
        len(metrics),
    )


def test_P_one_metric():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = sl.evaluators.Prequential(metrics=accuracy_score)
    evaluator.process(stream, clf, interval=100)

    assert evaluator.scores.shape == (1, (stream.n_chunks - 1) * 2, 1,)
