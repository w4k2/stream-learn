"""Classifier tests."""

import sys
import pytest

from sklearn.metrics import accuracy_score, roc_auc_score

import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from strlearn.evaluators import ContinousRebuild, TriggeredRebuildSupervised, TriggeredRebuildUnsupervised
from ..metrics import balanced_accuracy_score, f1_score, geometric_mean_score_1
from .test_utils import StreamSubset
from ..detectors import DDM, CentroidDistanceDriftDetector


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)


def test_TTT_single_clf():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = sl.evaluators.TestThenTrain(verbose=True)
    evaluator.process(stream, clf)

    assert evaluator.scores.shape == (1, stream.n_chunks - 1, 2)

    cscores = sl.utils.scores_to_cummean(evaluator.scores)


def test_STDT_single_clf():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = sl.evaluators.SparseTrainDenseTest(verbose=True)
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


def test_ContinousRebuild():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = sl.evaluators.ContinousRebuild(verbose=True)
    evaluator.process(stream, clf)

    assert evaluator.scores.shape == (stream.n_chunks - 1, 1)
    print(evaluator.scores)


def test_ContinousRebuild_poker_benchmark():
    benchmark = sl.streams.Poker(n_chunks=500)
    stream = StreamSubset(benchmark, yield_n_chunks=15)
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = ContinousRebuild(verbose=True)
    evaluator.process(stream, clf)

    assert evaluator.scores.shape == (15 - 1, 1)
    print(evaluator.scores)


@pytest.mark.parametrize("evaluator_class", [TriggeredRebuildSupervised])
def test_labeling_delay(evaluator_class):
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = evaluator_class(verbose=True)
    detector = DDM()
    evaluator.process(stream, clf, detector)

    assert evaluator.scores.shape == (stream.n_chunks - 1, 1)
    print(evaluator.scores)


def test_labeling_delay_unsupervised():
    stream = get_stream()
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = TriggeredRebuildUnsupervised(verbose=True)
    detector = CentroidDistanceDriftDetector()
    evaluator.process(stream, clf, detector)

    assert evaluator.scores.shape == (stream.n_chunks - 1, 1)
    print(evaluator.scores)


@pytest.mark.parametrize("evaluator_class", [TriggeredRebuildSupervised])
def test_labeling_delay_poker_benchmark(evaluator_class):
    benchmark = sl.streams.Poker(n_chunks=500)
    stream = StreamSubset(benchmark, yield_n_chunks=15)
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = evaluator_class(verbose=True)
    detector = DDM()
    evaluator.process(stream, clf, detector)

    assert evaluator.scores.shape == (15 - 1, 1)
    print(evaluator.scores)


def test_labeling_delay_unsupervised_poker_benchmark():
    benchmark = sl.streams.Poker(n_chunks=500)
    stream = StreamSubset(benchmark, yield_n_chunks=15)
    clf = sl.classifiers.ASC(base_clf=GaussianNB())
    evaluator = TriggeredRebuildUnsupervised(verbose=True)
    detector = CentroidDistanceDriftDetector()
    evaluator.process(stream, clf, detector)

    assert evaluator.scores.shape == (15 - 1, 1)
    print(evaluator.scores)

# TODO: labeling delay 0
# TODO: partial True and False
