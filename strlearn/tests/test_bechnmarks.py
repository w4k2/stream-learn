import pytest
import strlearn as sl

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from strlearn.streams import Covtype, Electricity, Insects, Poker  # LED, SEA, SineGenerator, WaveformGenerator


@pytest.mark.parametrize("benchmark_class", [Covtype, Electricity, Poker])
def test_can_iterate(benchmark_class):
    stream = benchmark_class()
    for _ in stream:
        pass


@pytest.mark.parametrize("benchmark_class", [Covtype, Electricity, Poker])
def test_can_train(benchmark_class):
    stream = benchmark_class()
    clf = GaussianNB()
    ttt = sl.evaluators.TestThenTrain([accuracy_score])
    ttt.process(stream, clf)
    acc = ttt.scores
    assert acc[0, 0, -1] > 0.2


@pytest.mark.parametrize("drift_mode,subsample", [("abrupt", True), ("abrupt", False), ("gradual", True), ("gradual", False), ("incremental", True), ("incremental", False)])
def test_insects_can_iterate(drift_mode, subsample):
    stream = Insects(drift_mode, subsample)
    for _ in stream:
        pass


@pytest.mark.parametrize("drift_mode,subsample", [("abrupt", True), ("abrupt", False), ("gradual", True), ("gradual", False), ("incremental", True), ("incremental", False)])
def test_insects_can_train(drift_mode, subsample):
    stream = Insects(drift_mode, subsample)
    clf = GaussianNB()
    ttt = sl.evaluators.TestThenTrain([accuracy_score])
    ttt.process(stream, clf)
    acc = ttt.scores
    assert acc[0, 0, -1] > 0.2
