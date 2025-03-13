import pytest
import os
import strlearn as sl

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from strlearn.streams import Covtype, Electricity, Insects, Poker  # LED, SEA, SineGenerator, WaveformGenerator

from .test_utils import StreamSubset


@pytest.mark.parametrize("benchmark_class", [Covtype, Electricity, Poker])
def test_can_iterate(benchmark_class):
    benchmark_stream = benchmark_class()
    stream_subset = StreamSubset(benchmark_stream, yield_n_chunks=5)
    for i, _ in enumerate(stream_subset):
        pass


@pytest.mark.parametrize("benchmark_class", [Covtype, Electricity, Poker])
def test_can_train(benchmark_class):
    benchmark_stream = benchmark_class()
    stream_subset = StreamSubset(benchmark_stream, yield_n_chunks=3)
    clf = GaussianNB()
    ttt = sl.evaluators.TestThenTrain([accuracy_score])
    ttt.process(stream_subset, clf)
    acc = ttt.scores
    assert acc[0, 0, -1] > 0.2


@pytest.mark.parametrize("benchmark_class", [Covtype, Electricity, Poker])
def test_benchmark_str(benchmark_class):
    stream = benchmark_class(chunk_size=200, n_chunks=100)
    assert str(stream) == f'{benchmark_class.__name__}(chunk_size=200, n_chunks=100)'


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


@pytest.mark.parametrize("drift_mode,subsample", [("abrupt", True), ("abrupt", False), ("gradual", True), ("gradual", False), ("incremental", True), ("incremental", False)])
def test_insects_str(drift_mode, subsample):
    stream = Insects(drift_mode, subsample, chunk_size=200, n_chunks=250)
    assert str(stream) == f'Insects(drift_mode={drift_mode}, subsample={subsample}, chunk_size=200, n_chunks=250)'


def test_insects_incorrect_drift_mode():
    with pytest.raises(ValueError):
        Insects("incorrect_drift_mode")


def test_downloading_files():
    data_path = sl.streams.utils.get_data_path()
    os.remove(data_path / "electricity.csv")
    stream = Electricity()
    for _ in stream:
        pass
