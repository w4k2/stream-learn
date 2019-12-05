"""Basic tests."""
import sys
import pytest
import strlearn as sl
import numpy as np

sys.path.insert(0, "../..")
from sklearn.neural_network import MLPClassifier


def test_generator_same():
    n_chunks = 10
    stream_one = sl.streams.StreamGenerator(random_state=5, n_chunks=n_chunks)
    stream_two = sl.streams.StreamGenerator(random_state=5, n_chunks=n_chunks)
    for i in range(n_chunks):
        X_one, y_one = stream_one.get_chunk()
        X_two, y_two = stream_two.get_chunk()

        assert np.array_equal(X_one, X_two)
        assert np.array_equal(y_one, y_two)


def test_generator_incremental():
    stream = sl.streams.StreamGenerator(n_drifts=1, incremental=True)
    while stream.get_chunk():
        pass


def test_generator_incremental_reocurring():
    stream = sl.streams.StreamGenerator(n_drifts=2, incremental=True, reocurring=True)
    while stream.get_chunk():
        pass


def test_generator_gradual_nonreocurring():
    stream = sl.streams.StreamGenerator(n_drifts=2, reocurring=False)
    while stream.get_chunk():
        pass


def test_generator_static_balance():
    stream = sl.streams.StreamGenerator(weights=[0.1, 0.9])
    while stream.get_chunk():
        pass


def test_generator_dynamic_balance():
    stream = sl.streams.StreamGenerator(weights=(2, 5, 0.9))
    while stream.get_chunk():
        pass


def test_generator_nonuniform_flip():
    stream = sl.streams.StreamGenerator(y_flip=(0.1, 0.9))
    while stream.get_chunk():
        pass


def test_wrong_flip_tuple():
    with pytest.raises(Exception):
        stream = sl.streams.StreamGenerator(y_flip=(0.1, 0.9, 0.5))
        stream.get_chunk()


def test_wrong_flip_type():
    with pytest.raises(Exception):
        stream = sl.streams.StreamGenerator(y_flip="life is strange")
        stream.get_chunk()


def test_generators_drying():
    stream = sl.streams.StreamGenerator()
    while stream.get_chunk():
        pass


def test_generator_drifted():
    stream = sl.streams.StreamGenerator(n_drifts=1)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_generator_stationary():
    stream = sl.streams.StreamGenerator(n_drifts=0)
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_generator_str():
    stream = sl.streams.StreamGenerator()
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    print(evaluator.scores)
    assert str(stream) == "gr_css999_rs1410_nd0_ln1_d0_50000"

    stream = sl.streams.StreamGenerator(y_flip=(0.5, 0.5))
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    print(evaluator.scores)
    assert str(stream) == "gr_css999_rs1410_nd0_ln50_50_d50_50000"


"""
def test_arff_parser():
    stream = sl.streams.ARFFParser("strlearn/streams/Agrawal")
    clf = MLPClassifier()
    evaluator = sl.evaluators.TestThenTrainEvaluator()
    evaluator.process(clf, stream)
"""
