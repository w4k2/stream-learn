"""Basic tests."""
import sys
import numpy as np
import requests
import pytest
import strlearn as sl
import os
from sklearn.naive_bayes import GaussianNB

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


def test_generator_incremental_recurring():
    stream = sl.streams.StreamGenerator(n_drifts=2, incremental=True, recurring=True)
    while stream.get_chunk():
        pass


def test_generator_gradual_nonrecurring():
    stream = sl.streams.StreamGenerator(n_drifts=2, recurring=False)
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
    clf = GaussianNB()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_generator_stationary():
    stream = sl.streams.StreamGenerator(n_drifts=0)
    clf = GaussianNB()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_generator_str():
    stream = sl.streams.StreamGenerator()
    clf = GaussianNB()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    print(evaluator.scores)
    assert str(stream) == "gr_n_css999_rs1410_nd0_ln1_d50_50000"

    stream = sl.streams.StreamGenerator(y_flip=(0.5, 0.5))
    clf = GaussianNB()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    print(evaluator.scores)
    assert str(stream) == "gr_n_css999_rs1410_nd0_ln50_50_d50_50000"

@pytest.fixture(scope="session", autouse=True)
def stream_filepath():
    filepath = "test_stream.arff"
    yield filepath

@pytest.fixture(scope="session", autouse=True)
def stream_filepath_csv():
    filepath = "test_stream.csv"
    yield filepath

@pytest.fixture(scope="session", autouse=True)
def stream_filepath_npy():
    filepath = "test_stream.npy"
    yield filepath

# os.remove(filepath)

def test_generator_save_to_csv(stream_filepath_csv):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_one.save_to_csv(stream_filepath_csv)

def test_generator_save_to_npy(stream_filepath_npy):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_one.save_to_npy(stream_filepath_npy)

def test_generator_save_to_arff(stream_filepath):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_one.save_to_arff(stream_filepath)

def test_csvparser(stream_filepath_csv):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_two = sl.streams.CSVParser(
        stream_filepath_csv, chunk_size=chunk_size, n_chunks=n_chunks
    )

    for i in range(n_chunks):
        X_one, y_one = stream_one.get_chunk()
        X_two, y_two = stream_two.get_chunk()

        assert np.allclose(X_one, X_two)
        assert np.array_equal(y_one, y_two)


def test_npyparser(stream_filepath_npy):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_two = sl.streams.NPYParser(
        stream_filepath_npy, chunk_size=chunk_size, n_chunks=n_chunks
    )

    for i in range(n_chunks):
        X_one, y_one = stream_one.get_chunk()
        X_two, y_two = stream_two.get_chunk()

        assert np.allclose(X_one, X_two)
        assert np.array_equal(y_one, y_two)


def test_arffparser(stream_filepath):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_two = sl.streams.ARFFParser(
        stream_filepath, chunk_size=chunk_size, n_chunks=n_chunks
    )

    for i in range(n_chunks):
        X_one, y_one = stream_one.get_chunk()
        X_two, y_two = stream_two.get_chunk()

        assert np.allclose(X_one, X_two)
        assert np.array_equal(y_one, y_two)

def test_arffparser_str(stream_filepath):
    stream = sl.streams.ARFFParser(stream_filepath)
    assert str(stream) == stream_filepath


def test_arffparser_is_dry(stream_filepath):
    n_chunks = 10
    chunk_size = 20
    stream = sl.streams.ARFFParser(
        stream_filepath, chunk_size=chunk_size, n_chunks=n_chunks
    )
    assert not stream.is_dry()


def test_arffparser_reset(stream_filepath):
    stream = sl.streams.ARFFParser(stream_filepath)
    stream.reset()
    assert stream.chunk_id == 0
    assert not stream.is_dry()


def test_arff_parser():
    filename = "stream.arff"
    n_chunks = 20
    stream_original = sl.streams.StreamGenerator(
        n_drifts=1, incremental=True, n_chunks=n_chunks
    )
    stream_original.save_to_arff(filename)
    stream_parsed = sl.streams.ARFFParser(filename)

    for i in range(n_chunks):
        X_a, y_a = stream_original.get_chunk()
        X_b, y_b = stream_parsed.get_chunk()

        assert np.array_equal(X_a, X_b)
        assert np.array_equal(y_a, y_b)
