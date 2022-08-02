"""Basic tests."""
import chunk
import sys
import numpy as np
import requests
import pytest
from sklearn.metrics import accuracy_score
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
    stream = sl.streams.StreamGenerator(random_state=1410)
    clf = GaussianNB()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    print(evaluator.scores)
    assert str(stream) == "gr_n_css999_rs1410_nd0_ln1_d50_50000"

    stream = sl.streams.StreamGenerator(y_flip=(0.5, 0.5), random_state=1410)
    clf = GaussianNB()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
    print(evaluator.scores)
    assert str(stream) == "gr_n_css999_rs1410_nd0_ln50_50_d50_50000"


@pytest.fixture(scope="session", autouse=True)
def stream_filepath_arff():
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

    stream_read = sl.streams.CSVParser(stream_filepath_csv, chunk_size=chunk_size, n_chunks=n_chunks)
    for (X_original, y_original), (X_read, y_read) in zip(stream_one, stream_read):
        assert np.isclose(X_original, X_read).all()
        assert np.isclose(y_original, y_read).all()


def test_generator_save_to_npy(stream_filepath_npy):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_one.save_to_npy(stream_filepath_npy)


def test_generator_save_to_arff(stream_filepath_arff):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_one.save_to_arff(stream_filepath_arff)


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


def test_can_train_with_csvparser(stream_filepath_csv):
    n_chunks = 10
    chunk_size = 20
    stream = sl.streams.CSVParser(
        stream_filepath_csv, chunk_size=chunk_size, n_chunks=n_chunks
    )

    clf = GaussianNB()
    ttt = sl.evaluators.TestThenTrain([accuracy_score])
    ttt.process(stream, clf)
    acc = ttt.scores
    assert acc[0, 0, -1] > 0.5


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


def test_arffparser(stream_filepath_arff):
    n_chunks = 10
    chunk_size = 20
    stream_one = sl.streams.StreamGenerator(
        random_state=5, chunk_size=chunk_size, n_chunks=n_chunks
    )
    stream_two = sl.streams.ARFFParser(
        stream_filepath_arff, chunk_size=chunk_size, n_chunks=n_chunks
    )

    for i in range(n_chunks):
        X_one, y_one = stream_one.get_chunk()
        X_two, y_two = stream_two.get_chunk()

        assert np.allclose(X_one, X_two)
        assert np.array_equal(y_one, y_two)


def test_arffparser_str(stream_filepath_arff):
    stream = sl.streams.ARFFParser(stream_filepath_arff)
    assert str(stream) == stream_filepath_arff


def test_arffparser_is_dry(stream_filepath_arff):
    n_chunks = 10
    chunk_size = 20
    stream = sl.streams.ARFFParser(
        stream_filepath_arff, chunk_size=chunk_size, n_chunks=n_chunks
    )
    assert not stream.is_dry()


def test_arffparser_reset(stream_filepath_arff):
    stream = sl.streams.ARFFParser(stream_filepath_arff)
    stream.reset()
    assert stream.chunk_id == 0
    assert not stream.is_dry()


def test_arff_parser_different_stream():
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


def test_can_train_with_arff_stream():
    filename = "stream.arff"
    stream = sl.streams.ARFFParser(filename, chunk_size=100, n_chunks=40)

    clf = GaussianNB()
    ttt = sl.evaluators.TestThenTrain([accuracy_score])
    ttt.process(stream, clf)
    acc = ttt.scores
    assert acc[0, 0, -1] > 0.5


@pytest.fixture(scope="session", autouse=True)
def arff_content():
    stream_real_file_content = \
        """@relation test
@attribute feature_1 real
@attribute feature_2 real
@attribute feature_3 real
@attribute feature_4 real
@attribute class {True, False}
@data
0.324, -0.7654, 1.3456, 0.98765, True
0.324, -0.456, 1.3456, 0.345, False
0.324, -0.8765, 1.3456, 0.765, True
0.234, -0.2, 1.3456, 0.34, False
0.098, -0.98, 1.345, 0.98765, False
0.324, -0.7654, 1.67, 0.23, True"""
    yield stream_real_file_content


def test_arff_parser_can_parse_real_feature(arff_content):
    filename = "stream_real_feature.arff"
    with open(filename, 'w+') as f:
        f.write(arff_content)
    stream = sl.streams.ARFFParser(filename)

    clf = GaussianNB()
    ttt = sl.evaluators.TestThenTrain([accuracy_score])
    ttt.process(stream, clf)
    acc = ttt.scores
    assert acc[0, 0, -1] > 0.5


def test_can_iterate():
    stream = sl.streams.StreamGenerator(chunk_size=100, n_features=50)
    for X, y in stream:
        assert X.shape[0] == 100
        assert X.shape[1] == 50
        assert y.shape[0] == 100


def test_can_iterate_arf(stream_filepath_arff):
    stream = sl.streams.ARFFParser(stream_filepath_arff, chunk_size=20, n_chunks=10)
    for X, y in stream:
        assert X.shape[0] == 20
        assert X.shape[1] == 20
        assert y.shape[0] == 20


def test_can_iterate_csv(stream_filepath_csv):
    n_chunks = 10
    chunk_size = 20
    stream = sl.streams.CSVParser(
        stream_filepath_csv, chunk_size=chunk_size, n_chunks=n_chunks
    )

    for X, y in stream:
        assert X.shape[0] == chunk_size
        assert X.shape[1] == 20
        assert y.shape[0] == chunk_size


def test_can_iterate_npy(stream_filepath_npy):
    n_chunks = 10
    chunk_size = 20
    stream = sl.streams.NPYParser(
        stream_filepath_npy, chunk_size=chunk_size, n_chunks=n_chunks
    )

    for X, y in stream:
        assert X.shape[0] == chunk_size
        assert X.shape[1] == 20
        assert y.shape[0] == chunk_size
