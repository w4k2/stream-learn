"""Basic tests."""
import sys

import strlearn as sl

sys.path.insert(0, "../..")


def test_arff_dividable():
    """Testing ARFF parser on dividable stream."""
    parser = sl.utils.ARFF("toyset.arff", 1000)
    print("RELATION: %s" % parser.relation)
    print("NUMERIC: %s" % parser.numeric_atts)
    print("NOMINAL: %s" % parser.nominal_atts)

    while not parser.is_dry:
        X, y = parser.get_chunk()
        print(X.shape)
        # print("Chunk %i of size %s" % (i, X.shape))
    parser.close()


def test_arff_non_dividable():
    """Testing ARFF parser on non-dividable stream."""
    parser = sl.utils.ARFF("toyset.arff", 1001)
    print("RELATION: %s" % parser.relation)
    print("NUMERIC: %s" % parser.numeric_atts)
    print("NOMINAL: %s" % parser.nominal_atts)

    while not parser.is_dry:
        X, y = parser.get_chunk()
        print(X.shape)
        # print("Chunk %i of size %s" % (i, X.shape))
    parser.close()


def test_generator_sudden():
    stream = sl.utils.StreamGenerator(drift_type="sudden")
    learner = sl.learners.TestAndTrain(stream)
    learner.run()


def test_generator_gradual():
    stream = sl.utils.StreamGenerator(drift_type="gradual")
    learner = sl.learners.TestAndTrain(stream)
    learner.run()


def test_arff_learner():
    """Test processing."""
    ds = sl.utils.ARFF("toyset.arff")
    learner = sl.learners.TestAndTrain(ds)
    learner.run()
