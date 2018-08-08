"""Basic tests."""
import sys
import strlearn as sl

sys.path.insert(0, '../..')


def test_arff_a():
    """Testing ARFF parser on dividable stream."""
    parser = sl.utils.ARFF('toyset.arff')
    print("RELATION: %s" % parser.relation)
    print("NUMERIC: %s" % parser.numeric_atts)
    print("NOMINAL: %s" % parser.nominal_atts)

    while not parser.is_dry:
        X, y = parser.get_chunk(1000)
        print(X.shape)
        # print("Chunk %i of size %s" % (i, X.shape))
    parser.close()


def test_arff_b():
    """Testing ARFF parser on non-dividable stream."""
    parser = sl.utils.ARFF('toyset.arff')
    print("RELATION: %s" % parser.relation)
    print("NUMERIC: %s" % parser.numeric_atts)
    print("NOMINAL: %s" % parser.nominal_atts)

    while not parser.is_dry:
        X, y = parser.get_chunk(1001)
        print(X.shape)
        # print("Chunk %i of size %s" % (i, X.shape))
    parser.close()


def test_tat_learner():
    """Test processing."""
    ds = sl.utils.ARFF('toyset.arff')
    learner = sl.learners.TestAndTrain(ds)
    learner.run()
