"""Basic tests."""
import sys
import strlearn as sl

sys.path.insert(0, '../..')


def test_basic():
    """Test primitive processing."""
    X, y = sl.utils.load_arff('toyset.arff')
    learner = sl.learners.TestAndTrain(X, y)
    learner.run()


def test_arff():
    """Testing ARFF parser."""
    parser = sl.utils.ARFF('toyset.arff')
    print("RELATION: %s" % parser.relation)
    print("NUMERIC: %s" % parser.numeric_atts)
    print("NOMINAL: %s" % parser.nominal_atts)

    while True:
        X, y = parser.get_chunk(500)
        # print("Chunk %i of size %s" % (i, X.shape))
        if parser.is_dry:
            break
    parser.close()
