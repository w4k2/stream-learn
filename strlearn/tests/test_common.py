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
    parser = sl.utils.ARFF()
