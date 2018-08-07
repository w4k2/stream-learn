import sys
import strlearn as sl

sys.path.insert(0, '../..')


def test_basic():
    X, y = sl.utils.load_arff('../toyset.arff')
    learner = sl.learners.TestAndTrain(X, y)
    learner.run()
