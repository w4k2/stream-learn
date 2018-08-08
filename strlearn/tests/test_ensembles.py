"""Testing variations of ensemble methods."""
import sys
import strlearn as sl

sys.path.insert(0, '../..')


def test_WAE():
    """Bare WAE."""
    stream = sl.utils.ARFF('toyset.arff')
    clf = sl.ensembles.WAE()
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


def test_pp_WAE():
    """Post pruned WAE."""
    stream = sl.utils.ARFF('toyset.arff')
    clf = sl.ensembles.WAE(post_pruning=True)
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


def test_WAE_wcm():
    """Various weight computation methods of WAE."""
    methods = ('same_for_each', 'proportional_to_accuracy', 'kuncheva',
               'pta_related_to_whole', 'bell_curve')
    for method in methods:
        stream = sl.utils.ARFF('toyset.arff')
        clf = sl.ensembles.WAE(weight_calculation_method=method)
        learner = sl.learners.TestAndTrain(stream, clf)
        learner.run()


def test_WAE_am():
    """Various aging methods of WAE."""
    methods = ('weights_proportional', 'constant', 'gaussian')
    for method in methods:
        stream = sl.utils.ARFF('toyset.arff')
        clf = sl.ensembles.WAE(aging_method=method)
        learner = sl.learners.TestAndTrain(stream, clf)
        learner.run()


def test_WAE_rejuvenation():
    """Rejuvenation of WAE."""
    stream = sl.utils.ARFF('toyset.arff')
    clf = sl.ensembles.WAE(rejuvenation_power=.5)
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


def test_pp_WAE_rejuvenation():
    """Post pruning with rejuvenation WAE."""
    stream = sl.utils.ARFF('toyset.arff')
    clf = sl.ensembles.WAE(rejuvenation_power=.5, post_pruning=True)
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


"""
def test_REA():
    # Testing REA.
    stream = sl.utils.ARFF('toyset.arff')
    clf = sl.ensembles.REA()
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()
"""
