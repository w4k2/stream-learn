"""Testing variations of ensemble methods."""
import sys
from .context import strlearn as sl


def test_WAE():
    """Bare WAE."""
    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    clf = sl.ensembles.WAE()
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


def test_pp_WAE():
    """Post pruned WAE."""
    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    clf = sl.ensembles.WAE(post_pruning=True)
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


def test_WAE_wcm():
    """Various weight computation methods of WAE."""
    methods = (
        "same_for_each",
        "proportional_to_accuracy",
        "kuncheva",
        "pta_related_to_whole",
        "bell_curve",
    )
    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    for method in methods:
        clf = sl.ensembles.WAE(weight_calculation_method=method)
        learner = sl.learners.TestAndTrain(stream, clf)
        learner.run()
        stream.reset()


def test_WAE_am():
    """Various aging methods of WAE."""
    methods = ("weights_proportional", "constant", "gaussian")
    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    for method in methods:
        clf = sl.ensembles.WAE(aging_method=method)
        learner = sl.learners.TestAndTrain(stream, clf)
        learner.run()
        stream.reset()


def test_WAE_rejuvenation():
    """Rejuvenation of WAE."""
    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    clf = sl.ensembles.WAE(rejuvenation_power=0.5)
    learner = sl.learners.TestAndTrain(stream, clf)
    learner.run()


def test_pp_WAE_rejuvenation():
    """Post pruning with rejuvenation WAE."""
    stream = sl.utils.StreamGenerator(
        drift_type="sudden", n_chunks=10, n_drifts=1, n_features=4, chunk_size=100
    )
    clf = sl.ensembles.WAE(rejuvenation_power=0.5, post_pruning=True)
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
