"""Ensemble tests."""

import sys
import strlearn as sl
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, "../..")


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10)


def test_CBE():
    "Bare CBE"
    stream = get_stream()
    clf = sl.ensembles.ChunkBasedEnsemble(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_OOB():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.OOB(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_OB():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.OnlineBagging(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_UOB():
    """Bare WAE."""
    stream = get_stream()
    clf = sl.ensembles.UOB(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_pp_WAE():
    """Post pruned WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), post_pruning=True, n_estimators=5)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm1():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), weight_calculation_method="same_for_each")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm2():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(
        GaussianNB(), weight_calculation_method="proportional_to_accuracy"
    )
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm3():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(
        GaussianNB(), weight_calculation_method="pta_related_to_whole"
    )
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm4():
    """Various weight computation methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), weight_calculation_method="bell_curve")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_am2():
    """Various aging methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), aging_method="constant")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_am3():
    """Various aging methods of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), aging_method="gaussian")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_rejuvenation():
    """Rejuvenation of WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), rejuvenation_power=0.5)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_pp_WAE_rejuvenation():
    """Post pruning with rejuvenation WAE."""
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), rejuvenation_power=0.5, post_pruning=True)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)
