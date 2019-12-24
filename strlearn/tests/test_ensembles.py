"""Ensemble tests."""

import sys
import numpy as np
import strlearn as sl
import pytest
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, "../..")


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10, n_features=10)


def get_different_stream():
    return sl.streams.StreamGenerator(n_chunks=10, n_features=4)


def test_ensembles_fit():
    clf1 = sl.ensembles.SEA(GaussianNB())
    clf2 = sl.ensembles.WAE(GaussianNB())
    clf3 = sl.ensembles.OOB(GaussianNB())
    clf4 = sl.ensembles.OnlineBagging(GaussianNB())
    clf5 = sl.ensembles.UOB(GaussianNB())

    clfs = (clf1, clf2, clf3, clf4, clf5)

    stream = get_stream()
    X, y = stream.get_chunk()

    for clf in clfs:
        clf.fit(X, y)

    clf.fit(X, y)


def test_features():
    "Bare CBE"
    clfs = [
        sl.ensembles.SEA(GaussianNB()),
        sl.ensembles.OOB(GaussianNB()),
        sl.ensembles.UOB(GaussianNB()),
        sl.ensembles.WAE(GaussianNB()),
        sl.ensembles.OnlineBagging(GaussianNB()),
    ]
    stream = get_stream()
    different_stream = get_different_stream()

    for clf in clfs:
        X_a, y_a = stream.get_chunk()
        X_b, y_b = different_stream.get_chunk()
        clf.partial_fit(X_a, y_a, stream.classes_)

        with pytest.raises(ValueError):
            clf.partial_fit(X_b, y_b)


def test_pred():
    """Pred error"""
    clfs = [
        sl.ensembles.SEA(GaussianNB()),
        sl.ensembles.OOB(GaussianNB()),
        sl.ensembles.UOB(GaussianNB()),
        sl.ensembles.WAE(GaussianNB()),
        sl.ensembles.OnlineBagging(GaussianNB()),
    ]
    stream = get_stream()
    different_stream = get_different_stream()

    for clf in clfs:
        X_a, y_a = stream.get_chunk()
        X_b, y_b = different_stream.get_chunk()
        clf.partial_fit(X_a, y_a, stream.classes_)

        with pytest.raises(ValueError):
            clf.predict(X_b)


def test_SEA():
    "Bare SEA"
    stream = get_stream()
    clf = sl.ensembles.SEA(GaussianNB(), n_estimators=5)
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
    clf = sl.ensembles.WAE(
        GaussianNB(), weight_calculation_method="same_for_each", n_estimators=5
    )
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
