"""Ensemble tests."""

import sys

from sklearn.naive_bayes import GaussianNB

import numpy as np
import pytest
import strlearn as sl


def get_clfs():
    return [
        sl.ensembles.SEA(GaussianNB()),
        sl.ensembles.OOB(GaussianNB()),
        sl.ensembles.UOB(GaussianNB()),
        sl.ensembles.WAE(GaussianNB()),
        sl.ensembles.AWE(GaussianNB()),
        sl.ensembles.AUE(GaussianNB()),
        sl.ensembles.DWM(GaussianNB()),
        sl.ensembles.OnlineBagging(GaussianNB()),
        sl.ensembles.REA(GaussianNB()),
        sl.ensembles.KMC(GaussianNB()),
        sl.ensembles.LearnppCDS(GaussianNB()),
        sl.ensembles.LearnppNIE(GaussianNB()),
        sl.ensembles.OUSE(GaussianNB())
    ]


def get_stream():
    return sl.streams.StreamGenerator(n_chunks=10, n_features=10)


def get_different_stream():
    return sl.streams.StreamGenerator(n_chunks=10, n_features=4)


def get_imbalanced_stream():
    return sl.streams.StreamGenerator(n_chunks=2, n_features=10, weights=[0.99,0.01])


def test_predict_proba():
    clfs = get_clfs()
    stream = get_stream()
    X, y = stream.get_chunk()

    for clf in clfs:
        if hasattr(clf, "predict_proba"):
            clf.fit(X, y)
            pp = clf.predict_proba(X)
            y_pred = clf.predict(X)
            y_pred_pp = np.argmax(pp, axis=1)

            assert np.array_equal(y_pred, y_pred_pp)


def test_ensembles_fit():
    clfs = get_clfs()
    stream = get_stream()
    X, y = stream.get_chunk()

    for clf in clfs:
        clf.fit(X, y)

    clf.fit(X, y)


def test_features():
    "Bare CBE"
    clfs = get_clfs()
    stream = get_stream()
    different_stream = get_different_stream()

    X_a, y_a = stream.get_chunk()
    X_b, y_b = different_stream.get_chunk()

    for clf in clfs:
        clf.partial_fit(X_a, y_a, stream.classes_)

        with pytest.raises(ValueError):
            clf.partial_fit(X_b, y_b)


def test_pred():
    "Pred error"
    clfs = get_clfs()
    stream = get_stream()
    different_stream = get_different_stream()

    X_a, y_a = stream.get_chunk()
    X_b, y_b = different_stream.get_chunk()

    for clf in clfs:
        clf.partial_fit(X_a, y_a, stream.classes_)

        with pytest.raises(ValueError):
            clf.predict(X_b)


def test_minority_majority_name_split():
    clfs = get_clfs()
    stream = get_imbalanced_stream()
    X, y = stream.get_chunk()

    for clf in clfs:
        if hasattr(clf, "minority_majority_name"):
            clf.fit(X, y)


def test_one_class():
    clfs = get_clfs()
    stream = get_imbalanced_stream()
    X, y = stream.get_chunk()

    minority_ma = np.ma.masked_where(y == 0, y)
    X = X[minority_ma.mask]
    y = y[minority_ma.mask]


    for clf in clfs:
        if hasattr(clf, "minority_majority_name"):
            with pytest.raises(ValueError):
                clf.fit(X, y)


def test_DWM():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.DWM(GaussianNB(), p=stream.chunk_size))


def test_AWE():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.AWE(GaussianNB(), n_estimators=5))


def test_AUE():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.AUE(GaussianNB(), n_estimators=5))


def test_SEA():
    "Bare SEA"
    stream = get_stream()
    clf = sl.ensembles.SEA(GaussianNB(), n_estimators=5)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE():
    "Bare WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_OOB():
    "Bare WAE."
    stream = get_stream()
    clf = sl.ensembles.OOB(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_OB():
    "Bare WAE."
    stream = get_stream()
    clf = sl.ensembles.OnlineBagging(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_UOB():
    "Bare WAE."
    stream = get_stream()
    clf = sl.ensembles.UOB(GaussianNB())
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_pp_WAE():
    "Post pruned WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), post_pruning=True, n_estimators=5)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm1():
    "Various weight computation methods of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(
        GaussianNB(), weight_calculation_method="same_for_each", n_estimators=5
    )
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm2():
    "Various weight computation methods of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(
        GaussianNB(), weight_calculation_method="proportional_to_accuracy"
    )
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm3():
    "Various weight computation methods of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(
        GaussianNB(), weight_calculation_method="pta_related_to_whole"
    )
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_wcm4():
    "Various weight computation methods of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), weight_calculation_method="bell_curve")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_am2():
    "Various aging methods of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), aging_method="constant")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_am3():
    "Various aging methods of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), aging_method="gaussian")
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_WAE_rejuvenation():
    "Rejuvenation of WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), rejuvenation_power=0.5)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_pp_WAE_rejuvenation():
    "Post pruning with rejuvenation WAE."
    stream = get_stream()
    clf = sl.ensembles.WAE(GaussianNB(), rejuvenation_power=0.5, post_pruning=True)
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, clf)


def test_REA():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.REA(GaussianNB(), n_estimators=5))


def test_KMC():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.KMC(GaussianNB(), n_estimators=5))


def test_learnppCDS():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.LearnppCDS(GaussianNB(), n_estimators=5))


def test_learnppNIE():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.LearnppNIE(GaussianNB(), n_estimators=5))


def test_OUSE():
    stream = get_stream()
    evaluator = sl.evaluators.TestThenTrain()
    evaluator.process(stream, sl.ensembles.OUSE(GaussianNB(), n_estimators=5, n_chunks=5))
