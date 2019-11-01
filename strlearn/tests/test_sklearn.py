"""Checking compliance with the sklearn API"""

import sys
import strlearn as sl
from sklearn.utils.estimator_checks import check_estimator

sys.path.insert(0, "../..")


def test_estimators():
    "Checking CBE, WAE and ASC."
    check_estimator(sl.ensembles.ChunkBasedEnsemble)
    check_estimator(sl.ensembles.WAE)
    check_estimator(sl.classifiers.AccumulatedSamplesClassifier)
