"""Checking compliance with the sklearn API"""

import sys

from sklearn.utils.estimator_checks import check_estimator

import strlearn as sl

sys.path.insert(0, "../..")


def test_estimators():
    "Checking CBE, WAE and ASC."
    check_estimator(sl.classifiers.AccumulatedSamplesClassifier)
