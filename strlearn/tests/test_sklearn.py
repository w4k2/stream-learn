"""Checking scikit-learn compatibility of estimators."""
from sklearn.utils.estimator_checks import check_estimator
from strlearn import ensembles


def test_WAE_estimator():
    """Testing WAE."""
    return check_estimator(ensembles.WAE)

# def test_REA_estimator():
#    return check_estimator(ensembles.REA)
