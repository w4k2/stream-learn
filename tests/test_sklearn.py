"""Checking scikit-learn compatibility of estimators."""
from sklearn.utils.estimator_checks import check_estimator
from .context import strlearn as sl


def test_WAE_estimator():
    """Testing WAE."""
    return check_estimator(sl.ensembles.WAE)


# def test_REA_estimator():
#    return check_estimator(ensembles.REA)
