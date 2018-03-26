import numpy as np


def minority_majority_split(X, y, minority_name, majority_name):
    """Returns minority and majority data

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The training input samples.
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    minority : array-like, shape (n_samples, n_features)
        Minority class samples.
    majority : array-like, shape (n_samples, n_features)
        Majority class samples.
    """

    minority_ma = np.ma.masked_where(y == minority_name, y)
    minority = X[minority_ma.mask]

    majority_ma = np.ma.masked_where(y == majority_name, y)
    majority = X[majority_ma.mask]

    return minority, majority


def minority_majority_name(y):
    """Returns the name of minority and majority class

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        The target values.

    Returns
    -------
    minority_name : object
        Name of minority class.
    majority_name : object
        Name of majority class.
    """

    unique, counts = np.unique(y, return_counts=True)
    if counts[0] > counts[1]:
        majority_name = unique[0]
        minority_name = unique[1]
    else:
        majority_name = unique[1]
        minority_name = unique[0]

    return minority_name, majority_name
