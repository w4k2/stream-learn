"""Evaluation metrics."""

import numpy as np

def binary_confusion_matrix(y_true, y_pred):
    # tn, fp, fn, tp
    return tuple([np.sum((y_pred == i%2) * (y_true == i//2)) for i in range(4)])

def specificity(y_true, y_pred):
    """
    Calculates the specificity.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples, )
        Estimated targets as returned by a classifier.

    Returns
    -------
    specificity : float
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    return tn/(tn+fp)


def recall(y_true, y_pred):
    """
    Calculates the recall.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples, )
        Estimated targets as returned by a classifier.

    Returns
    -------
    recall : float
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    return np.nan_to_num(tp/(tp+fn))

def precision(y_true, y_pred):
    """
    Calculates the precision.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples, )
        Estimated targets as returned by a classifier.

    Returns
    -------
    precision : float
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    return np.nan_to_num(tp/(tp+fp))

def f_score(y_true, y_pred):
    """
    Calculates the f1_score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples, )
        Estimated targets as returned by a classifier.

    Returns
    -------
    f1 : float
    """
    pre, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(2 * pre * rec / (pre + rec))

def bac(y_true, y_pred):
    """
    Calculates the balanced accuracy score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples, )
        Estimated targets as returned by a classifier.

    Returns
    -------
    bac : float
    """
    spe, rec = specificity(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num((rec+spe)/2)

def geometric_mean_score(y_true, y_pred):
    """
    Calculates the geometric mean score.

    Parameters
    ----------
    y_true : array-like, shape (n_samples, )
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples, )
        Estimated targets as returned by a classifier.

    Returns
    -------
    gmean : float
    """
    spe, rec = specificity(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(np.sqrt(rec*spe))
