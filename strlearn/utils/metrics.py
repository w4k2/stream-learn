"""Evaluation metrics."""

import numpy as np

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
    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    recall = TP / (TP + FP)

    return np.nan_to_num(recall)


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
    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    precision = TP / (TP + FN)

    return np.nan_to_num(precision)


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
    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    recall = TP / (TP + FP)
    precision = TP / (TP + FN)

    f1 = (2 * (precision * recall)) / (precision + recall)
    return np.nan_to_num(f1)

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
    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    recall_a = TP / (TP + FP)
    recall_b = TN / (TN + FN)

    bac = (recall_a + recall_b) / 2

    return np.nan_to_num(bac)

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

    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    recall_a = TP / (TP + FP)
    recall_b = TN / (TN + FN)

    gmean = (recall_a * recall_b)**(1/2)

    return np.nan_to_num(gmean)
