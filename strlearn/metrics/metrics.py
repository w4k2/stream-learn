"""Evaluation metrics."""

import numpy as np

def binary_confusion_matrix(y_true, y_pred):
    # tn, fp, fn, tp
    b = y_pred.astype(bool)
    a = np.logical_not(b)
    d = y_true.astype(bool)
    c = np.logical_not(d)
    ttt = (np.count_nonzero(a*c), np.count_nonzero(b*c),
           np.count_nonzero(a*d), np.count_nonzero(b*d))

    return ttt


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
    return np.nan_to_num(tn / (tn + fp))


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
    return np.nan_to_num(tp / (tp + fn))


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
    return np.nan_to_num(tp / (tp + fp))


def fbeta_score(y_true, y_pred, beta):
    pre, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(
        (1 + np.power(beta, 2)) * pre * rec / (np.power(beta, 2) * pre + rec)
    )


def f1_score(y_true, y_pred):
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
    return fbeta_score(y_true, y_pred, 1)


def balanced_accuracy_score(y_true, y_pred):
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
    balanced_accuracy_score : float
    """
    spe, rec = specificity(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num((rec + spe) / 2)


def geometric_mean_score_1(y_true, y_pred):
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
    return np.nan_to_num(np.sqrt(rec * spe))


def geometric_mean_score_2(y_true, y_pred):
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
    pre, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(np.sqrt(rec * pre))
