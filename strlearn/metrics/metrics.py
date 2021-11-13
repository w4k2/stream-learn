"""Evaluation metrics."""

import numpy as np
from warnings import warn

def binary_confusion_matrix(y_true, y_pred):
    """
    Calculates the binary confusion matrics.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: tuple, (TN, FP, FN, TP)
    :returns: Elements of binary confusion matrix.
    """
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

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Specificity score.
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    return np.nan_to_num(tn / (tn + fp))


def recall(y_true, y_pred):
    """
    Calculates the recall.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Recall score.
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    score = np.nan
    try:
        score = np.nan_to_num(tp / (tp + fn))
    except:
        warn('Recall metric could not be calculated, NaN returned')
    return score


def precision(y_true, y_pred):
    """
    Calculates the precision.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Precision score.
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    return np.nan_to_num(tp / (tp + fp))


def fbeta_score(y_true, y_pred, beta):
    """
    Calculates the F-beta score.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.
    :type beta: float
    :param beta: Beta parameter

    :rtype: float
    :returns: F-beta score.
    """
    pre, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(
        (1 + np.power(beta, 2)) * pre * rec / (np.power(beta, 2) * pre + rec)
    )


def f1_score(y_true, y_pred):
    """
    Calculates the f1_score.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: F1 score.
    """
    return fbeta_score(y_true, y_pred, 1)


def balanced_accuracy_score(y_true, y_pred):
    """
    Calculates the balanced accuracy score.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Balanced accuracy score.
    """
    spe, rec = specificity(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num((rec + spe) / 2)


def geometric_mean_score_1(y_true, y_pred):
    """
    Calculates the geometric mean score.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Geometric mean score.
    """
    spe, rec = specificity(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(np.sqrt(rec * spe))


def geometric_mean_score_2(y_true, y_pred):
    """
    Calculates the geometric mean score.

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Geometric mean score.
    """
    pre, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(np.sqrt(rec * pre))
