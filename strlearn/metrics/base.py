"""Evaluation metrics."""

import numpy as np
from warnings import warn

def binary_confusion_matrix(y_true, y_pred):
    """
    Calculates the binary confusion matrics.

    .. image:: plots/confusion_matrix.png

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

    .. math::

        Specificity = \\frac{tn}{tn + fp}

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Specificity score.
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)
    score = np.nan
    try:
        score = np.nan_to_num(tn / (tn + fp))
    except:
        warn('Recall metric could not be calculated, NaN returned')
    return score


def recall(y_true, y_pred):
    """
    Calculates the recall.

    Recall (also known as sensitivity or true positive rate) represents the
    classifier's ability to find all the positive data samples in the dataset
    (e.g. the minority class instances) and is denoted as

    .. math::

        Recall = \\frac{tp}{tp + fn}

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

    Precision (also called positive predictive value) expresses the probability
    of correct detection of positive samples and is denoted as

    .. math::

        Precision = \\frac{tp}{tp + fp}

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Precision score.
    """
    tn, fp, fn, tp = binary_confusion_matrix(y_true, y_pred)

    score = np.nan
    try:
        score = np.nan_to_num(tp / (tp + fp))
    except:
        warn('Recall metric could not be calculated, NaN returned')
    return score


def fbeta_score(y_true, y_pred, beta):
    """
    Calculates the F-beta score.

    The F-beta score can be interpreted as a weighted harmonic mean of precision and recall taking both metrics into account and punishing extreme values. The ``beta`` parameter determines the recall's weight. ``beta`` < 1 gives more weight to precision, while ``beta`` > 1 prefers recall. The formula for the F-beta score is

    .. math::

        F_\\beta = (1+\\beta^2) * \\frac{Precision * Recall}{(\\beta^2 * Precision) + Recall}

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

    The F1 score can be interpreted as a F-beta score, where :math:`\beta` parameter equals 1. It is a harmonic mean of precision and recall.
    The formula for the F1 score is

    .. math::

        F_1 = 2 * \\frac{Precision * Recall}{Precision + Recall}

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

    The balanced accuracy for the multiclass problems is defined as the average of recall obtained on each class. For binary problems it is denoted by the average of recall and specificity (also called true negative rate).

    .. math::

        BAC = \\frac{Recall + Specificity}{2}

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

    The geometric mean (G-mean) tries to maximize the accuracy on each of the classes while keeping these accuracies balanced. For N-class problems it is a N root of the product of class-wise recall. For binary classification G-mean is denoted as the squared root of the product of the recall and specificity.

    .. math::

        Gmean1 = \sqrt{Recall * Specificity}

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

    The alternative definition of G-mean measure. For binary classification G-mean is denoted as the squared root of the product of the recall and precision.

    .. math::
        Gmean2 = \sqrt{Recall * Precision}

    :type y_true: array-like, shape (n_samples)
    :param y_true: True labels.
    :type y_pred: array-like, shape (n_samples)
    :param y_pred: Predicted labels.

    :rtype: float
    :returns: Geometric mean score.
    """
    pre, rec = precision(y_true, y_pred), recall(y_true, y_pred)
    return np.nan_to_num(np.sqrt(rec * pre))
