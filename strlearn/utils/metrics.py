import numpy as np


def f_score(y_true, y_pred):
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


def recall(y_true, y_pred):
    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    recall = TP / (TP + FP)
    precision = TP / (TP + FN)

    f1 = (2 * (precision * recall)) / (precision + recall)
    return np.nan_to_num(recall)


def precision(y_true, y_pred):
    P = y_true == 1
    N = y_true == 0

    TP = np.sum(y_pred[P] == 1)
    FP = np.sum(y_pred[P] == 0)

    TN = np.sum(y_pred[N] == 0)
    FN = np.sum(y_pred[N] == 1)

    precision = TP / (TP + FN)

    return np.nan_to_num(precision)


def bac(y_true, y_pred):
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

    neg_true = 1 - y_true
    neg_pred = 1 - y_pred

    recall_1 = recall(neg_true[neg_true == 1], neg_pred[neg_true == 1])
    recall_2 = recall(y_true[y_true == 1], y_pred[y_true == 1])

    gmean = (recall_1 * recall_2)**(1/2)

    return np.nan_to_num(gmean)
