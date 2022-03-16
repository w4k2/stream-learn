import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.naive_bayes import GaussianNB


class SampleWeightedMetaEstimator(BaseEstimator, ClassifierMixin):
    """
    Sample Weighted Meta Estimator.
    """

    def __init__(self, base_classifier=GaussianNB()):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        if not hasattr(self, 'clf_'):
            self.clf_ = clone(self.base_classifier)

        self.clf_.fit(X, y)

    def partial_fit(self, X, y, classes, sample_weight):
        if not hasattr(self, 'clf_'):
            self.clf_ = clone(self.base_classifier)

        X_ = np.array([]).reshape(0, X.shape[1])
        y_ = np.array([])

        for i in range(1, np.max(sample_weight) + 1):
            mask = sample_weight >= i
            X_ = np.concatenate((X_, X[mask, :]), axis=0)
            y_ = np.concatenate((y_, y[mask]))

        self.clf_.partial_fit(X_, y_, classes)

        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)

    def predict(self, X):
        return self.clf_.predict(X)
