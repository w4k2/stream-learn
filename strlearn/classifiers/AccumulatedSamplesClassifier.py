from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import numpy as np


class AccumulatedSamplesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=GaussianNB):
        self.base_classifier = base_classifier

    def fit(self, X, y):
        self.clf = self.base_classifier().fit(X, y)

    def partial_fit(self, X, y, classes=None):
        self._X = (
            np.concatenate((self._X, X), axis=0) if hasattr(self, "_X") else np.copy(X)
        )
        self._y = (
            np.concatenate((self._y, y), axis=0) if hasattr(self, "_y") else np.copy(y)
        )
        self.fit(self._X, self._y)

    def predict(self, X):
        return self.clf.predict(X)
