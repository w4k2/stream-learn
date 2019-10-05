from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class AccumulatedSamplesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def set_base_clf(self, base_clf=GaussianNB):
        self._base_clf = base_clf

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        self.classes_, _ = np.unique(y, return_inverse=True)
        self._clf = self._base_clf().fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        X, y = check_X_y(X, y)
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self._X = (
            np.concatenate((self._X, X), axis=0) if hasattr(self, "_X") else np.copy(X)
        )
        self._y = (
            np.concatenate((self._y, y), axis=0) if hasattr(self, "_y") else np.copy(y)
        )

        self._clf = self._base_clf().fit(self._X, self._y)

        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict(X)
