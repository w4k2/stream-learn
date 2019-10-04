from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn import base
import numpy as np
from sklearn.naive_bayes import GaussianNB


class DumbEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble_size=5):
        self.ensemble_size = ensemble_size
        self.ensemble_ = []

    def set_base_clf(self, base_clf=GaussianNB()):
        self._base_clf = base_clf

    def fit(self, X, y):
        self.set_base_clf()

    def partial_fit(self, X, y):
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
        X, y = check_X_y(X, y)

        self.X_, self.y_ = X, y

        self.ensemble_.append(base.clone(self._base_clf).fit(self.X_, self.y_))

        if len(self.ensemble_) > self.ensemble_size:
            del self.ensemble_[0]

    def ensemble_support_matrix(self, X):
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):

        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return prediction
