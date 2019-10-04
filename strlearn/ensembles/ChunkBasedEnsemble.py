from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
import numpy as np
from sklearn.naive_bayes import GaussianNB


class ChunkBasedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble_size=5):
        self.ensemble_size = ensemble_size
        # self.ensemble_ = []

    def set_base_clf(self, base_clf=GaussianNB):
        self._base_clf = base_clf

    def fit(self, X, y):
        self.partial_fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        X, y = check_X_y(X, y)
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
            self.ensemble_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self.ensemble_.append(self._base_clf().fit(self.X_, self.y_))

        if len(self.ensemble_) > self.ensemble_size:
            del self.ensemble_[0]

        return self

    def ensemble_support_matrix(self, X):
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return self.classes_[prediction]
        # return prediction
