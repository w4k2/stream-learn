"""
Dumb Delay Pool.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn import neighbors
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from deslib.des import KNORAE, KNORAU
from deslib.dcs import Rank, LCA
from imblearn.over_sampling import RandomOverSampler

measure = accuracy_score


class PrunedEnsemble(BaseEstimator, ClassifierMixin):
    """
    DumbDelayPool.

    Opis niezwykle istotnego klasyfikatora

    References
    ----------
    .. [1] A. Kowalski, B. Nowak, "Bardzo ważna praca o klasyfikatorze
    niezwykle istotnym dla przetrwania gatunku ludzkiego."

    """

    def __init__(self, ensemble_size=3, alpha=0.05):
        """Initialization."""
        self.ensemble_size = ensemble_size
        self.alpha = alpha

    def set_base_clf(self, base_clf=neighbors.KNeighborsClassifier()):
        """Establish base classifier."""
        self._base_clf = base_clf

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        candidate_clf = base.clone(self._base_clf)
        candidate_clf.fit(X, y)

        self.ensemble_ = [candidate_clf]

        # Return the classifier
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
        X, y = check_X_y(X, y)

        self.X_, self.y_ = X, y

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes
            self.ensemble_ = []
            self.previous_X = self.X_
            self.previous_y = self.y_

        self.previous_X = self.X_
        self.previous_y = self.y_

        # Testing all models
        scores = np.array([measure(y, clf.predict(X)) for clf in self.ensemble_])

        # Pruning
        if len(self.ensemble_) > 1:
            alpha_good = scores > (0.5 + self.alpha)
            # print(scores)
            self.ensemble_ = [self.ensemble_[i] for i in np.where(alpha_good)[0]]

        if len(self.ensemble_) > self.ensemble_size - 1:
            worst = np.argmin(scores)
            del self.ensemble_[worst]

        # Preparing and training new candidate
        self.ensemble_.append(base.clone(self._base_clf).fit(X, y))

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        """Hard decision."""
        # print("PREDICT")
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return prediction

    def score(self, X, y):
        return measure(y, self.predict(X))
