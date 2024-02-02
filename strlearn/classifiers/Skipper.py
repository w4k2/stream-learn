import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import clone

class Skipper(BaseEstimator, ClassifierMixin):
    """
    Skipper.
    """

    def __init__(self, base_clf, n_skips=5):
        """Initialization."""
        self.base_clf = base_clf
        self.n_skips = n_skips
        self.counter = 0

    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        self.clf_ = clone(self.base_clf).fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        if not hasattr(self, 'clf_'):
            self.clf_ = clone(self.base_clf)
            
        X, y = check_X_y(X, y)
        
        if self.counter % self.n_skips == 0:
            self.clf_.partial_fit(X, y, classes)

        self.counter += 1
        
        return self

    def predict(self, X):
        check_is_fitted(self, "clf_")
        X = check_array(X)

        return self.clf_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "clf_")
        X = check_array(X)

        return self.clf_.predict_proba(X)
