from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import numpy as np

class StreamingEnsemble(ClassifierMixin):
    """Abstract, base ensemble streaming class"""
    def __init__(self, base_estimator, n_estimators, weighted=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.weighted = weighted

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
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

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.nan_to_num(np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_]))

    def predict_proba(self, X):
        """Predict proba."""
        esm = self.ensemble_support_matrix(X)
        if self.weighted:
            esm *= self.weights_[:, np.newaxis, np.newaxis]

        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.

        :rtype: array-like, shape (n_samples, )
        :returns: The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        prediction = np.argmax(self.predict_proba(X), axis=1)

        # Return prediction
        return self.classes_[prediction]

    def prior_proba(self, y):
        """Calculate prior probability for given labels"""
        return np.unique(y, return_counts=True)[1]/len(y)

    def msei(self, clf, X, y):
        """MSEi score from original AWE algorithm."""
        pprobas = clf.predict_proba(X)
        probas = np.zeros(len(y))
        for label in self.classes_:
            probas[y == label] = pprobas[y == label, label]
        return np.sum(np.power(1 - probas, 2)) / len(y)

    def mser(self, y):
        """MSEr score from original AWE algorithm."""
        prior_proba = self.prior_proba(y)
        return np.sum(prior_proba * np.power((1-prior_proba), 2))
