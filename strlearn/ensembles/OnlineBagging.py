"""Online Bagging."""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class OnlineBagging(BaseEstimator, ClassifierMixin):
    """

    """

    def __init__(self, ensemble_size=5):
        """Initialization."""
        self.ensemble_size = ensemble_size

    def set_base_clf(self, base_clf=GaussianNB):
        """Establishing base classifier."""
        self._base_clf = base_clf
        self.ensemble_ = []
        for size in range(self.ensemble_size):
            self.ensemble_.append(self._base_clf())

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self.weights = []
        for instance in range(self.X_.shape[0]):
            K = np.asarray([np.random.poisson(1, 1)[0] for i in range(self.ensemble_size)])
            self.weights.append(K)

        self.weights = np.asarray(self.weights).T


        for w, base_model in enumerate(self.ensemble_):
            base_model.partial_fit(self.X_, self.y_, self.classes_, sample_weight=self.weights[w])

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
