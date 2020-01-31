"""Oversamping-based Online Bagging."""

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OOB(BaseEnsemble, ClassifierMixin):
    """
    Oversamping-Based Online Bagging.
    """

    def __init__(self, base_estimator=None, n_estimators=5, time_decay_factor=0.9):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.time_decay_factor = time_decay_factor

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # time decayed class sizes tracking
        if not hasattr(self, "last_instance_sizes"):
            self.current_tdcs_ = np.zeros((1, 2))
        else:
            self.current_ctdcs_ = self.last_instance_sizes

        self.chunk_tdcs = np.ones((self.X_.shape[0], self.classes_.shape[0]))

        for iteration, label in enumerate(self.y_):
            if label == 0:
                self.current_tdcs_[0, 0] = (
                    self.current_tdcs_[0, 0] * self.time_decay_factor
                ) + (1 - self.time_decay_factor)
                self.current_tdcs_[0, 1] = (
                    self.current_tdcs_[0, 1] * self.time_decay_factor
                )
            else:
                self.current_tdcs_[0, 1] = (
                    self.current_tdcs_[0, 1] * self.time_decay_factor
                ) + (1 - self.time_decay_factor)
                self.current_tdcs_[0, 0] = (
                    self.current_tdcs_[0, 0] * self.time_decay_factor
                )

            self.chunk_tdcs[iteration] = self.current_tdcs_

        self.last_instance_sizes = self.current_tdcs_

        # improved OOB
        self.weights = []
        for instance, label in enumerate(self.y_):
            if (
                label == 1
                and self.chunk_tdcs[instance][1] < self.chunk_tdcs[instance][0]
            ):
                lmbda = self.chunk_tdcs[instance][0] / \
                    self.chunk_tdcs[instance][1]
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )
            elif (
                label == 0
                and self.chunk_tdcs[instance][0] < self.chunk_tdcs[instance][1]
            ):
                lmbda = self.chunk_tdcs[instance][1] / \
                    self.chunk_tdcs[instance][0]
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )
            else:
                lmbda = 1
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )

            self.weights.append(K)

        self.weights = np.asarray(self.weights).T

        for w, base_model in enumerate(self.ensemble_):
            base_model.partial_fit(
                self.X_, self.y_, self.classes_, sample_weight=self.weights[w]
            )

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
