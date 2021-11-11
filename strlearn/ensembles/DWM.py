import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

class DWM(ClassifierMixin, BaseEnsemble):
    """DWM"""
    def __init__(self, base_estimator=None, beta=.5, theta=.01, p = 1, weighted_support=False):
        """Initialization."""
        self.base_estimator = base_estimator
        self.beta = beta
        self.theta = theta
        self.p = p
        self.weighted_support = weighted_support


    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.weights_ = np.array([])
            self.age = 0

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Iterate patterns
        for i, x in enumerate(X):
            # Get label
            y_true = y[i]

            # Kickstart
            if len(self.ensemble_) == 0:
                self.ensemble_.append(clone(self.base_estimator).partial_fit([x], [y_true], classes=self.classes_))
                self.weights_ = np.concatenate((self.weights_, np.array([1])))

            # Gather predictions
            y_preds = np.array([clf.predict([x]) for clf in self.ensemble_])[:,0]

            # Update weights
            if i % self.p == 0:
                self.weights_[y_preds != y_true] = self.weights_[y_preds != y_true] * self.beta

            # Calculate sigma
            sigma = np.array([np.sum((y_preds == cid)*self.weights_) for cid in self.classes_])

            # Get prediction (So logical)
            prediction = np.argmax(sigma)

            # Period validation
            if i % self.p == 0:
                # Normalize weights
                self.weights_ /= np.sum(self.weights_)

                # Theta pruning
                hermann_pruner = self.weights_ > self.theta
                to_destroy = np.where(hermann_pruner==False)[0]
                for j in np.flip(to_destroy):
                    del self.ensemble_[j]
                self.weights_ = self.weights_[hermann_pruner]

                # Add new estimator
                if prediction != y_true:
                    self.ensemble_.append(clone(self.base_estimator).partial_fit([x], [y_true], classes=self.classes_))
                    self.weights_ = np.concatenate((self.weights_, np.array([1])))

            # Update models
            [clf.partial_fit([x], [y_true]) for clf in self.ensemble_]

        self.age += 1

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
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

        esm = np.nan_to_num(self.ensemble_support_matrix(X))
        if self.weighted_support:
            esm = esm * self.weights_[:, np.newaxis, np.newaxis]

        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
