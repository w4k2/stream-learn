from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import numpy as np


class KMC(ClassifierMixin, BaseEnsemble):

    """
    References
    ----------
    .. [1] Wang, Yi, Yang Zhang, and Yong Wang. "Mining data streams
           with skewed distribution by static classifier ensemble."
           Opportunities and Challenges for Next-Generation Applied
           Intelligence. Springer, Berlin, Heidelberg, 2009. 65-71.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.weights_ = []

        # Check if is more than one class
        if len(np.unique(y)) == 1:
            raise ValueError("Only one class in data chunk.")

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        # Resample data
        res_X, res_y = self._resample(X, y)

        # Train new model
        new_classifier = clone(self.base_estimator).fit(res_X, res_y)

        if len(self.ensemble_) < self.n_estimators:

            # Append new estimator
            self.ensemble_.append(new_classifier)
            self.weights_.append(1)

        else:

            # Remove the worst model when ensemble becomes too large
            auc_array = []

            for i in range(len(self.ensemble_)):
                y_score = self.ensemble_[i].predict_proba(res_X)
                auc_array.append(roc_auc_score(res_y, y_score[:, 1]))

            j = np.argmin(auc_array)

            y_score = new_classifier.predict_proba(res_X)
            new_auc = roc_auc_score(res_y, y_score[:, 1])

            if new_auc > auc_array[j]:
                self.ensemble_[j] = new_classifier
                auc_array[j] = new_auc

            for i in range(len(self.ensemble_)):
                self.weights_[i] = auc_array[i]

    def _resample(self, X, y):
        minority, majority = self.minority_majority_split(X, y,
                                                     self.minority_name,
                                                     self.majority_name)

        # Undersample majority array
        km = KMeans(n_clusters=len(minority)).fit(X)
        majority = km.cluster_centers_

        res_X = np.concatenate((majority, minority), axis=0)
        res_y = len(majority)*[self.majority_name] + len(minority)*[self.minority_name]

        return res_X, res_y

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
        esm = esm * np.array(self.weights_)[:, np.newaxis, np.newaxis]
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority : array-like, shape = [n_samples, n_features]
            Minority class samples.
        majority : array-like, shape = [n_samples, n_features]
            Majority class samples.
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns the name of minority and majority class

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority_name : object
            Name of minority class.
        majority_name : object
            Name of majority class.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name
