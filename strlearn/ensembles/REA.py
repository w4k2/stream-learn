import numpy as np
import math
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.neighbors import NearestNeighbors


class REA(ClassifierMixin, BaseEnsemble):
    """
    Recursive Ensemble Approach.

    Sheng Chen, and Haibo He. "Towards incremental learning of nonstationary imbalanced data stream: a multiple selectively recursive approach." Evolving Systems 2.1 (2011): 35-50.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 post_balance_ratio=0.5,
                 k_parameter=10,
                 weighted_support=True):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.post_balance_ratio = post_balance_ratio
        self.k_parameter = k_parameter
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

        # Append new estimator
        self.ensemble_.append(clone(self.base_estimator).fit(res_X, res_y))

        # # Remove the worst model when ensemble becomes too large
        # if len(self.ensemble_) >= self.n_estimators:
        #     worst = np.argmin(self.weights_)
        #     del self.ensemble_[worst]
        #     np.delete(self.weights_, worst)

        # Recalculate the ensemble weights
        weights = []
        for clf in self.ensemble_:
            y_pred = clf.predict(X).astype(int)
            probas_ = clf.predict_proba(X)[np.arange(len(X)), y_pred]
            weights.append(math.log(1/(np.sum((1-probas_)**2)/len(X))))
        self.weights_ = np.array(weights)

        return self

    def _resample(self, X, y):

        # Split the data
        minority, majority = self.minority_majority_split(X, y, self.minority_name, self.majority_name)

        # Check if the minority data has been previously accumulated
        if not hasattr(self, "minority_data_"):
            self.minority_data_ = minority
            return X, y

        # Calculate actual imbalanced ratio
        imbalanced_ratio = len(minority[:, 0])/float(len(X[:, 0]))

        # Resample data
        if self.post_balance_ratio > imbalanced_ratio:

            knn = NearestNeighbors(n_neighbors=self.k_parameter).fit(X)

            indices = knn.kneighbors(self.minority_data_, return_distance=False)

            min_count = np.count_nonzero(y[indices] == self.minority_name, axis=1)

            a = np.arange(0, len(min_count))
            min_count = np.insert(np.expand_dims(min_count, axis=1), 1, a, axis=1)
            min_count = min_count[min_count[:, 0].argsort()]
            min_count = min_count[::-1]

            sorted_minority = min_count[:, 1].astype("int")

            n_instances = int((self.post_balance_ratio - imbalanced_ratio)*len(y))
            n_instances = min(n_instances, len(sorted_minority))
            new_minority = self.minority_data_[sorted_minority[0:n_instances]]

            res_X = np.concatenate((new_minority, majority), axis=0)
            res_y = np.concatenate((np.full(len(new_minority), self.minority_name), np.full(len(majority), self.majority_name)), axis=0)

        else:
            res_X = X
            res_y = y

        self.minority_data_ = np.concatenate((minority, self.minority_data_), axis=0)

        return res_X, res_y

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        esm = esm * np.array(self.weights_)[:, np.newaxis, np.newaxis]
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

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.
        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (array-like, shape = [n_samples, n_features], array-like, shape = [n_samples, n_features])
        :returns: Tuple of minority and majority class samples
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns minority and majority data

        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (object, object)
        :returns: Tuple of minority and majority class names.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name
