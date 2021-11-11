from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
import random


class OUSE(ClassifierMixin, BaseEnsemble):
    """
    OUSE

    Gao, Jing, et al. "Classifying Data Streams with Skewed Class Distributions and Concept Drifts." IEEE Internet Computing 12.6 (2008): 37-49.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 n_chunks=10):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.minority_data = []
        self.chunk_time_stamp = []
        self.chunk_sample_proba = []
        self.n_chunks = n_chunks
        self.time_stamp = 1

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

        new_minority = self._resample(X, y)
        minority, majority = self.minority_majority_split(X, y, self.minority_name, self.majority_name)

        majority_split = np.array_split(majority, self.n_estimators)

        self.ensemble_ = []
        for m_s in majority_split:
            res_X = np.concatenate((m_s, new_minority), axis=0)        # maj = self.label_encoder.inverse_transform(maj)

            res_y = len(m_s)*[self.majority_name] + len(new_minority)*[self.minority_name]
            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            self.ensemble_.append(new_classifier)

        self.time_stamp += 1

    def _resample(self, X, y):
        y = np.array(y)
        X = np.array(X)

        minority, majority = self.minority_majority_split(X, y, self.minority_name, self.majority_name)

        self.minority_data.append(minority.tolist())
        self.chunk_time_stamp.append(self.time_stamp)

        if len(self.minority_data) > self.n_chunks:
            del self.minority_data[0]
            del self.chunk_time_stamp[0]

        self.chunk_sample_proba = np.arange(len(self.minority_data))+1
        self.chunk_sample_proba = self.chunk_sample_proba / self.chunk_sample_proba.sum()

        number_of_instances = len(majority)/self.n_estimators

        chunk_indexes = np.random.choice(len(self.chunk_sample_proba), int(number_of_instances), p=self.chunk_sample_proba)
        cia, cca = np.unique(chunk_indexes, return_counts=True)

        new_minority = []
        for chunk_index, chunk_count in zip(cia, cca):
            if len(self.minority_data[chunk_index]) > chunk_count:
                new_minority.extend(random.sample(self.minority_data[chunk_index], chunk_count))
            else:
                new_minority.extend(self.minority_data[chunk_index])

        return new_minority

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
