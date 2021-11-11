from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import f1_score
import numpy as np
import math


class LearnppNIE(ClassifierMixin, BaseEnsemble):
    """
    LearnppNIE

    Ditzler, Gregory, and Robi Polikar. "Incremental learning of concept drift from streaming imbalanced data." IEEE Transactions on Knowledge and Data Engineering 25.10 (2013): 2283-2301.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=5,
                 param_a=1,
                 param_b=1):

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.param_a = param_a
        self.param_b = param_b

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

        self.ensemble_ += [self._new_sub_ensemble(X, y)]

        beta_mean = self._calculate_weights(X, y)

        self.weights_ = []
        for b in beta_mean:
            self.weights_.append(math.log(1/b))

        if len(self.ensemble_) >= self.n_estimators:
            ind = np.argmax(beta_mean)
            del self.ensemble_[ind]
            del self.weights_[ind]

    def _calculate_weights(self, X, y):
        beta = []
        for i in range(len(self.ensemble_)):
            epsilon = 1-f1_score(y, self._sub_ensemble_predict(i, X))
            if epsilon > 0.5:
                if i == len(self.ensemble_) - 1:
                    self.ensemble_[i] = self._new_sub_ensemble(X, y)
                    epsilon = 0.5
                else:
                    epsilon = 0.5
            beta.append(epsilon / float(1 - epsilon))

        sigma = []
        a = self.param_a
        b = self.param_b
        t = len(self.ensemble_)
        k = np.array(range(t))

        sigma = 1/(1 + np.exp(-a*(t-k-b)))

        sigma_mean = []
        for k in range(t):
            sigma_sum = 0
            for j in range(t-k):
                sigma_sum += sigma[j]
            sigma_mean.append(sigma[k]/sigma_sum)

        beta_mean = []
        for k in range(t):
            beta_sum = 0
            for j in range(t-k):
                beta_sum += sigma_mean[j]*beta[j]
            beta_mean.append(beta_sum)

        return beta_mean

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([self._sub_ensemble_predict_proba(idx, X) for idx, member_clf in enumerate(self.ensemble_)])

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

    def _new_sub_ensemble(self, X, y):
        y = np.array(y)
        X = np.array(X)

        minority, majority = self.minority_majority_split(X, y,
                                                     self.minority_name,
                                                     self.majority_name)

        T = self.n_estimators
        N = len(X)
        sub_ensemble = []
        for k in range(T):
            number_of_instances = int(math.floor(N/float(T)))
            idx = np.random.randint(len(majority), size=number_of_instances)
            sample = majority[idx, :]
            res_X = np.concatenate((sample, minority), axis=0)
            res_y = len(sample)*[self.majority_name] + len(minority)*[self.minority_name]
            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            sub_ensemble += [new_classifier]
        return sub_ensemble

    def _sub_ensemble_predict_proba(self, i, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble_[i]]
        return np.average(probas_, axis=0)

    def _sub_ensemble_predict(self, i, X):
        predictions = np.asarray([clf.predict(X) for clf in self.ensemble_[i]]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                  axis=1,
                                  arr=predictions)
        return maj

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
