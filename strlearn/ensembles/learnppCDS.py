from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import numpy as np
import math
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


class LearnppCDS(ClassifierMixin, BaseEnsemble):
    """
    LearnppCDS

    Ditzler, Gregory, and Robi Polikar. "Incremental learning of concept drift from streaming imbalanced data." IEEE Transactions on Knowledge and Data Engineering 25.10 (2013): 2283-2301.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 param_a=2,
                 param_b=2):

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


        self.n_instances = len(y)

        if self.ensemble_:
            y_pred = self.predict(X)

            E = (1 - accuracy_score(y, y_pred))

            eq = np.equal(y, y_pred)

            w = np.zeros(eq.shape)
            w[eq == True] = E/float(self.n_instances)
            w[eq == False] = 1/float(self.n_instances)

            w_sum = np.sum(w)

            D = w/w_sum

            res_X, res_y = self._resample(X, y)

            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            self.ensemble_.append(new_classifier)

            beta = []
            epsilon_sum_array = []

            for j in range(len(self.ensemble_)):
                y_pred = self.ensemble_[j].predict(X)

                eq_2 = np.not_equal(y, y_pred).astype(int)

                epsilon_sum = np.sum(eq_2*D)
                epsilon_sum_array.append(epsilon_sum)

                if epsilon_sum > 0.5:
                    if j is len(self.ensemble_) - 1:
                        self.ensemble_[j] = clone(self.base_estimator).fit(res_X, res_y)
                    else:
                        epsilon_sum = 0.5

            epsilon_sum_array = np.array(epsilon_sum_array)
            beta = epsilon_sum_array / (1 - epsilon_sum_array)

            sigma = []
            a = self.param_a
            b = self.param_b
            t = len(self.ensemble_)
            k = np.array(range(t))

            sigma = 1/(1 + np.exp(-a*(t-k-b)))

            sigma_mean = []
            for k in range(t):
                sigma_sum = np.sum(sigma[0:t-k])
                sigma_mean.append(sigma[k]/sigma_sum)

            beta_mean = []
            for k in range(t):
                beta_sum = np.sum(sigma_mean[0:t-k]*beta[0:t-k])
                beta_mean.append(beta_sum)

            self.weights_ = []
            for b in beta_mean:
                self.weights_.append(math.log(1/b))

            if t >= self.n_estimators:
                ind = np.argmax(beta_mean)
                del self.ensemble_[ind]
                del self.weights_[ind]

        else:
            res_X, res_y = self._resample(X, y)

            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            self.ensemble_.append(new_classifier)
            self.weights_ = [1]

    def _resample(self, X, y):
        minority, majority = self.minority_majority_split(X, y,
                                                    self.minority_name,
                                                    self.majority_name)
        if len(minority) > 6:
            res_X, res_y = SMOTE().fit_sample(X, y)
        else:
            res_X, res_y = SMOTE(k_neighbors=len(minority)-1).fit_sample(X, y)
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
        esm = esm * np.array(self.weights_)[:, np.newaxis, np.newaxis]
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
