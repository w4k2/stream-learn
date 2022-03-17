from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import numpy as np

class StreamingEnsemble(ClassifierMixin, BaseEstimator):
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
        """Partial fitting"""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        self.green_light = True

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Check label consistency
        if len(np.unique(y)) != len(np.unique(self.classes_)):
            y[:len(np.unique(self.classes_))] = np.copy(self.classes_)

        # Check if it is possible to train new estimator
        if len(np.unique(y)) != len(self.classes_):
            self.green_light = False

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        # print('ESM')
        return np.nan_to_num(np.array([member_clf.predict_proba(X)
                                       for member_clf in self.ensemble_]))

    def predict_proba(self, X):
        """Predict proba."""
        esm = self.ensemble_support_matrix(X)
        if self.weighted:
            esm *= np.array(self.weights_)[:, np.newaxis, np.newaxis]

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
