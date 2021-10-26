from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold
import numpy as np

class AWE(ClassifierMixin, BaseEnsemble):
    def __init__(self, base_estimator=None, n_estimators=10, n_splits=5):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_splits = n_splits

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Compute baseline
        posterior = np.unique(y, return_counts=True)[1]/len(y)
        mser = np.sum(posterior * np.power((1-posterior), 2))

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)

        # Calculate its scores
        scores = np.zeros(self.n_splits)
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            scores[fold] = self.msei_score(fold_candidate, self.X_[test], self.y_[test])

        # Save scores
        candidate_msei = np.mean(scores)
        candidate_weight = mser - candidate_msei

        # Calculate weights of current ensemble
        self.weights_ = [mser - self.msei_score(clf, self.X_, self.y_) for clf in self.ensemble_]

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]

        return self

    def msei_score(self, clf, X_, y_):
        pprobas = clf.predict_proba(X_)
        probas = np.zeros(len(y_))
        for label in self.classes_:
            probas[y_ == label] = pprobas[y_==label, label]
        return np.sum(np.power(1 - probas, 2))/len(y_)


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
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
