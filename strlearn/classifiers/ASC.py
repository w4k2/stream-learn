"""Accumulated samples classifier."""

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ASC(BaseEnsemble, ClassifierMixin):
    """
    Accumulated samples classifier.

    Classifier fitted on accumulated samples from all data chunks.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes, )
        The class labels.

    Examples
    --------
    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.classifiers.AccumulatedSamplesClassifier()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
    [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.92       0.91936979 0.91936231 0.91936979 0.9047619 ]
    ...
    [0.92       0.91907051 0.91877671 0.91907051 0.9245283 ]
    [0.885      0.8854889  0.88546135 0.8854889  0.87830688]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(self, base_clf=None):
        """Initialization."""
        self.base_clf = base_clf

    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)

        self.classes_, _ = np.unique(y, return_inverse=True)
        self._clf = clone(self.base_clf).fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)

        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self._X = (
            np.concatenate((self._X, X), axis=0) if hasattr(self, "_X") else np.copy(X)
        )
        self._y = (
            np.concatenate((self._y, y), axis=0) if hasattr(self, "_y") else np.copy(y)
        )

        self._clf = clone(self.base_clf).fit(self._X, self._y)

        return self

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
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict(X)

    def predict_proba(self, X):
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
        check_is_fitted(self, "classes_")
        X = check_array(X)

        return self._clf.predict_proba(X)
