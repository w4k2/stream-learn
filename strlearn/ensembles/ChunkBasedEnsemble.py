"""Chunk based ensemble."""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from sklearn.naive_bayes import GaussianNB


class ChunkBasedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Chunk based ensemble classifier.

    Ensemble classifier composed of estimators trained on the fixed
    number of previously seen data chunks.

    Parameters
    ----------
    ensemble_size : integer, optional (default=5)
        The maximum number of estimators trained using consecutive data chunks
        and maintained in the ensemble.

    Attributes
    ----------
    ensemble_ : list of classifiers
        The collection of fitted sub-estimators.
    classes_ : array-like, shape (n_classes, )
        The class labels.

    Examples
    --------
    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.ensembles.ChunkBasedEnsemble()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
   [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.925      0.92364329 0.92360881 0.92364329 0.91017964]
    ...
    [0.925      0.92427885 0.924103   0.92427885 0.92890995]
    [0.89       0.89016179 0.89015879 0.89016179 0.88297872]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(self, ensemble_size=5):
        """Initialization."""
        self.ensemble_size = ensemble_size

    def set_base_clf(self, base_clf=GaussianNB):
        """Establishing base classifier."""
        self._base_clf = base_clf

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
            self.ensemble_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        self.ensemble_.append(self._base_clf().fit(self.X_, self.y_))

        if len(self.ensemble_) > self.ensemble_size:
            del self.ensemble_[0]

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
