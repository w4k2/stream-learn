from sklearn.base import clone
from sklearn.model_selection import KFold
import numpy as np
from ..ensembles.base import StreamingEnsemble

class AUE(StreamingEnsemble):
    """ Accuracy Updated Ensemble"""
    def __init__(
        self, base_estimator=None, n_estimators=10, n_splits=5, epsilon=0.0000000001
    ):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.n_splits = n_splits
        self.epsilon = epsilon

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Compute baseline
        mser = self.mser(y)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)

        # Calculate its scores
        scores = []
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            if len(np.unique(y[train])) != len(self.classes_):
                continue
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            msei = self.msei(fold_candidate, self.X_[test], self.y_[test])
            scores.append(msei)

        # Save scores
        candidate_msei = np.mean(scores)
        candidate_weight = 1 / (candidate_msei + self.epsilon)

        # Calculate weights of current ensemble
        self.weights_ = [
            1 / (self.msei(clf, self.X_, self.y_) + self.epsilon)
            for clf in self.ensemble_
        ]

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]

        # AUE update procedure
        comparator = 1 / mser
        for i, clf in enumerate(self.ensemble_):
            if i == len(self.ensemble_) - 1:
                break
            clf.partial_fit(X, y)
        return self
