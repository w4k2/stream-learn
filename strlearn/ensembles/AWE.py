from sklearn.base import clone
from sklearn.model_selection import KFold
import numpy as np
from ..ensembles.base import StreamingEnsemble

class AWE(StreamingEnsemble):
    """
    Accuracy Weighted Ensemble
    """
    def __init__(self, base_estimator=None, n_estimators=10, n_splits=5):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.n_splits = n_splits


    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        super().partial_fit(X, y, classes)

        # Compute baseline
        mser = self.mser(y)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)

        # Calculate its scores
        scores = np.zeros(self.n_splits)
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            scores[fold] = self.msei(fold_candidate, self.X_[test], self.y_[test])

        # Save scores
        candidate_msei = np.mean(scores)
        candidate_weight = mser - candidate_msei

        # Calculate weights of current ensemble
        self.weights_ = [mser - self.msei(clf, self.X_, self.y_) for clf in self.ensemble_]

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]

        return self
