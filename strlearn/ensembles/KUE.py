import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from strlearn.ensembles.base import StreamingEnsemble
from strlearn.classifiers import SampleWeightedMetaEstimator as SWME
from sklearn.metrics import cohen_kappa_score

class KUE(StreamingEnsemble):
    """
    Kappa Updated Ensemble
    """
    def __init__(self, base_estimator=None,
                 n_estimators = 10,
                 n_candidates = 1):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.n_candidates = n_candidates
        self.weighted = True

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Select initial subspaces
        if not hasattr(self, 'subspaces'):
            self.n_features = X.shape[1]
            # TODO: Ensure at least one feature
            self.subspaces = np.random.uniform(0,1,size=(self.n_estimators,
                                                         self.n_features)) > .5

            while np.min(np.sum(self.subspaces, axis=1)) == 0:
                self.subspaces = np.random.uniform(0,1,size=(self.n_estimators,
                                                             self.n_features)) > .5

        # Train initial pool of classifiers
        if len(self.ensemble_) == 0:
            for sid, subspace in enumerate(self.subspaces):
                poi = np.random.poisson(size=y.shape)
                clf = SWME(self.base_estimator)
                clf.partial_fit(X[:,subspace], y,
                                self.classes_, poi)
                self.ensemble_.append(clf)

            # Compute Kappas
            self._compute_kappa(X, y)
        else:
            # Update models in pool
            for sid, subspace in enumerate(self.subspaces):
                poi = np.random.poisson(size=y.shape)
                self.ensemble_[sid].partial_fit(X[:, subspace],
                                                y,
                                                self.classes_,
                                                poi)

            # Compute Kappas
            self._compute_kappa(X, y)

            # Get candidate subspaces
            _subspaces = np.random.uniform(size=(self.n_candidates,
                                                 self.n_features))>.5

            while np.min(np.sum(_subspaces, axis=1)) == 0:
                _subspaces = np.random.uniform(size=(self.n_candidates,
                                                     self.n_features))>.5


            # Train and evaluate candidates
            for sid, subspace in enumerate(_subspaces):
                poi = np.random.poisson(size=y.shape)
                clf = SWME(self.base_estimator)
                clf.partial_fit(X[:,subspace],
                                y, self.classes_,
                                poi)

                y_pred = clf.predict(X[:,subspace])
                candidate_kappa = cohen_kappa_score(y, y_pred)

                worst_kappa = np.min(self.kappas_)
                worst_kappa_idx = np.argmin(self.kappas_)

                if candidate_kappa > worst_kappa:
                    self.ensemble_[worst_kappa_idx] = clf
                    self.subspaces[worst_kappa_idx] = subspace

                # Update weights
                self._compute_kappa(X, y)

        return self

    def _compute_kappa(self, X, y):
        """Compute kappas and weights."""
        self.kappas_ = np.array(
            [cohen_kappa_score(clf.predict(X[:,subspace]), y)
             for clf, subspace in zip(self.ensemble_,
                                      self.subspaces)]
        )
        self.weights_ = np.clip(self.kappas_, 0, 1)

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.nan_to_num(np.array([clf.predict_proba(X[:,subspace])
                                       for clf, subspace in zip(self.ensemble_,
                                                                self.subspaces)]))
