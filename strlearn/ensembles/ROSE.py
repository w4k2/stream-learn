import numpy as np
from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.model_selection import KFold
from strlearn.ensembles.base import StreamingEnsemble
from strlearn.classifiers import SampleWeightedMetaEstimator as SWME
from sklearn.metrics import cohen_kappa_score, accuracy_score

class ROSE(StreamingEnsemble):
    """
    Robust Online Self-Adjusting Ensemble
    """
    def __init__(self, base_estimator=None,
                 n_estimators = 10,
                 n_candidates = 1,
                 subspace_mean=.7,
                 buffer_limit=1000,
                 min_lambda=4):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.n_candidates = n_candidates
        self.weighted = True
        self.subspace_mean = subspace_mean
        self.min_lambda = min_lambda
        self.buffer_limit = buffer_limit

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self
        
        # Gather initial parameters
        if len(self.ensemble_) == 0:
            self.background_ensemble_ = []
            self.drift_detectors = []
            self.n_classes = len(np.unique(y))
            self.n_features = X.shape[1]
            self.sample_buffer = [np.zeros((0, self.n_features)) for label in self.classes_]
            self.class_buffer_limit = self.buffer_limit // self.n_classes
        
        # Update sliding window buffer
        for class_idx in self.classes_:
            # Get both elements of union
            current = self.sample_buffer[class_idx]
            incoming = X[y==self.classes_[class_idx]]
            
            # Make union
            union = np.concatenate((current, incoming))
            
            # Cut the history
            if union.shape[0] > self.class_buffer_limit:
                union = union[-self.class_buffer_limit:]
            
            # Store
            self.sample_buffer[class_idx] = union 
            
        # Select initial subspaces
        if not hasattr(self, 'subspaces'):
            # Establish random subspace size
            self.ers = (self.subspace_mean * self.n_features + ((1-self.subspace_mean)*self.n_features*np.random.normal(size=self.n_estimators))/2)/self.n_features
            self.subspaces = np.random.uniform(0,1,size=(self.n_estimators, self.n_features)) > self.ers[:,None]

            # TODO: Ensure at least one feature
            while np.min(np.sum(self.subspaces, axis=1)) == 0:
                self.ers = np.random.normal(self.subspace_mean, size=self.n_estimators)
                self.subspaces = np.random.uniform(0,1,size=(self.n_estimators,
                                                             self.n_features)) < self.ers[:,None]

        # Calculate lambada (it's actually lambda, but you know)
        hist = np.unique(y, return_counts=True)[1]       
        self.lambada = self.min_lambda + np.log10(np.max(hist)/np.min(hist)) * self.min_lambda

        # Train initial pool of classifiers
        if len(self.ensemble_) == 0:
            for sid, subspace in enumerate(self.subspaces):
                poi = np.random.poisson(lam=self.lambada, size=y.shape)
                clf = SWME(self.base_estimator)
                clf.partial_fit(X[:,subspace], y,
                                self.classes_, poi)
                self.ensemble_.append(clf)
                self.drift_detectors.append(ADWIN())

            # Compute Weights
            self._compute_weights(X, y)
        else:
            # Update models in pool
            for sid, subspace in enumerate(self.subspaces):
                poi = np.random.poisson(lam=self.lambada, size=y.shape)
                
                self.drift_detectors[sid].feed(
                    X[:, subspace], y, self.ensemble_[sid].predict(X[:, subspace])
                )
                self.ensemble_[sid].partial_fit(X[:, subspace],
                                                y,
                                                self.classes_,
                                                poi)
            
            # Verify drift detectors
            if np.max([dd.drift[-1] for dd in self.drift_detectors]) > 0:
                # If background ensemble is empty
                if len(self.background_ensemble_) == 0:
                    # Establish random subspace size
                    self.background_ers = (self.subspace_mean * self.n_features + ((1-self.subspace_mean)*self.n_features*np.random.normal(size=self.n_estimators))/2)/self.n_features
                    self.background_subspaces = np.random.uniform(0,1,size=(self.n_estimators, self.n_features)) > self.background_ers[:,None]

                    # TODO: Ensure at least one feature
                    while np.min(np.sum(self.background_subspaces, axis=1)) == 0:
                        self.background_ers = np.random.normal(self.subspace_mean, size=self.n_estimators)
                        self.background_subspaces = np.random.uniform(0,1,size=(self.n_estimators,
                                                                    self.n_features)) < self.background_ers[:,None]

                    # Make base training set [only for initialization during ADWIN warning]
                    adwin_X = np.concatenate(self.sample_buffer, axis=0)
                    adwin_y = np.concatenate([np.ones(l.shape[0]) * self.classes_[l_idx]  
                                            for l_idx, l in enumerate(self.sample_buffer)])
                    
                    # Initial background subspace train
                    for sid, subspace in enumerate(self.background_subspaces):
                        poi = np.random.poisson(lam=self.lambada, size=adwin_y.shape)
                        clf = SWME(self.base_estimator)
                        clf.partial_fit(adwin_X[:,subspace], adwin_y,
                                        self.classes_, poi)
                        self.background_ensemble_.append(clf)
                        
                else:
                    # Update models in pool
                    for sid, subspace in enumerate(self.background_subspaces):
                        poi = np.random.poisson(lam=self.lambada, size=y.shape)
                        self.background_ensemble_[sid].partial_fit(X[:, subspace],
                                                                   y,
                                                                   self.classes_,
                                                                   poi)
            # Compute weights
            self._compute_weights(X, y)
            
            # Release ensemble pressure
            if len(self.background_ensemble_) and np.sum([s.shape[0] for s in self.sample_buffer]) >= self.buffer_limit:
                # Establish best estimators from both ensembles
                best_idx = np.argsort(-np.concatenate((self.weights_,self.background_weights_)))[:self.n_estimators]
                
                # Construct new ensemble
                new_ensemble = []
                new_subspaces = []
                new_detectors = []
                
                for idx in best_idx:
                    if idx >= self.n_estimators:
                        ii = idx % self.n_estimators
                        clf = self.background_ensemble_[ii]
                        subspace = self.background_subspaces[ii]
                        detector = ADWIN()
                    else:
                        ii = idx
                        clf = self.ensemble_[ii]
                        subspace = self.subspaces[ii]
                        detector = self.drift_detectors[ii]
                    
                    new_ensemble.append(clf)
                    new_subspaces.append(subspace)
                    new_detectors.append(detector)
                
                # Store new ensemble
                self.ensemble_ = new_ensemble
                self.subspaces = np.array(new_subspaces)
                self.drift_detectors = new_detectors
                
                # Clear background ensemble
                self.background_ensemble_ = []
                self.background_subspaces = []

        return self

    def _compute_weights(self, X, y):
        """Compute kappas and weights."""
        self.kappas_ = np.array(
            [cohen_kappa_score(clf.predict(X[:,subspace]), y)
             for clf, subspace in zip(self.ensemble_,
                                      self.subspaces)]
        )
        self.accuracies_ = np.array(
            [accuracy_score(clf.predict(X[:,subspace]), y)
             for clf, subspace in zip(self.ensemble_,
                                      self.subspaces)]
        )
        self.weights_ = np.clip(self.kappas_*self.accuracies_, 0, 1)
        
        if len(self.background_ensemble_) > 0:
            self.background_kappas_ = np.array(
                [cohen_kappa_score(clf.predict(X[:,subspace]), y)
                for clf, subspace in zip(self.background_ensemble_,
                                         self.background_subspaces)]
            )
            self.background_accuracies_ = np.array(
                [accuracy_score(clf.predict(X[:,subspace]), y)
                for clf, subspace in zip(self.background_ensemble_,
                                         self.background_subspaces)]
            )
            self.background_weights_ = np.clip(self.background_kappas_*self.background_accuracies_, 0, 1)

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.nan_to_num(np.array([clf.predict_proba(X[:,subspace])
                                       for clf, subspace in zip(self.ensemble_,
                                                                self.subspaces)]))

class ADWIN(BaseEstimator, ClassifierMixin):
    def __init__(self, delta = 0.002):
        self.delta = delta
        self.drift = []

    def feed(self, X, y, pred):

        if not hasattr(self, "mu_W"):
            self.W = np.copy(X)
            self.Wy = np.copy(y)
            self.p = np.copy(pred)
            self.mu_W = []
            self.sizes = []
            self.drift.append(0)
        else:
            self.W = np.append(self.W, X, axis=0)
            self.Wy = np.append(self.Wy, y, axis=0)
            self.p = np.append(self.p, pred, axis=0)
            values = np.array(self.p == self.Wy)
            var = np.var(values)
            delta_p = self.delta/self.W.shape[0]

            step = int(np.sqrt(self.W.shape[0]))

            self.isdrift = False
            for i in range(1, self.W.shape[0], step):
                m = 1/((1/self.W[:i].shape[0]) + (1/self.W[i:].shape[0]))
                uw0, uw1 = np.mean(values[:i]), np.mean(values[i:])
                cut = np.sqrt((2/m) * var * np.log(2/delta_p)) + (2/(3*m)) * np.log(2/delta_p)

                if np.abs(uw0 - uw1) >= cut:
                    self.W = self.W[i:]
                    self.Wy = self.Wy[i:]
                    self.p = self.p[i:]
                    self.drift.append(2)
                    self.isdrift = True

                    break
            if self.isdrift == False:
                self.drift.append(0)

        self.mu_W.append(np.mean(self.W))
        self.sizes.append(self.W.shape[0])

        return self