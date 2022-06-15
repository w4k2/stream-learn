import numpy as np
from sklearn.base import clone
from ..ensembles.base import StreamingEnsemble

class OOB(StreamingEnsemble):
    """
    Oversamping-Based Online Bagging.
    """
    def __init__(self, base_estimator=None, n_estimators=5, time_decay_factor=0.9):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.time_decay_factor = time_decay_factor

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        if len(self.ensemble_) == 0:
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]

        # time decayed class sizes tracking
        if not hasattr(self, "last_instance_sizes"):
            self.current_tdcs_ = np.zeros((1, 2))
        else:
            self.current_tdcs_ = self.last_instance_sizes

        self.chunk_tdcs = np.ones((self.X_.shape[0], self.classes_.shape[0]))

        for iteration, label in enumerate(self.y_):
            if label == 0:
                self.current_tdcs_[0, 0] = (
                    self.current_tdcs_[0, 0] * self.time_decay_factor
                ) + (1 - self.time_decay_factor)
                self.current_tdcs_[0, 1] = (
                    self.current_tdcs_[0, 1] * self.time_decay_factor
                )
            else:
                self.current_tdcs_[0, 1] = (
                    self.current_tdcs_[0, 1] * self.time_decay_factor
                ) + (1 - self.time_decay_factor)
                self.current_tdcs_[0, 0] = (
                    self.current_tdcs_[0, 0] * self.time_decay_factor
                )

            self.chunk_tdcs[iteration] = self.current_tdcs_

        self.last_instance_sizes = self.current_tdcs_

        # improved OOB
        self.weights = []
        for instance, label in enumerate(self.y_):
            if (
                label == 1
                and self.chunk_tdcs[instance][1] < self.chunk_tdcs[instance][0]
            ):
                lmbda = self.chunk_tdcs[instance][0] / \
                    self.chunk_tdcs[instance][1]
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )
            elif (
                label == 0
                and self.chunk_tdcs[instance][0] < self.chunk_tdcs[instance][1]
            ):
                lmbda = self.chunk_tdcs[instance][1] / \
                    self.chunk_tdcs[instance][0]
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )
            else:
                lmbda = 1
                K = np.asarray(
                    [np.random.poisson(lmbda, 1)[0]
                     for i in range(self.n_estimators)]
                )

            self.weights.append(K)

        self.weights = np.asarray(self.weights).T

        for w, base_model in enumerate(self.ensemble_):
            base_model.partial_fit(
                self.X_, self.y_, self.classes_, sample_weight=self.weights[w]
            )

        return self
