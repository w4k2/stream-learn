import numpy as np
from sklearn.base import clone
from ..ensembles.base import StreamingEnsemble

class DWM(StreamingEnsemble):
    """DWM"""
    def __init__(self, base_estimator=None, beta=.5, theta=.01, p = 1, weighted=False):
        """Initialization."""
        super().__init__(base_estimator, n_estimators=0, weighted=weighted)
        self.beta = beta
        self.theta = theta
        self.p = p

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Initialization
        if len(self.ensemble_) == 0:
            self.weights_ = np.array([])
            self.age = 0

        # Iterate patterns
        for i, x in enumerate(X):
            # Get label
            y_true = y[i]

            # Kickstart
            if len(self.ensemble_) == 0:
                self.ensemble_.append(clone(self.base_estimator).partial_fit([x], [y_true], classes=self.classes_))
                self.weights_ = np.concatenate((self.weights_, np.array([1])))

            # Gather predictions
            y_preds = np.array([clf.predict([x]) for clf in self.ensemble_])[:,0]

            # Update weights
            if i % self.p == 0:
                self.weights_[y_preds != y_true] = self.weights_[y_preds != y_true] * self.beta

            # Calculate sigma
            sigma = np.array([np.sum((y_preds == cid)*self.weights_) for cid in self.classes_])

            # Get prediction (So logical)
            prediction = np.argmax(sigma)

            # Period validation
            if i % self.p == 0:
                # Normalize weights
                self.weights_ /= np.sum(self.weights_)

                # Theta pruning
                hermann_pruner = self.weights_ > self.theta
                to_destroy = np.where(hermann_pruner==False)[0]
                for j in np.flip(to_destroy):
                    del self.ensemble_[j]
                self.weights_ = self.weights_[hermann_pruner]

                # Add new estimator
                if prediction != y_true:
                    self.ensemble_.append(clone(self.base_estimator).partial_fit([x], [y_true], classes=self.classes_))
                    self.weights_ = np.concatenate((self.weights_, np.array([1])))

            # Update models
            [clf.partial_fit([x], [y_true]) for clf in self.ensemble_]

        self.age += 1

        return self
