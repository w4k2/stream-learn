import numpy as np
from sklearn.svm import LinearSVC


class MD3:
    def __init__(self, sigma=0.15, svc_max_iter=1000):
        self.sigma = sigma
        self.first = True
        self.svc_max_iter = svc_max_iter

    def feed(self, X, y=None):
        if self.first:
            self.model = LinearSVC(max_iter=self.svc_max_iter).fit(X, y)
            self.first = False
            self._is_drift = False

            self.rho_min = np.inf
            self.rho_max = -np.inf

        else:
            decision_scores = self.model.decision_function(X)
            self.rho = np.array(np.abs(decision_scores) < 1).sum() / X.shape[0]

            if self.rho < self.rho_min:
                self.rho_min = self.rho

            if self.rho > self.rho_max:
                self.rho_max = self.rho

            # print(self.rho_max - self.rho_min)
            if self.rho_max - self.rho_min > self.sigma:
                self._is_drift = True
                self.first = True
            else:
                self._is_drift = False
