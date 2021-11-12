from sklearn.base import clone
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from ..ensembles.base import StreamingEnsemble

class CDS(StreamingEnsemble):
    """
    CDS

    Ditzler, Gregory, and Robi Polikar. "Incremental learning of concept drift from streaming imbalanced data." IEEE Transactions on Knowledge and Data Engineering 25.10 (2013): 2283-2301.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 a=2,
                 b=2):
        super().__init__(base_estimator, n_estimators, weighted=True)
        self.a = a
        self.b = b

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        self.n_instances = len(y)

        if self.ensemble_:
            y_pred = self.predict(X)

            E = (1 - accuracy_score(y, y_pred))

            eq = np.equal(y, y_pred)

            w = np.zeros(eq.shape)
            w[eq == True] = E/float(self.n_instances)
            w[eq == False] = 1/float(self.n_instances)

            w_sum = np.sum(w)

            D = w/w_sum

            res_X, res_y = self._resample(X, y)

            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            self.ensemble_.append(new_classifier)

            beta = []
            epsilon_sum_array = []

            for j in range(len(self.ensemble_)):
                y_pred = self.ensemble_[j].predict(X)

                eq_2 = np.not_equal(y, y_pred).astype(int)

                epsilon_sum = np.sum(eq_2*D)
                epsilon_sum_array.append(epsilon_sum)

                if epsilon_sum > 0.5:
                    if j is len(self.ensemble_) - 1:
                        self.ensemble_[j] = clone(self.base_estimator).fit(res_X, res_y)
                    else:
                        epsilon_sum = 0.5

            epsilon_sum_array = np.array(epsilon_sum_array)
            beta = epsilon_sum_array / (1 - epsilon_sum_array)

            sigma = []
            a = self.a
            b = self.b
            t = len(self.ensemble_)
            k = np.array(range(t))

            sigma = 1/(1 + np.exp(-a*(t-k-b)))

            sigma_mean = []
            for k in range(t):
                sigma_sum = np.sum(sigma[0:t-k])
                sigma_mean.append(sigma[k]/sigma_sum)

            beta_mean = []
            for k in range(t):
                beta_sum = np.sum(sigma_mean[0:t-k]*beta[0:t-k])
                beta_mean.append(beta_sum)

            self.weights_ = []
            for b in beta_mean:
                self.weights_.append(np.log(1/b))

            if t >= self.n_estimators:
                ind = np.argmax(beta_mean)
                del self.ensemble_[ind]
                del self.weights_[ind]

        else:
            res_X, res_y = self._resample(X, y)

            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            self.ensemble_.append(new_classifier)
            self.weights_ = [1]

    def _resample(self, X, y):
        minority, majority = self.minority_majority_split(X, y,
                                                    self.minority_name,
                                                    self.majority_name)
        if len(minority) > 6:
            res_X, res_y = SMOTE().fit_resample(X, y)
        else:
            res_X, res_y = SMOTE(k_neighbors=len(minority)-1).fit_resample(X, y)
        return res_X, res_y
