from sklearn.base import clone
from sklearn.metrics import f1_score
from ..ensembles.base import StreamingEnsemble
import numpy as np

class NIE(StreamingEnsemble):
    """
    NIE

    Ditzler, Gregory, and Robi Polikar. "Incremental learning of concept drift from streaming imbalanced data." IEEE Transactions on Knowledge and Data Engineering 25.10 (2013): 2283-2301.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=5,
                 param_a=1,
                 param_b=1):
        super().__init__(base_estimator, n_estimators, weighted=True)
        self.param_a = param_a
        self.param_b = param_b

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        self.ensemble_ += [self._new_sub_ensemble(X, y)]

        beta_mean = self._calculate_weights(X, y)

        self.weights_ = []
        for b in beta_mean:
            self.weights_.append(np.log(1/b))

        if len(self.ensemble_) >= self.n_estimators:
            ind = np.argmax(beta_mean)
            del self.ensemble_[ind]
            del self.weights_[ind]

    def _calculate_weights(self, X, y):
        beta = []
        for i in range(len(self.ensemble_)):
            epsilon = 1-f1_score(y, self._sub_ensemble_predict(i, X))
            if epsilon > 0.5:
                if i == len(self.ensemble_) - 1:
                    self.ensemble_[i] = self._new_sub_ensemble(X, y)
                    epsilon = 0.5
                else:
                    epsilon = 0.5
            beta.append(epsilon / float(1 - epsilon))

        sigma = []
        a = self.param_a
        b = self.param_b
        t = len(self.ensemble_)
        k = np.array(range(t))

        sigma = 1/(1 + np.exp(-a*(t-k-b)))

        sigma_mean = []
        for k in range(t):
            sigma_sum = 0
            for j in range(t-k):
                sigma_sum += sigma[j]
            sigma_mean.append(sigma[k]/sigma_sum)

        beta_mean = []
        for k in range(t):
            beta_sum = 0
            for j in range(t-k):
                beta_sum += sigma_mean[j]*beta[j]
            beta_mean.append(beta_sum)

        return beta_mean

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([self._sub_ensemble_predict_proba(idx, X) for idx, member_clf in enumerate(self.ensemble_)])

    def _new_sub_ensemble(self, X, y):
        y = np.array(y)
        X = np.array(X)

        minority, majority = self.minority_majority_split(X, y,
                                                     self.minority_name,
                                                     self.majority_name)

        T = self.n_estimators
        N = len(X)
        sub_ensemble = []
        for k in range(T):
            number_of_instances = int(np.floor(N/float(T)))
            idx = np.random.randint(len(majority), size=number_of_instances)
            sample = majority[idx, :]
            res_X = np.concatenate((sample, minority), axis=0)
            res_y = len(sample)*[self.majority_name] + len(minority)*[self.minority_name]
            new_classifier = clone(self.base_estimator).fit(res_X, res_y)
            sub_ensemble += [new_classifier]
        return sub_ensemble

    def _sub_ensemble_predict_proba(self, i, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble_[i]]
        return np.average(probas_, axis=0)

    def _sub_ensemble_predict(self, i, X):
        predictions = np.asarray([clf.predict(X) for clf in self.ensemble_[i]]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                  axis=1,
                                  arr=predictions)
        return maj
