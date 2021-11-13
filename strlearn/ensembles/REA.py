import numpy as np
from sklearn.base import clone
from ..ensembles.base import StreamingEnsemble
from sklearn.neighbors import NearestNeighbors


class REA(StreamingEnsemble):
    """
    Recursive Ensemble Approach.

    Sheng Chen, and Haibo He. "Towards incremental learning of nonstationary imbalanced data stream: a multiple selectively recursive approach." Evolving Systems 2.1 (2011): 35-50.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 post_balance_ratio=0.5,
                 k_parameter=10,
                 weighted=False,
                 pruning=False):
        super().__init__(base_estimator, n_estimators, weighted=weighted)
        self.post_balance_ratio = post_balance_ratio
        self.k_parameter = k_parameter
        self.pruning = pruning

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        # Resample data
        res_X, res_y = self._resample(X, y)

        # Append new estimator
        self.ensemble_.append(clone(self.base_estimator).fit(res_X, res_y))

        # # Remove the worst model when ensemble becomes too large
        if self.pruning:
            if len(self.ensemble_) >= self.n_estimators:
                worst = np.argmin(self.weights_)
                del self.ensemble_[worst]
                np.delete(self.weights_, worst)

        # Recalculate the ensemble weights
        weights = []
        for clf in self.ensemble_:
            y_pred = clf.predict(X).astype(int)
            probas_ = clf.predict_proba(X)[np.arange(len(X)), y_pred]
            weights.append(np.log(1/(np.sum((1-probas_)**2)/len(X))))
        self.weights_ = np.array(weights)

        return self

    def _resample(self, X, y):
        # Split the data
        minority, majority = self.minority_majority_split(X, y, self.minority_name, self.majority_name)

        # Check if the minority data has been previously accumulated
        if not hasattr(self, "minority_data_"):
            self.minority_data_ = minority
            return X, y

        # Calculate actual imbalanced ratio
        imbalanced_ratio = len(minority[:, 0])/float(len(X[:, 0]))

        # Resample data
        if self.post_balance_ratio > imbalanced_ratio:

            knn = NearestNeighbors(n_neighbors=self.k_parameter).fit(X)

            indices = knn.kneighbors(self.minority_data_, return_distance=False)

            min_count = np.count_nonzero(y[indices] == self.minority_name, axis=1)

            a = np.arange(0, len(min_count))
            min_count = np.insert(np.expand_dims(min_count, axis=1), 1, a, axis=1)
            min_count = min_count[min_count[:, 0].argsort()]
            min_count = min_count[::-1]

            sorted_minority = min_count[:, 1].astype("int")

            n_instances = int((self.post_balance_ratio - imbalanced_ratio)*len(y))
            n_instances = min(n_instances, len(sorted_minority))
            new_minority = self.minority_data_[sorted_minority[0:n_instances]]

            res_X = np.concatenate((new_minority, majority), axis=0)
            res_y = np.concatenate((np.full(len(new_minority), self.minority_name), np.full(len(majority), self.majority_name)), axis=0)

        else:
            res_X = X
            res_y = y

        self.minority_data_ = np.concatenate((minority, self.minority_data_), axis=0)

        return res_X, res_y
