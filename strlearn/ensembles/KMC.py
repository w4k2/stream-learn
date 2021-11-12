from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from ..ensembles.base import StreamingEnsemble
import numpy as np

class KMC(StreamingEnsemble):
    """
    KMC

    Wang, Yi, Yang Zhang, and Yong Wang. "Mining data streams
           with skewed distribution by static classifier ensemble."
           Opportunities and Challenges for Next-Generation Applied
           Intelligence. Springer, Berlin, Heidelberg, 2009. 65-71.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10):
        super().__init__(base_estimator, n_estimators, weighted=True)

    def partial_fit(self, X, y, classes=None):
        super().partial_fit(X, y, classes)
        if not self.green_light:
            return self

        if len(self.ensemble_) == 0:
            self.weights_ = []

        # Find minority and majority names
        if not hasattr(self, "minority_name") or not hasattr(self, "majority_name"):
            self.minority_name, self.majority_name = self.minority_majority_name(y)

        # Resample data
        res_X, res_y = self._resample(X, y)

        # Train new model
        new_classifier = clone(self.base_estimator).fit(res_X, res_y)

        if len(self.ensemble_) < self.n_estimators:
            # Append new estimator
            self.ensemble_.append(new_classifier)
            self.weights_.append(1)
        else:
            # Remove the worst model when ensemble becomes too large
            auc_array = []

            for i in range(len(self.ensemble_)):
                y_score = self.ensemble_[i].predict_proba(res_X)
                auc_array.append(roc_auc_score(res_y, y_score[:, 1]))

            j = np.argmin(auc_array)

            y_score = new_classifier.predict_proba(res_X)
            new_auc = roc_auc_score(res_y, y_score[:, 1])

            if new_auc > auc_array[j]:
                self.ensemble_[j] = new_classifier
                auc_array[j] = new_auc

            for i in range(len(self.ensemble_)):
                self.weights_[i] = auc_array[i]

    def _resample(self, X, y):
        minority, majority = self.minority_majority_split(X, y,
                                                     self.minority_name,
                                                     self.majority_name)

        # Undersample majority array
        km = KMeans(n_clusters=len(minority)).fit(X)
        majority = km.cluster_centers_

        res_X = np.concatenate((majority, minority), axis=0)
        res_y = len(majority)*[self.majority_name] + len(minority)*[self.minority_name]

        return res_X, res_y
