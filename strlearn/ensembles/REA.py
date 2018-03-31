from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
from strlearn.utils import minority_majority_name, minority_majority_split
import math


class REA(BaseEstimator):
    """Recursive Ensemble Approach for Nonstationary Imbalanced Data Stream [1]

    Parameters
    ----------
    base_classifier : object, optional (default = SVC(probability = True))
        The base classifier used to create ensemble.

    n_classifiers : int, optional (default=10)
        The number of base classifier in the ensemble.

    balance_ratio : float, optional (default = 0.5)
        Data balance ratio after resampling.

    References
    ----------
    .. [1] Sheng Chen, and Haibo He. "Towards incremental learning of
           nonstationary imbalanced data stream: a multiple selectively
           recursive approach." Evolving Systems 2.1 (2011): 35-50.
    """

    def __init__(self, base_classifier=SVC(probability=True), n_classifiers=10,
                 balance_ratio=0.5):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.balance_ratio = balance_ratio

        self.classifier_array = []
        self.classifier_weights = []
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.minority_data = None
        self.label_encoder = None
        self.iterator = 1

    def partial_fit(self, X, y, classes=None):
        """Incremental fit on a batch of samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        classes : array-like, shape (n_classes,), optional (default=None)
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.
        """
        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)

        res_X, res_y = self._resample(X, y)

        new_classifier = self.base_classifier.fit(res_X, res_y)

        self.classifier_array.append(new_classifier)

        s1 = 1/float(len(X))
        weights = []
        for clf in self.classifier_array:
            proba = clf.predict_proba(X)
            s2 = 0
            for i, x in enumerate(X):
                probas = proba[i][y[i]]
                s2 += math.pow((1 - probas), 2)
            s3 = math.log(1/float(s1*s2))
            weights.append(s3)

        self.classifier_weights = weights

    def _resample(self, X, y):
        """Resampling method for balancing data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Imbalanced vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        ----------
        res_X : array-like, shape (n_samples, n_features)
            Resampled vectors, where n_samples is the number of samples and
            n_features is the number of features.
        res_y : array-like, shape (n_samples,)
            Target values.
        """
        y = np.array(y)
        X = np.array(X)

        minority, majority = minority_majority_split(X, y, self.minority_name, self.majority_name)

        if self.minority_data is None:
            self.minority_data = minority
            self.iterator += 1
            return X, y

        ratio = len(minority[:, 0])/float(len(X[:, 0]))

        if self.balance_ratio > ratio:
            if ((len(minority) + len(self.minority_data)) / float(len(X) + len(self.minority_data))) <= self.balance_ratio:
                new_minority = np.concatenate(
                    (minority, self.minority_data),
                    axis=0)

            else:
                knn = NearestNeighbors(n_neighbors=3).fit(X, y)
                distance, indicies = knn.kneighbors(self.minority_data)
                a = np.arange(0, len(distance))
                distance = np.insert(distance, -1, a, axis=1)
                distance = distance[distance[:, 0].argsort()]
                new_minority = minority
                for i in range(int(len(X) * 2 * (self.balance_ratio - ratio))):
                    # print i
                    new_minority = np.insert(new_minority, -1,
                                             self.minority_data[int(distance[i][1])], axis=0)

            res_X = np.concatenate((new_minority, majority), axis=0)
            res_y = np.concatenate((np.full(len(new_minority), self.minority_name), np.full(len(majority), self.majority_name)), axis=0)

        else:
            res_X = X
            res_y = y

        self.minority_data = np.concatenate((minority, self.minority_data),
                                            axis=0)
        self.iterator += 1

        return res_X, res_y

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        C : array-like, shape (n_samples,)
            Predicted target values for X
        """
        predictions = np.asarray([clf.predict(X) for clf in self.classifier_array]).T
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.classifier_weights)), axis=1, arr=predictions)
        maj = self.label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        C : array-like, shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes`.
        """
        probas_ = [clf.predict_proba(X) for clf in self.classifier_array]
        return np.average(probas_, axis=0, weights=self.classifier_weights)

    def score(self, X, y):
        """Return accuracy scores for the test vector X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        accuracy : array-like, shape (n_samples,)
            Predicted target values for X
        """
        supports = self.predict_proba(X)
        decisions = np.argmax(supports, axis=1)
        accuracy = metrics.accuracy_score(y, self.classes[decisions])
        return accuracy
