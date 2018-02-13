from enum import Enum
import numpy as np
import warnings
from sklearn import metrics
warnings.simplefilter('always')

PRUNING_CRITERION = ('accuracy', 'diversity', 'weighted')

class Pruner(object):
    def __init__(self, pruning_criterion='diversity', testing_set_size = 30, pruning_permutations = 5, ensemble_size = 20):
        self.pruning_permutations = pruning_permutations
        self.testing_set_size = testing_set_size
        self.pruning_criterion = pruning_criterion
        self.ensemble_size = ensemble_size
        self.X = None
        self.y = None
        self.ensemble = None
        self.classes = None

    def prepare_testing_set(self, testing_set):
        self.X, self.y = testing_set
        idx = np.random.choice(len(self.y), self.testing_set_size, replace=False)
        self.y = self.y[idx]
        self.X = self.X[idx,:]

    def prune(self, ensemble, testing_set, classes):
        self.ensemble = ensemble
        self.classes = classes
        best_permutation = None
        if len(self.ensemble) > self.ensemble_size:
            if testing_set is not None:
                self.prepare_testing_set(testing_set)
                if self.pruning_criterion == 'diversity':
                    best_permutation = self.diversity()
                elif self.pruning_criterion == 'accuracy':
                    best_permutation = self.accuracy()
                elif self.pruning_criterion == 'weighted':
                    best_permutation = self.diversity()

        return best_permutation

    def accuracy(self):
        """
        Accuracy pruning.
        """
        best_measure = -1
        best_permutation = None
        for i in xrange(self.pruning_permutations):
            permutation =  np.random.permutation(len(self.ensemble))[:self.ensemble_size]

            current_measure = self.score(permutation)
            if current_measure > best_measure:
                best_measure = current_measure
                best_permutation = permutation
        return best_permutation

    def score(self, permutation):
        supports = None
        for cid in permutation:
            member_clf = self.ensemble[cid]
            support = member_clf.predict_proba(self.X)
            if supports is None:
                supports = support
            else:
                supports += support
        decisions = np.argmax(supports, axis=1)
        _y = np.array([self.classes.index(a) for a in self.y])
        accuracy = metrics.accuracy_score(_y, decisions)
        return accuracy

    def diversity(self):
        """
        Optimization based pruning on Partridge Krzanowski metric.
        """
        best_measure = 0
        best_permutation = None
        for i in xrange(self.pruning_permutations):
            permutation =  np.random.permutation(len(self.ensemble))[:self.ensemble_size]
            incorrects, tries = (0,0)
            accuA, accuB = (0,0)
            for l, clf_id in enumerate(permutation):
                clf = self.ensemble[clf_id]
                incorrect = self.testing_set_size - np.sum(clf.predict(self.X) == self.y)

                incorrects += incorrect
                tries += self.ensemble_size

                p = float(incorrects) / float(tries)

                a = ((l + 1) * (l) * p) / (self.ensemble_size * (self.ensemble_size - 1));
                b = (l + 1) * p / self.ensemble_size;

                accuA += a
                accuB += b

            current_measure = 1 - (accuA / accuB)
            if current_measure > best_measure:
                best_measure = current_measure
                best_permutation = permutation
        return best_permutation
