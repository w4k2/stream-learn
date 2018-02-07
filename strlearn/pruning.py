from enum import Enum
import numpy as np

class PruningCriterion(Enum):
    DIVERSITY = 1
    ENSEMBLE_ACCURACY = 2
    WEIGHTED_COMBINATION = 3

class Pruner(object):
    def __init__(self, pruning_criterion, testing_set_size = 30, pruning_permutations = 4, ensemble_size = 20):
        self.pruning_permutations = pruning_permutations
        self.testing_set_size = testing_set_size
        self.pruning_criterion = pruning_criterion
        self.ensemble_size = ensemble_size
        self.X = None
        self.y = None
        self.ensemble = None

    def prepare_testing_set(self, testing_set):
        self.X, self.y = testing_set
        idx = np.random.choice(len(self.y), self.testing_set_size, replace=False)
        self.y = self.y[idx]
        self.X = self.X[idx,:]

    def prune(self, ensemble, testing_set):
        self.ensemble = ensemble
        best_permutation = None
        if len(self.ensemble) > self.ensemble_size:
            if testing_set is not None:
                self.prepare_testing_set(testing_set)
                if self.pruning_criterion is PruningCriterion.DIVERSITY:
                    best_permutation = self.diversity()
                elif self.pruning_criterion == PruningCriterion.ENSEMBLE_ACCURACY:
                    pass
                else:
                    pass

        return best_permutation

    def diversity(self):
        # Partridge Krzanowski
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
