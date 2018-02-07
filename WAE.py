from sklearn.base import BaseEstimator
from sklearn import neural_network, base
from enum import Enum
from sklearn import neighbors
from sklearn import metrics
import numpy as np
import pruning
import line_profiler
reload(pruning)


class WeightCalculationMethod(Enum):
    SAME_FOR_EACH = 1
    PROPORTIONAL_TO_ACCURACY = 2
    AGED_PROPORTIONAL_TO_ACCURACY = 3
    KUNCHEVA = 4
    PROPORTIONAL_TO_ACCURACY_RELATED_TO_WHOLE_ENSEMBLE = 5
    PROPORTIONAL_TO_ACCURACY_RELATED_TO_WHOLE_ENSEMBLE_USING_BELL_CURVE = 6


class AgingMethod(Enum):
    WEIGHTS_PROPORTIONAL = 1
    CONSTANT = 2
    GAUSSIAN = 3


class WAE(BaseEstimator):
    '''
    Modified Weighted Aged Ensemble applied to the data stream classification as proposed by M. Wozniak, 2014
    '''

    def __init__(self, base_classifier=neighbors.KNeighborsClassifier(), ensemble_size=20, theta=.1, is_post_pruning=False, pruning_criterion=pruning.PruningCriterion.DIVERSITY, weight_calculation_method=WeightCalculationMethod.KUNCHEVA, aging_method=AgingMethod.WEIGHTS_PROPORTIONAL, is_rejuvenating=False, rejuvenation_power=.5):
        self.pruning_criterion = pruning_criterion
        self.ensemble_size = ensemble_size
        self.pruner = pruning.Pruner(
            self.pruning_criterion, ensemble_size=self.ensemble_size)
        self.base_classifier = base_classifier
        self.theta = theta
        self.is_post_pruning = is_post_pruning
        self.weight_calculation_method = weight_calculation_method
        self.aging_method = aging_method
        self.is_rejuvenating = is_rejuvenating
        self.rejuvenation_power = rejuvenation_power

        self.age = 0
        self.overall_accuracy = 0
        self.ensemble = []

        self.iterations = []    # wiek w iteracjach
        self.ages = []          # wiek wedlug miary
        self.accuracies = []
        self.weights = []

        self.previous_training_set = None
        self.classes = None

    def prune(self):
        # TODO: Poprawienie tablic z wiekiem
        best_permutation = self.pruner.prune(
            self.ensemble, self.previous_training_set)
        self.ensemble = [self.ensemble[clf_id] for clf_id in best_permutation]
        self.iterations = [self.iterations[clf_id]
                           for clf_id in best_permutation]
        return best_permutation

    def partial_fit(self, X, y, classes):
        if self.age > 0:
            self.overall_accuracy = self.score(
                self.previous_training_set[0], self.previous_training_set[1])
        #print "\nPARTIAL FIT --- OA = %.3f" % self.overall_accuracy
        # Preparing and training new candidate
        self.classes = classes
        candidate_clf = base.clone(self.base_classifier)
        candidate_clf.fit(X, y)
        self.ensemble.append(candidate_clf)
        self.iterations.append(0)

        # Pre-pruning
        if len(self.ensemble) > self.ensemble_size and not self.is_post_pruning:
            self.prune()

        self.set_accuracies()
        self.set_ages()
        self.set_weights()

        # Zabijamy wszystkie o ujemnej wadze
        self.extinction()

        # Wagi musza sumowac sie do jedynki, ale dopiero po wymieraniu
        self.weights /= np.sum(self.weights)

        # Post-pruning
        if len(self.ensemble) > self.ensemble_size and self.is_post_pruning:
            best_permutation = self.prune()
            self.ages = [self.ages[clf_id] for clf_id in best_permutation]
            self.accuracies = [self.accuracies[clf_id] for clf_id in best_permutation]
            self.weights = [self.weights[clf_id] for clf_id in best_permutation]

        # Ending procedure
        self.previous_training_set = (X, y)
        self.age += 1

    def set_accuracies(self):
        if self.age > 0:
            X, y = self.previous_training_set
            self.accuracies = np.array(
                [m_clf.score(X, y) for m_clf in self.ensemble])

    def set_weights(self):
        if self.age > 0:
            if self.weight_calculation_method == WeightCalculationMethod.SAME_FOR_EACH:
                self.weights = np.ones(len(self.ensemble))
            elif self.weight_calculation_method == WeightCalculationMethod.PROPORTIONAL_TO_ACCURACY:
                self.weights = np.copy(self.accuracies)
            elif self.weight_calculation_method == WeightCalculationMethod.AGED_PROPORTIONAL_TO_ACCURACY:
                self.weights = self.accuracies / np.sqrt(self.iterations)
            elif self.weight_calculation_method == WeightCalculationMethod.KUNCHEVA:
                self.weights = self.accuracies / (1 - self.accuracies)
            elif self.weight_calculation_method == WeightCalculationMethod.PROPORTIONAL_TO_ACCURACY_RELATED_TO_WHOLE_ENSEMBLE:
                self.weights = self.accuracies / self.overall_accuracy
            elif self.weight_calculation_method == WeightCalculationMethod.PROPORTIONAL_TO_ACCURACY_RELATED_TO_WHOLE_ENSEMBLE_USING_BELL_CURVE:
                self.weights = 1./(2. * np.pi) * np.exp((self.overall_accuracy - self.accuracies)/2.)
        self.weights = np.nan_to_num(self.weights)

    def set_ages(self):
        self.iterations = np.array(self.iterations) + 1
        if self.age > 0:
            if self.aging_method == AgingMethod.WEIGHTS_PROPORTIONAL:
                self.ages = self.accuracies / np.sqrt(self.iterations)
            elif self.aging_method == AgingMethod.CONSTANT:
                self.ages = 1 - (self.iterations * self.theta)
            elif self.aging_method == AgingMethod.GAUSSIAN:
                self.ages = 1. / (2. * np.pi) * \
                    np.exp((self.iterations * self.theta) / 2.)

        self.iterations = list(self.iterations)

    def extinction(self):
        #TODO: Tu nalezy faktycznie usunac martwe
        still_alive = self.ages > 0

    def predict_proba(self, X):
        supports = None
        for cid, member_clf in enumerate(self.ensemble):
            weight = 1.
            if self.age > 1:
                weight = self.weights[cid]
            support = member_clf.predict_proba(X) * weight

            if supports is None:
                supports = support
            else:
                supports += support

        return supports

    def score(self, X, y):
        supports = self.predict_proba(X)
        decisions = np.argmax(supports, axis=1)
        _y = np.array([self.classes.index(a) for a in y])
        accuracy = metrics.accuracy_score(_y, decisions)
        return accuracy
