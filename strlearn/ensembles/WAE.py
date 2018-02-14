from sklearn.base import BaseEstimator
from sklearn import neural_network, base
from enum import Enum
from sklearn import neighbors
from sklearn import metrics
import numpy as np
import pruning
import warnings
warnings.simplefilter('always')

WEIGHT_CALCULATION_METHOD = ('same_for_each', 'proportional_to_accuracy','aged_proportional_to_accuracy','kuncheva','proportional_to_accuracy_related_to_whole_ensemble','proportional_to_accuracy_related_to_whole_ensemble_using_bell_curve')

AGING_METHOD = ('weights_proportional', 'constant', 'gaussian')

class WAE(BaseEstimator):
    """Weighted Aging Ensemble
    lorem ipsum

    References
    ----------
    .. [1] A. Kasprzeak, M. Wozniak, "Modifications of the Weighted Aged Ensemble algorithm applied to the data stream classification - experimental analysis of chosen characteristics"
    """

    def __init__(self, base_classifier=neighbors.KNeighborsClassifier(), ensemble_size=20, theta=.1, is_post_pruning=False, pruning_criterion='diversity', weight_calculation_method='kuncheva', aging_method='weights_proportional', rejuvenation_power=0.):
        self.pruning_criterion = pruning_criterion
        self.ensemble_size = ensemble_size
        self.base_classifier = base_classifier
        self.theta = theta
        self.is_post_pruning = is_post_pruning
        self.weight_calculation_method = weight_calculation_method
        self.aging_method = aging_method
        self.rejuvenation_power = rejuvenation_power

        self.age = 0
        self.overall_accuracy = 0

        # Ensemble and its parameters
        self.ensemble = []
        self.iterations = np.array([])    # wiek w iteracjach
        self.ages = []          # wiek wedlug miary
        self.weights = []

        self.previous_training_set = None
        self.classes = None

    def __str__(self):
        return "WAE_wcm_%i_am_%i_j_%i_jp_%s_t_%s_pp_%i_n_%i_pc_%i" % (
            self.weight_calculation_method,
            self.aging_method,
            self.is_rejuvenating,
            ("%.3f" % self.rejuvenation_power)[2:],
            ("%.3f" % self.theta)[2:],
            self.is_post_pruning,
            self.ensemble_size,
            self.pruning_criterion.value
        )

    def _prune(self):
        #warnings.warn("I want to prune")
        pruner = pruning.Pruner(self.pruning_criterion, ensemble_size=self.ensemble_size)
        return pruner.prune(self.ensemble, self.previous_training_set,classes=self.classes)

    def _filter_ensemble(self, combination):
        self.ensemble = [self.ensemble[clf_id] for clf_id in combination]
        self.iterations = self.iterations[combination]
        self.ages = self.ages[combination]
        self.weights = self.weights[combination]

    def partial_fit(self, X, y, classes):
        """Partial fitting"""
        #warnings.warn("Partial fit %i [size %i]" % (self.age, len(self.ensemble)))
        if self.age > 0:
            self.overall_accuracy = self.score(
                self.previous_training_set[0],
                self.previous_training_set[1]
            )

        # Pre-pruning
        if len(self.ensemble) > self.ensemble_size and not self.is_post_pruning:
            self._rejuvenate()
            best_permutation = self._prune()

        # Preparing and training new candidate
        self.classes = classes
        candidate_clf = base.clone(self.base_classifier)
        candidate_clf.fit(X, y)
        self.ensemble.append(candidate_clf)
        self.iterations = np.append(self.iterations, [1])

        best_permutation = None

        self._set_weights()
        self._set_ages()

        if best_permutation is not None:
            self._filter_ensemble(best_permutation)

        self._extinct()

        # Post-pruning
        if len(self.ensemble) > self.ensemble_size and self.is_post_pruning:
            self._rejuvenate()
            best_permutation = self._prune()
            self._filter_ensemble(best_permutation)

        #TODO Connect weights and ages?
        self.weights *= self.ages

        # Weights normalization
        self.weights /= np.sum(self.weights)

        # Ending procedure
        self.previous_training_set = (X, y)
        self.age += 1
        self.iterations += 1

    def _accuracies(self):
        X, y = self.previous_training_set
        return np.array(
            [m_clf.score(X, y) for m_clf in self.ensemble])

    def _set_weights(self):
        if self.age > 0:
            if self.weight_calculation_method == 'same_for_each':
                self.weights = np.ones(len(self.ensemble))

            elif self.weight_calculation_method == 'proportional_to_accuracy':
                self.weights = self._accuracies()

            elif self.weight_calculation_method == 'aged_proportional_to_accuracy':
                accuracies = self._accuracies()
                self.weights = accuracies / np.sqrt(self.iterations)

            elif self.weight_calculation_method == 'kuncheva':
                accuracies = self._accuracies()
                self.weights = accuracies / (1.0000001 - accuracies)

            elif self.weight_calculation_method == 'proportional_to_accuracy_related_to_whole_ensemble':
                accuracies = self._accuracies()
                self.weights = accuracies / self.overall_accuracy
                self.weights[self.weights < self.theta] = 0

            elif self.weight_calculation_method == 'proportional_to_accuracy_related_to_whole_ensemble_using_bell_curve':
                accuracies = self._accuracies()
                self.weights = 1./(2. * np.pi) * np.exp((self.overall_accuracy - accuracies)/2.)
                self.weights[self.weights < self.theta] = 0

        self.weights = np.nan_to_num(self.weights)

    def _set_ages(self):
        if self.age > 0:
            if self.aging_method == 'weights_proportional':
                accuracies = self._accuracies()
                self.ages = accuracies / np.sqrt(self.iterations)

            elif self.aging_method == 'constant':
                self.ages = self.weights - self.theta
                self.ages[self.ages < self.theta] = 0

            elif self.aging_method == 'gaussian':
                self.ages = 1. / (2. * np.pi) * \
                    np.exp((self.iterations * self.theta) / 2.)

    def _rejuvenate(self):
        if self.rejuvenation_power > 0:
            w = np.sum(self.weights) / len(self.weights)
            mask = self.weights > w
            self.iterations[mask] -= self.rejuvenation_power * (self.weights[mask] - w)

    def _extinct(self):
        combination = np.array(np.where(self.weights > 0))[0]
        if len(combination) > 0:
            self._filter_ensemble(combination)

    def predict_proba(self, X):
        """Predict proba"""
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
        """Accuracy score"""
        supports = self.predict_proba(X)
        decisions = np.argmax(supports, axis=1)
        _y = np.array([self.classes.index(a) for a in y])
        accuracy = metrics.accuracy_score(_y, decisions)
        #warnings.warn("Scored %.3f" % accuracy)
        return accuracy
