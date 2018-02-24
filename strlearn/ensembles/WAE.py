from sklearn.base import BaseEstimator
from sklearn import neural_network, base
from enum import Enum
from sklearn import neighbors
from sklearn import metrics
import numpy as np
import pruning
import warnings
warnings.simplefilter('always')

WEIGHT_CALCULATION_METHOD = ('same_for_each', 'proportional_to_accuracy','kuncheva','proportional_to_accuracy_related_to_whole_ensemble','proportional_to_accuracy_related_to_whole_ensemble_using_bell_curve')

AGING_METHOD = ('weights_proportional', 'constant', 'gaussian')

class WAE(BaseEstimator):
    """Weighted Aging Ensemble
    lorem ipsum

    References
    ----------
    .. [1] A. Kasprzak, M. Wozniak, "Modifications of the Weighted Aged Ensemble algorithm applied to the data stream classification - experimental analysis of chosen characteristics"
    """

    def __init__(self, base_classifier=neighbors.KNeighborsClassifier(), ensemble_size=20, theta=.1, is_post_pruning=False, pruning_criterion='accuracy', weight_calculation_method='kuncheva', aging_method='weights_proportional', rejuvenation_power=0.):
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
        self.iterations = np.array([])
        self.weights = [1]

        self.ensemble_support_matrix = None

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
        y = self.previous_training_set[1]
        _y = np.array([self.classes.index(a) for a in y])
        pruner = pruning.OneOffPruner(self.ensemble_support_matrix, _y, self.pruning_criterion)
        return pruner.best_permutation

    def _filter_ensemble(self, combination):
        self.ensemble = [self.ensemble[clf_id] for clf_id in combination]
        self.iterations = self.iterations[combination]
        self.weights = self.weights[combination]

    def partial_fit(self, X, y, classes):
        """Partial fitting"""
        if self.age > 0:
            self.overall_accuracy = self.score(
                self.previous_training_set[0],
                self.previous_training_set[1]
            )

        # Pre-pruning
        if len(self.ensemble) > self.ensemble_size and not self.is_post_pruning:
            best_permutation = self._prune()
            self._filter_ensemble(best_permutation)

        # Preparing and training new candidate
        self.classes = classes
        candidate_clf = base.clone(self.base_classifier)
        candidate_clf.fit(X, y)
        self.ensemble.append(candidate_clf)
        self.iterations = np.append(self.iterations, [1])

        self._set_weights()
        self._rejuvenate()
        self._aging()
        self._extinct()

        # Post-pruning
        if len(self.ensemble) > self.ensemble_size and self.is_post_pruning:
            best_permutation = self._prune()
            self._filter_ensemble(best_permutation)

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

            elif self.weight_calculation_method == 'kuncheva':
                accuracies = self._accuracies()
                self.weights = np.log(accuracies / (1.0000001 - accuracies))
                self.weights[self.weights < 0] = 0

            elif self.weight_calculation_method == 'proportional_to_accuracy_related_to_whole_ensemble':
                accuracies = self._accuracies()
                self.weights = accuracies / self.overall_accuracy
                self.weights[self.weights < self.theta] = 0

            elif self.weight_calculation_method == 'proportional_to_accuracy_related_to_whole_ensemble_using_bell_curve':
                accuracies = self._accuracies()
                self.weights = 1./(2. * np.pi) * np.exp((self.overall_accuracy - accuracies)/2.)
                self.weights[self.weights < self.theta] = 0

        self.weights = np.nan_to_num(self.weights)

    def _aging(self):
        if self.age > 0:
            if self.aging_method == 'weights_proportional':
                accuracies = self._accuracies()
                self.weights = accuracies / np.sqrt(self.iterations)

            elif self.aging_method == 'constant':
                self.weights -= self.theta * self.iterations
                self.weights[self.weights < self.theta] = 0

            elif self.aging_method == 'gaussian':
                self.weights = 1. / (2. * np.pi) * \
                    np.exp((self.iterations * self.weights) / 2.)

    def _rejuvenate(self):
        if self.rejuvenation_power > 0 and len(self.weights) > 0:
            w = np.sum(self.weights) / len(self.weights)
            mask = self.weights > w
            self.iterations[mask] -= self.rejuvenation_power * (self.weights[mask] - w)
            #TODO do przemyslenia

    def _extinct(self):
        combination = np.array(np.where(self.weights > 0))[0]
        if len(combination) > 0:
            self._filter_ensemble(combination)

    def predict_proba(self, X):
        """Predict proba"""
        # Establish support matrix for ensemble
        self.ensemble_support_matrix = np.array([member_clf.predict_proba(X) for member_clf in self.ensemble])

        # Weight support before acumulation
        weighted_support = self.ensemble_support_matrix * self.weights[:, np.newaxis, np.newaxis]

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        return acumulated_weighted_support

    def score(self, X, y):
        """Accuracy score"""
        #print "Scoring, ESM:"
        #print self.ensemble_support_matrix
        supports = self.predict_proba(X)
        decisions = np.argmax(supports, axis=1)
        _y = np.array([self.classes.index(a) for a in y])
        accuracy = metrics.accuracy_score(_y, decisions)
        return accuracy
