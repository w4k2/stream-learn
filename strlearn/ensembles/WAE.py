"""
Weighted Aging Ensemble.

"""

from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn import neighbors
import numpy as np
from ..ensembles import pruning

WEIGHT_CALCULATION = (
    "same_for_each",
    "proportional_to_accuracy",
    "kuncheva",
    "pta_related_to_whole",
    "bell_curve",
)

AGING_METHOD = ("weights_proportional", "constant", "gaussian")


class WAE(BaseEnsemble, ClassifierMixin):
    """
    Weighted Aging Ensemble.
    """
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        theta=0.1,
        post_pruning=False,
        pruning_criterion="accuracy",
        weight_calculation_method="kuncheva",
        aging_method="weights_proportional",
        rejuvenation_power=0.0,
    ):
        """Initialization."""
        self.base_estimator = base_estimator
        self.pruning_criterion = pruning_criterion
        self.n_estimators = n_estimators
        self.theta = theta
        self.post_pruning = post_pruning
        self.weight_calculation_method = weight_calculation_method
        self.aging_method = aging_method
        self.rejuvenation_power = rejuvenation_power

    def _prune(self):
        X, y = self.previous_X, self.previous_y
        pruner = pruning.OneOffPruner(
            self.ensemble_support_matrix(X), y, self.pruning_criterion
        )
        self._filter_ensemble(pruner.best_permutation)

    def _filter_ensemble(self, combination):
        self.ensemble_ = [self.ensemble_[clf_id] for clf_id in combination]
        self.iterations_ = self.iterations_[combination]
        self.weights_ = self.weights_[combination]

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        candidate_clf = base.clone(self.base_estimator)
        candidate_clf.fit(X, y)

        self.ensemble_ = [candidate_clf]
        self.weights_ = np.array([1])
        self.classes_, _ = np.unique(y, return_inverse=True)
        self.age_ = 0
        self.iterations_ = np.array([])

        # Return the classifier
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes

            self.ensemble_ = []
            self.weights_ = np.array([1])
            self.age_ = 0
            self.iterations_ = np.array([])

        """Partial fitting"""
        if self.age_ > 0:
            self.overall_accuracy = self.score(self.previous_X, self.previous_y)

        # Pre-pruning
        if len(self.ensemble_) > self.n_estimators and not self.post_pruning:
            self._prune()

        # Preparing and training new candidate
        self.classes_ = classes
        candidate_clf = base.clone(self.base_estimator)
        candidate_clf.fit(X, y)
        self.ensemble_.append(candidate_clf)
        self.iterations_ = np.append(self.iterations_, [1])

        self._set_weights()
        self._rejuvenate()
        self._aging()
        self._extinct()

        # Post-pruning
        if len(self.ensemble_) > self.n_estimators and self.post_pruning:
            self._prune()

        # Weights normalization
        self.weights_ = self.weights_ / np.sum(self.weights_)

        # Ending procedure
        self.previous_X, self.previous_y = (X, y)
        self.age_ += 1
        self.iterations_ += 1

    def _accuracies(self):
        return np.array(
            [m_clf.score(self.previous_X, self.previous_y) for m_clf in self.ensemble_]
        )

    def _set_weights(self):
        if self.age_ > 0:
            if self.weight_calculation_method == "same_for_each":
                self.weights_ = np.ones(len(self.ensemble_))

            elif self.weight_calculation_method == "kuncheva":
                accuracies = self._accuracies()
                self.weights_ = np.log(accuracies / (1.0000001 - accuracies))
                self.weights_[self.weights_ < 0] = 0

            elif self.weight_calculation_method == "pta_related_to_whole":
                accuracies = self._accuracies()
                self.weights_ = accuracies / self.overall_accuracy
                self.weights_[self.weights_ < self.theta] = 0

            elif self.weight_calculation_method == "bell_curve":
                accuracies = self._accuracies()
                self.weights_ = (
                    1.0
                    / (2.0 * np.pi)
                    * np.exp((self.overall_accuracy - accuracies) / 2.0)
                )
                self.weights_[self.weights_ < self.theta] = 0

        self.weights_ = np.nan_to_num(self.weights_)

    def _aging(self):
        if self.age_ > 0:
            if self.aging_method == "weights_proportional":
                accuracies = self._accuracies()
                self.weights_ = accuracies / np.sqrt(self.iterations_)

            elif self.aging_method == "constant":
                self.weights_ -= self.theta * self.iterations_
                self.weights_[self.weights_ < self.theta] = 0

            elif self.aging_method == "gaussian":
                self.weights_ = (
                    1.0
                    / (2.0 * np.pi)
                    * np.exp((self.iterations_ * self.weights_) / 2.0)
                )

    def _rejuvenate(self):
        if self.rejuvenation_power > 0 and len(self.weights_) > 0:
            w = np.sum(self.weights_) / len(self.weights_)
            mask = self.weights_ > w
            self.iterations_[mask] -= self.rejuvenation_power * (
                self.weights_[mask] - w
            )
            # TODO do przemyslenia

    def _extinct(self):
        combination = np.array(np.where(self.weights_ > 0))[0]
        if len(combination) > 0:
            self._filter_ensemble(combination)

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Weight support before acumulation
        weighted_support = (
            self.ensemble_support_matrix(X) * self.weights_[:, np.newaxis, np.newaxis]
        )

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        return acumulated_weighted_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]
