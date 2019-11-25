from copy import deepcopy
from typing import Optional, List, Union

import numpy as np
from attr import attrib, attrs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import check_X_y, check_array

from strlearn.base import ClassifierSupplier
from strlearn.base.exceptions import BaseClassifierDoesNotSupportPartialFitting
from strlearn.base.types import Classifier
from strlearn.ensembles.voting.SupportsExtractor import SupportsExtractor
from strlearn.ensembles.voting.WeightedMajorityPredictionCombiner import WeightedMajorityPredictionCombiner


@attrs(auto_attribs=True)
class OALE(BaseEstimator, ClassifierMixin):
    _classifier_supplier: ClassifierSupplier = ClassifierSupplier(classifier=GaussianNB())

    _block_size: int = 30  # I
    _dynamic_clfs_limit: int = 10  # D

    _initial_selection_ratio: float = 0.10  # r
    _threshold_adjustment_step: float = 0.05  # s
    _margin_threshold: float = 0.3  # theta
    _random_strategy_threshold: float = 0.05  # sigma

    _classes: List = attrib(factory=list, init=False)
    _cache: List = attrib(factory=list, init=False)  # of tuples (x,y)

    _processed_instances: int = attrib(default=0, init=False)
    _theta_m: float = attrib(default=0, init=False)

    _stable_clf: Optional[Classifier] = attrib(default=None, init=False)
    _stable_clf_weight: float = attrib(init=False, default=0.5)
    _dynamic_clf_weights: List[float] = attrib(factory=list, init=False)
    _dynamic_clfs: List[Classifier] = attrib(factory=list, init=False)

    def partial_fit(self, x, y, classes=None):
        x, y = check_X_y(x, y)

        self._classes = classes
        if self._classes is None:
            self._classes, _ = np.unique(y, return_inverse=True)

        for x_single, y_single in zip(x, y):
            self._partial_fit(x_single, y_single)

    def _partial_fit(self, x_new, y_new):

        new_instance = (x_new, y_new)
        self._processed_instances += 1

        if self._processed_instances < self._block_size:  # fill the circular array for the first time
            self._cache.append(new_instance)

        elif self._processed_instances == self._block_size:  # the first fill of the circular array
            self._cache.append(new_instance)
            new_clf = self._create_new_classifier()
            self._stable_clf = new_clf  # create C_s
            self._add_new_dynamic_clf(deepcopy(new_clf))

        else:  # more instances processed than block size
            i = (self._processed_instances - 1) % self._block_size  # i is the current index for a

            self._deal_instance(new_instance, i)

            i = (i + 1) % self._block_size  # i moves circularly
            if i == 0:  # new instances fill A again
                self._add_new_dynamic_clf(self._create_new_classifier())

                self._theta_m = self._margin_threshold * 2 / len(self._classes)  # reset theta_m for UncertaintyStrategy

                if len(self._dynamic_clfs) > self._dynamic_clfs_limit:
                    self._delete_oldest_dynamic_classifier()

                self._update_weights()

        return self

    def _add_new_dynamic_clf(self, clf):
        self._dynamic_clfs = np.append(self._dynamic_clfs, clf)
        self._dynamic_clf_weights = np.append(self._dynamic_clf_weights, 1 / self._dynamic_clfs_limit)

    def _delete_oldest_dynamic_classifier(self):
        self._dynamic_clfs = np.delete(self._dynamic_clfs, 0)
        self._dynamic_clf_weights = np.delete(self._dynamic_clf_weights, 0)

    def predict(self, x):
        x = check_array(x)

        ensemble, weights = self._get_compact_ensemble_with_weights()

        return WeightedMajorityPredictionCombiner(
            ensemble=ensemble,
            weights=weights,
            classes=self._classes) \
            .predict(x)

    def predict_proba(self, x):
        x = check_array(x)

        ensemble, weights = self._get_compact_ensemble_with_weights()

        return WeightedMajorityPredictionCombiner(
            ensemble=ensemble,
            weights=weights,
            classes=self._classes) \
            .predict_proba(x)

    def _update_classifier(self, x, y, clf):
        def is_collection(arr):
            return isinstance(arr, Union[np.ndarray, List].__args__)

        y = np.array([y]) if not is_collection(y) else y
        x = np.array([x]) if not is_collection(x) else x  # TODO(bgulowaty): extract this
        x = np.array([x]) if x.ndim == 1 else x

        try:
            clf.partial_fit(x, y, self._classes)
        except Exception as e:
            raise BaseClassifierDoesNotSupportPartialFitting(e)

    def _random_strategy(self):
        if self._random_strategy_threshold <= np.random.uniform():
            return True
        return False

    def _uncertainty_strategy(self, x):
        margin = self._calculate_margin(x)

        if margin < self._theta_m:
            self._theta_m = self._theta_m * (1 - self._threshold_adjustment_step)
            return True

        return False

    def _get_compact_ensemble_with_weights(self):
        return (
            np.concatenate(([self._stable_clf], self._dynamic_clfs)),
            np.concatenate(([self._stable_clf_weight], self._dynamic_clf_weights))
        )

    def _calculate_margin(self, x):
        ensemble, weights = self._get_compact_ensemble_with_weights()

        supports = SupportsExtractor(
            ensemble=ensemble,
            weights=weights,
            classes=self._classes) \
            .extract([x])[0]

        max_to_min_indices = np.argsort(supports)[::-1]

        return supports[max_to_min_indices[0]] - supports[max_to_min_indices[1]]

    def _get_randomly_chosen_instances_to_label(self):
        instances_to_label_count = int(self._initial_selection_ratio * len(self._cache))
        random_idxs = np.random.choice(len(self._cache), instances_to_label_count, replace=False)

        return np.take(self._cache, random_idxs, axis=0)

    def _create_new_classifier(self):
        instances = self._get_randomly_chosen_instances_to_label()
        x = np.stack(instances[:, 0])  # TODO(bgulowaty): make this more elegant
        y = np.stack(instances[:, 1])

        if self._stable_clf is not None:
            self._update_stable_classifier(x, y)

        return self._classifier_supplier.get(x, y, self._classes)

    def _update_weights(self):
        self._dynamic_clf_weights = [current_weight * (1 - 1 / self._dynamic_clfs_limit)
                                     for current_weight in self._dynamic_clf_weights]
        self._dynamic_clf_weights[-1] = 1 / self._dynamic_clfs_limit

    def _update_stable_classifier(self, x, y):
        self._update_classifier(x, y, self._stable_clf)

    def _update_dynamic_classifiers(self, x, y):
        for clf in self._dynamic_clfs:
            self._update_classifier(x, y, clf)

    def _deal_instance(self, new_instance, i):
        x, y = self._cache[i]

        labeling = self._uncertainty_strategy(x)

        if labeling is False:
            labeling = self._random_strategy()

        if labeling is True:
            self._update_stable_classifier(x, y)
            self._update_dynamic_classifiers(x, y)

        self._cache[i] = new_instance
