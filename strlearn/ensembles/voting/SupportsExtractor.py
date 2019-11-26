from typing import List

import numpy as np
from attr import attrs
from sklearn.preprocessing import normalize

from strlearn.base.types import Classifier


@attrs(auto_attribs=True, frozen=True)
class SupportsExtractor:
    _ensemble: List[Classifier] = None
    _weights: List[int] = None
    _classes: List = None
    _normalized: bool = False

    def extract(self, x):
        all_members_can_return_supports = all([hasattr(clf, 'predict_proba') for clf in self._ensemble])
        if all_members_can_return_supports:
            supports_by_clf = [clf.predict_proba(x) * weight for (weight, clf) in zip(self._weights, self._ensemble)]
            return self._normalize_if_required(sum(supports_by_clf))

        else:
            predictions_by_clf = [clf.predict(x) for clf in self._ensemble]
            supports_by_clf = [
                np.vstack(
                    [(predictions == clazz).T * weight for clazz in self._classes]
                ) for (weight, predictions) in zip(self._weights, predictions_by_clf)
            ]

            return self._normalize_if_required(sum(supports_by_clf).T)

    def _normalize_if_required(self, supports):
        return normalize(supports, axis=0) if self._normalized else supports
