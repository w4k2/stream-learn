from copy import deepcopy
from typing import Optional

from attr import attrs

from strlearn.base.types import Classifier, ClassifierProvider


@attrs(frozen=True, auto_attribs=True)
class ClassifierSupplier:
    _provider: Optional[ClassifierProvider] = None
    _classifier: Optional[Classifier] = None

    def get(self, x, y, classes=None):
        if self._provider is not None:
            return self._provider(x, y, classes)
        elif self._classifier is not None:
            clf = deepcopy(self._classifier)
            clf.partial_fit(x, y, classes)
            return clf

        raise ValueError("No provider or classifier instance were passed")
