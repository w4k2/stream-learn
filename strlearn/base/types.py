from typing import NewType, Union, Callable, List

from sklearn.base import BaseEstimator, ClassifierMixin

Classifier = NewType('Classifier', Union[BaseEstimator, ClassifierMixin])
ClassifierProvider = NewType('ClassifierProvider',
                             Callable[[List, List, List], Classifier])  # (X, y, classes) -> clf
