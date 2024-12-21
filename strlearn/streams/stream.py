import abc


class Steam(abc.ABC):
    def __init__(self, n_features: int, n_targets: int, n_classes: int):
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_classes = n_classes

    @abc.abstractmethod
    def get_chunk(self):
        raise NotImplementedError

    @abc.abstractmethod
    def is_dry(self):
        raise NotImplementedError

    def __next__(self):
        while not self.is_dry():
            yield self.get_chunk()

    def __iter__(self):
        return next(self)
