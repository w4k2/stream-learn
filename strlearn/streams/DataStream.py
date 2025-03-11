import abc


class DataStream(abc.ABC):

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
