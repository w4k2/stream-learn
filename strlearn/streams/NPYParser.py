import numpy as np
from sklearn import preprocessing
from .DataStream import DataStream


class NPYParser(DataStream):
    """ Stream-aware parser of datasets in numpy format.

    :type path: string
    :param path: Path to the npy file.
    :type chunk_size: integer, optional (default=200)
    :param chunk_size: The number of instances in each data chunk.
    :type n_chunks: integer, optional (default=250)
    :param n_chunks: The number of data chunks, that the stream is composed of.

    :Example:

    >>> import strlearn as sl
    >>> stream = sl.streams.NPYParser("Agrawal.npy")
    >>> clf = sl.classifiers.AccumulatedSamplesClassifier()
    >>> evaluator = sl.evaluators.PrequentialEvaluator()
    >>> evaluator.process(clf, stream)
    >>> stream.reset()
    >>> print(evaluator.scores_)
    ...
    [[0.855      0.80815508 0.79478582 0.80815508 0.89679715]
    [0.795      0.75827674 0.7426779  0.75827674 0.84644195]
    [0.8        0.75313899 0.73559983 0.75313899 0.85507246]
    ...
    [0.885      0.86181169 0.85534199 0.86181169 0.91119691]
    [0.895      0.86935764 0.86452058 0.86935764 0.92134831]
    [0.87       0.85104088 0.84813907 0.85104088 0.9       ]]
    """

    def __init__(self, path, chunk_size='auto', n_chunks=250):
        # Read file.
        self.name = path
        self.path = path
        n_samples = np.load(self.path).shape[0]
        if chunk_size == 'auto':
            chunk_size = n_samples // n_chunks
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        if self.chunk_size * self.n_chunks > n_samples:
            raise ValueError(f'Cannot create stream, chunk_size * n_chunks should be smaller or equal to number of all samples, got {self.chunk_size * self.n_chunks} > {n_samples}')

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

        self.chunk_id = 0
        self.starting_chunk = False

    def _make_classification(self):
        ds = np.load(self.path)
        self.classes_ = np.unique(ds[:, -1]).astype(int)
        return ds[:, :-1], ds[:, -1]

    def __str__(self):
        return f'NPYParser("{self.name}", chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'

    def is_dry(self):
        """
        Checking if we have reached the end of the stream.

        :returns: flag showing if the stream has ended
        :rtype: boolean
        """
        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def get_chunk(self):
        """
        Generating a data chunk of a stream.

        Used by all evaluators but also accesible for custom evaluation.

        :returns: Generated samples and target values.
        :rtype: tuple {array-like, shape (n_samples, n_features), array-like, shape (n_samples, )}
        """
        if hasattr(self, "X"):
            self.previous_chunk = self.current_chunk
        else:
            self.X, self.y = self._make_classification()
            self.reset()

        self.chunk_id += 1

        if self.chunk_id < self.n_chunks:
            start, end = (
                self.chunk_size * self.chunk_id,
                self.chunk_size * self.chunk_id + self.chunk_size,
            )

            self.current_chunk = (self.X[start:end], self.y[start:end])
            return self.current_chunk

    def reset(self):
        """Reset stream to the beginning."""
        self.previous_chunk = None
        self.chunk_id = -1
