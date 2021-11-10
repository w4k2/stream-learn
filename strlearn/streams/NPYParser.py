"""
NPY Parser.
A class to parse datasets in NPY standard.
"""
import numpy as np
from sklearn import preprocessing

class NPYParser:
    """
    Stream-aware parser of datasets in CSV format.

    Parameters
    ----------
    path : string
        Path to the CSV file.
    chunk_size : integer, optional (default=200)
        The number of instances in each data chunk.
    n_chunks : integer, optional (default=250)
        The number of data chunks, that the stream
        is composed of.

    Attributes
    ----------

    Examples
    --------
    >>> import strlearn as sl
    >>> stream = sl.streams.CSVParser("Agrawal.csv")
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

    def __init__(self, path, chunk_size=200, n_chunks=250):
        """Initializer."""
        # Read file.
        self.name = path
        self.path = path
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

        self.chunk_id = 0
        self.starting_chunk = False


    def _make_classification(self):
        # Read CSV
        ds = np.load(self.path)
        self.classes_ = np.unique(ds[:,-1]).astype(int)
        return ds[:,:-1], ds[:,-1]

    def __str__(self):
        return self.name

    def is_dry(self):
        """Checking if we have reached the end of the stream."""

        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def get_chunk(self):
        """
        Generating a data chunk of a stream.

        Returns
        -------
        current_chunk : tuple {array-like, shape (n_samples, n_features),
        array-like, shape (n_samples, )}
            Generated samples and target values.
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
        else:
            return None

    def reset(self):
        self.previous_chunk = None
        self.chunk_id = -1
