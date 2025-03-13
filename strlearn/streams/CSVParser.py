import pandas as pd
import numpy as np
import logging
import subprocess
from sklearn import preprocessing
from .DataStream import DataStream


class CSVParser(DataStream):
    """ Stream-aware parser of datasets in CSV format.

    :type path: string
    :param path: Path to the csv file.
    :type chunk_size: integer or 'auto', optional (default='auto'). 'auto' computes the chunk size automatically based on number of samples in the stream.
    :param chunk_size: The number of instances in each data chunk.
    :type n_chunks: integer, optional (default=250)
    :param n_chunks: The number of data chunks, that the stream is composed of.
    :type classes: array-like, optional (default=None)
    :param classes: list of classes in the datastream.
    :type delimiter: str, optional (default=',')
    :param delimiter: delimeter used to separate values in csv file.
    :type read_header: bool, optional (default=False)
    :param read_header: if True use first line of file as names for columns.

    :Example:

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

    def __init__(self, path, chunk_size='auto', n_chunks=250, classes=None, delimiter=',', read_header=False):
        self.name = path
        self.path = path
        n_samples = self.num_samples(read_header)
        if chunk_size == 'auto':
            chunk_size = n_samples // n_chunks
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        if self.chunk_size * self.n_chunks > n_samples:
            raise ValueError(
                f'Cannot create stream, chunk_size * n_chunks should be smaller or equal to number of all samples, got {self.chunk_size * self.n_chunks} > {n_samples}')
        self.classes_ = classes
        if classes is None:
            logging.warning('CSVParser: classes argument passed to constructor is None, it will be deduced based on first chunk')

        self.chunk_id = 0

        self.delimiter = delimiter
        self.read_header = read_header
        self.csv_iterator = iter(self.chunk_iter())

    def num_samples(self, read_header) -> int:
        n_lines = int(subprocess.check_output(f'wc -l {self.path}', shell=True).split()[0])
        if read_header:
            n_lines -= 1
        return n_lines

    def __str__(self):
        return f'CSVParser("{self.name}", chunk_size={self.chunk_size}, n_chunks={self.n_chunks}, classes={self.classes_}, delimiter="{self.delimiter}")'

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
        if hasattr(self, "current_chunk"):
            self.previous_chunk = self.current_chunk
        else:
            self.reset()

        self.chunk_id += 1

        if self.chunk_id < self.n_chunks:
            current_X, current_y = next(self.csv_iterator)
            if self.chunk_id == 0 and self.classes_ is None:
                label_encoder = preprocessing.LabelEncoder()
                self.classes_ = np.unique(label_encoder.fit_transform(current_y))
            self.current_chunk = (current_X, current_y)
            return self.current_chunk

    def chunk_iter(self):
        with open(self.path, 'r') as f:
            header = 0 if self.read_header else None
            for chunk in pd.read_csv(f, chunksize=self.chunk_size, delimiter=self.delimiter, header=header):
                chunk = chunk.to_numpy().astype(float)
                X, y = chunk[:, :-1], chunk[:, -1]
                yield X, y

    def reset(self):
        """Reset stream to the beginning."""
        self.previous_chunk = None
        self.chunk_id = -1
        self.csv_iterator = iter(self.chunk_iter())
