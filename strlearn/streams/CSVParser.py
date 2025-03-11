import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
# from .DataStream import DataStream
from DataStream import DataStream


class CSVParser(DataStream):
    """ Parser of datasets in CSV format. Prealoads all data at once. Could be faster for smaller streams.

    :type path: string
    :param path: Path to the csv file.
    :type chunk_size: integer or 'auto', optional (default='auto'). 'auto' computes the chunk size automatically based on number of samples in the stream.
    :param chunk_size: The number of instances in each data chunk.
    :type n_chunks: integer, optional (default=250)
    :param n_chunks: The number of data chunks, that the stream is composed of.

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

    def __init__(self, path, chunk_size='auto', n_chunks=250, skip_firstline=False, delimiter=','):
        self.name = path
        self.path = path
        n_lines = self.num_lines(skip_firstline, delimiter)
        if chunk_size == 'auto':
            chunk_size = n_lines // n_chunks
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        if self.chunk_size * self.n_chunks > n_lines:
            raise ValueError(f'Cannot create stream, chunk_size * n_chunks should be smaller or equal to number of all samples, got {self.chunk_size * self.n_chunks} > {n_lines}')

        self.chunk_id = 0
        self.skip_firstline = skip_firstline
        self.delimiter = delimiter

    def num_lines(self, skip_firstline, delimiter) -> int:
        with open(self.path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=delimiter)
            for _ in csv_reader:
                pass
            n_lines = csv_reader.line_num
        if skip_firstline:
            n_lines -= 1
        return n_lines

    def _make_classification(self):
        csv_content = self._read_csv()
        csv_np = np.array(csv_content).astype(float)
        return csv_np[:, :-1], csv_np[:, -1]

    def _read_csv(self):
        with open(self.path, 'r') as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            lines = [line for line in reader]
            if self.skip_firstline:
                lines.pop(0)
            return lines

    def __str__(self):
        return f'CSVParser("{self.name}", chunk_size={self.chunk_size}, n_chunks={self.n_chunks}, skip_firstline={self.skip_firstline}, delimiter={self.delimiter})'

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
            self.label_encoder = preprocessing.LabelEncoder()
            self.classes_ = np.unique(self.label_encoder.fit_transform(self.y))
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


class IncrementalCSVParser(DataStream):
    """ Stream-aware parser of datasets in CSV format.

    :type path: string
    :param path: Path to the csv file.
    :type chunk_size: integer or 'auto', optional (default='auto'). 'auto' computes the chunk size automatically based on number of samples in the stream.
    :param chunk_size: The number of instances in each data chunk.
    :type n_chunks: integer, optional (default=250)
    :param n_chunks: The number of data chunks, that the stream is composed of.

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

    def __init__(self, path, chunk_size='auto', n_chunks=250, skip_firstline=False, delimiter=','):
        self.name = path
        self.path = path
        n_lines = self.num_lines(skip_firstline, delimiter)
        if chunk_size == 'auto':
            chunk_size = n_lines // n_chunks
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        if self.chunk_size * self.n_chunks > n_lines:
            raise ValueError(f'Cannot create stream, chunk_size * n_chunks should be smaller or equal to number of all samples, got {self.chunk_size * self.n_chunks} > {n_lines}')

        self.chunk_id = 0
        self.starting_chunk = False

        self.skip_firstline = skip_firstline
        self.delimiter = delimiter
        self.csv_iterator = iter(self.line_iter())
        self.label_encoder = preprocessing.LabelEncoder()

    def num_lines(self, skip_firstline, delimiter) -> int:
        with open(self.path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=delimiter)
            for _ in csv_reader:
                pass
            n_lines = csv_reader.line_num
        if skip_firstline:
            n_lines -= 1
        return n_lines

    def __str__(self):
        return f'CSVParser("{self.name}", chunk_size={self.chunk_size}, n_chunks={self.n_chunks}, skip_firstline={self.skip_firstline}, delimiter={self.delimiter})'

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
            current_X = []
            current_y = []
            while len(current_X) < self.chunk_size:
                X, y = next(self.csv_iterator)
                current_X.append(X)
                current_y.append(y)
            current_X = np.stack(current_X)
            current_y = np.stack(current_y)
            self.classes_ = np.unique(self.label_encoder.fit_transform(current_y))  # TODO should we in this case update classes with each chunk?
            self.current_chunk = (current_X, current_y)
            return self.current_chunk

    def line_iter(self):
        with open(self.path, 'r') as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            first_line = True
            for line in reader:
                if self.skip_firstline and first_line:
                    first_line = False
                    continue
                line_np = np.array(line).astype(float)
                X, y = line_np[:-1], line_np[-1]
                yield X, y

    def reset(self):
        """Reset stream to the beginning."""
        self.previous_chunk = None
        self.chunk_id = -1


class IncrementalCSVParser1(DataStream):
    """ Stream-aware parser of datasets in CSV format.

    :type path: string
    :param path: Path to the csv file.
    :type chunk_size: integer or 'auto', optional (default='auto'). 'auto' computes the chunk size automatically based on number of samples in the stream.
    :param chunk_size: The number of instances in each data chunk.
    :type n_chunks: integer, optional (default=250)
    :param n_chunks: The number of data chunks, that the stream is composed of.

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

    def __init__(self, path, chunk_size='auto', n_chunks=250, skip_firstline=False, delimiter=','):
        self.name = path
        self.path = path
        n_lines = self.num_lines(skip_firstline, delimiter)
        if chunk_size == 'auto':
            chunk_size = n_lines // n_chunks
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        if self.chunk_size * self.n_chunks > n_lines:
            raise ValueError(f'Cannot create stream, chunk_size * n_chunks should be smaller or equal to number of all samples, got {self.chunk_size * self.n_chunks} > {n_lines}')

        self.chunk_id = 0
        self.starting_chunk = False

        self.skip_firstline = skip_firstline
        self.delimiter = delimiter
        self.csv_iterator = iter(self.chunk_iter())
        self.label_encoder = preprocessing.LabelEncoder()

    def num_lines(self, skip_firstline, delimiter) -> int:
        with open(self.path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=delimiter)
            for _ in csv_reader:
                pass
            n_lines = csv_reader.line_num
        if skip_firstline:
            n_lines -= 1
        return n_lines

    def __str__(self):
        return f'CSVParser("{self.name}", chunk_size={self.chunk_size}, n_chunks={self.n_chunks}, skip_firstline={self.skip_firstline}, delimiter={self.delimiter})'

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
            # current_X = []
            # current_y = []
            # while len(current_X) < self.chunk_size:
            #     X, y = next(self.csv_iterator)
            #     current_X.append(X)
            #     current_y.append(y)
            # current_X = np.stack(current_X)
            # current_y = np.stack(current_y)

            current_X, current_y = next(self.csv_iterator)
            self.classes_ = np.unique(self.label_encoder.fit_transform(current_y))  # TODO should we in this case update classes with each chunk?
            self.current_chunk = (current_X, current_y)
            return self.current_chunk

    def chunk_iter(self):
        for chunk in pd.read_csv(self.path, chunksize=self.chunk_size, delimiter=self.delimiter):
            chunk = chunk.to_numpy().astype(float)
            X, y = chunk[:, :-1], chunk[:, -1]
            yield X, y

    def reset(self):
        """Reset stream to the beginning."""
        self.previous_chunk = None
        self.chunk_id = -1


if __name__ == '__main__':
    import time
    import pickle

    # filepath = '/home/jkozal/Documents/PWr/active-learning-data-streams/data/accelerometer/accelerometer.csv'
    filepath = '/home/jkozal/Documents/PWr/active-learning-data-streams/data/wine/winequality-red.csv'

    start = time.time()
    csv_parser = CSVParser(filepath, skip_firstline=True, delimiter=';', n_chunks=10)
    all_X1 = []
    for X, y in csv_parser:
        all_X1.append(X)
    all_X1 = np.concatenate(all_X1, axis=0)
    end = time.time()
    print(all_X1)
    print('shape = ', all_X1.shape)
    duration = end - start
    print(f'normal parser duration = {duration}')
    print(f'object size {len(pickle.dumps(csv_parser))}')
    print(len(pickle.dumps(csv_parser.X)))

    start = time.time()
    csv_parser = IncrementalCSVParser(filepath, skip_firstline=True, delimiter=';', n_chunks=10)
    all_X2 = []
    for X, y in csv_parser:
        all_X2.append(X)
    all_X2 = np.concatenate(all_X2, axis=0)
    end = time.time()
    print(all_X2)
    print('shape = ', all_X2.shape)
    duration = end - start
    print(f'incremental parser duration = {duration}')
    csv_parser.csv_iterator = None
    print(f'object size {len(pickle.dumps(csv_parser))}')

    start = time.time()
    csv_parser = IncrementalCSVParser1(filepath, skip_firstline=True, delimiter=';', n_chunks=10)
    all_X3 = []
    for X, y in csv_parser:
        all_X3.append(X)
    all_X3 = np.concatenate(all_X3, axis=0)
    end = time.time()
    print(all_X3)
    print('shape = ', all_X3.shape)
    duration = end - start
    print(f'incremental parser duration = {duration}')
    csv_parser.csv_iterator = None
    print(f'object size {len(pickle.dumps(csv_parser))}')

    assert (all_X1 == all_X2).all()
    assert (all_X3 == all_X2).all()

    """
    result for 3.7 Mb csv file:

    normal parser duration = 0.46501827239990234
    object size 6198709
    4896164
    incremental parser duration = 0.8528943061828613
    object size 52803
    incremental parser duration = 0.1684119701385498
    object size 52804

    result for 84.2 kB csv file:
    normal parser duration = 0.010370254516601562
    object size 155525
    140874
    incremental parser duration = 0.024079084396362305
    object size 1931
    incremental parser duration = 0.12195658683776855
    object size 1932

    ten sam plik dla n_chunks=10, chunksize=160
    normal parser duration = 0.010422229766845703
    object size 184907
    140874
    incremental parser duration = 0.01478266716003418
    object size 31361
    incremental parser duration = 0.009056806564331055
    object size 31362
    """
