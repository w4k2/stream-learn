import numpy as np
from sklearn import preprocessing
from .DataStream import DataStream

ATYPES = ("nominal", "numeric")


class ARFFParser(DataStream):
    """ Stream-aware parser of datasets in ARFF format.

    :type path: string
    :param path: Path to the ARFF file.
    :type chunk_size: integer, optional (default=200)
    :param chunk_size: The number of instances in each data chunk.
    :type n_chunks: integer, optional (default=250)
    :param n_chunks: The number of data chunks, that the stream is composed of.

    :Example:

    >>> import strlearn as sl
    >>> stream = sl.streams.ARFFParser("Agrawal.arff")
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
        self._f = open(path, "r")

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

        n_header_lines = self.analyze_header()
        n_lines = self.num_lines()
        n_samples = n_lines - n_header_lines
        if chunk_size == 'auto':
            chunk_size = n_samples // n_chunks
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

        self.chunk_id = 0
        self.starting_chunk = False
        if self.chunk_size * self.n_chunks > n_samples:
            raise ValueError(f'Cannot create stream, chunk_size * n_chunks should be smaller or equal to number of all samples, got {self.chunk_size * self.n_chunks} > {n_lines - n_header_lines}')

        self.types = np.array(self.types)
        self.nominal_atts = np.where(self.types == "nominal")[0]
        self.numeric_atts = np.where(self.types == "numeric")[0]
        self.n_attributes = self.types.shape[0]
        self.is_dry_ = False

        # Read first line
        self.a_line = self._f.readline()

    def analyze_header(self):
        header_lines = 0
        while True:
            line = self._f.readline()[:-1]
            pos = self._f.tell()
            header_lines += 1
            if line == "@data":
                line = self._f.readline()
                if line not in ["\n", "\r\n"]:
                    self._f.seek(pos)
                break

            elements = line.split(" ")
            if elements[0] == "@attribute":
                if elements[1] == "class":
                    # Analyze classes
                    if elements[-1] == '':
                        elements.pop()
                    if len(elements) > 3:
                        _, _, class_names = line.split(" ", maxsplit=2)
                        class_names = class_names[1:-1].replace(" ", "").split(",")
                    else:
                        class_names = elements[2][1:-1].split(",")

                    self.le = preprocessing.LabelEncoder()
                    self.le.fit(np.array(class_names))
                    self.classes_ = self.le.transform(self.le.classes_)
                    self.n_classes = len(self.classes_)
                else:
                    self.names.append(elements[1])
                    if elements[2][0] == "{" and elements[2][1] != "'":
                        self.types.append("nominal")
                        le = preprocessing.LabelEncoder()
                        le.fit(np.array(elements[2][1:-1].split(",")))
                        self.lencs.update({len(self.names) - 1: le})
                    elif elements[2][0] == "{" and elements[2][1] == "'":
                        self.types.append("nominal")
                        le = preprocessing.LabelEncoder()
                        temporary = np.array(elements[2][1:-1].split(","))
                        temporary = np.array([element[1:-1] for element in temporary])
                        le.fit(temporary)
                        self.lencs.update({len(self.names) - 1: le})
                    elif elements[2] in ["numeric", "real"]:
                        self.types.append("numeric")
            elif elements[0] == "@relation":
                self.relation = " ".join(elements[1:])
        return header_lines

    def num_lines(self) -> int:
        with open(self.name, 'r') as f:
            n_lines = sum(1 for _ in f)
        return n_lines

    def __del__(self):
        if hasattr(self, "_f"):
            self._f.close()

    def __str__(self):
        return f'ARFFParser("{self.name}", chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'

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
        if self.chunk_id == 0 and self.starting_chunk is False:
            self.previous_chunk = None
            self.chunk_id = -1
            self.starting_chunk = True
        else:
            self.previous_chunk = self.current_chunk

        size = self.chunk_size
        X, y = np.zeros((size, self.n_attributes)), []
        for i in range(size):
            if not self.a_line[-1] == "\n":
                self.is_dry_ = True
                line = self.a_line
            elif self.a_line == "\n":  # test arff files have two empty lines at the end
                self.is_dry_ = True
                continue
            else:
                line = self.a_line[:-1]
            elements = line.split(",")

            # Get class
            if elements[-1] == "":
                y.append(elements[-2].strip())
            else:
                y.append(elements[-1].strip())

            # Read attributes
            attributes = np.array(elements[:-1])
            attributes = np.array([att.strip() for att in attributes])
            # print(attributes)
            # exit()

            # Get nominal
            X[i, self.numeric_atts] = attributes[self.numeric_atts]
            X[i, self.nominal_atts] = [
                self.lencs[j].transform([attributes[j]])[0] for j in self.nominal_atts
            ]

            if self.is_dry_:
                break
            else:
                # Catch dry stream with length dividable by chunk size
                self.a_line = self._f.readline()

        y = self.le.transform(y)
        self.chunk_id += 1
        self.current_chunk = X[: i + 1, :], y[: i + 1]
        return self.current_chunk

    def reset(self):
        "Reset processed stream and close ARFF file."
        self.is_dry_ = False
        self.chunk_id = 0
        self._f.close()
