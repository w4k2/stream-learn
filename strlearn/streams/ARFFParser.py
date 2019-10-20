"""
ARFF Parser.
A class to parse datasets in ARFF standard.
"""
import numpy as np
from sklearn import preprocessing

ATYPES = ("nominal", "numeric")


class ARFFParser:
    """Stream-aware parser of datasets in ARFF format."""

    def __init__(self, path, chunk_size=500, n_chunks=100, n_classes=2):
        """Initializer."""
        # Read file.
        self.name = path
        self.path = "%s.arff" % path
        self._f = open("%s.arff" % path, "r")
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.n_classes = n_classes

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

        self.chunk_id = 0
        # Analyze its header
        while True:
            line = self._f.readline()[:-1]
            if line == "@data":
                self._f.readline()
                break

            elements = line.split(" ")
            if elements[0] == "@attribute":
                if elements[1] == "class":
                    # Analyze classes
                    self.le = preprocessing.LabelEncoder()
                    self.le.fit(np.array(elements[2][1:-1].split(",")))
                    self.classes = self.le.transform(self.le.classes_)
                else:
                    self.names.append(elements[1])
                    if elements[2][0] == "{":
                        self.types.append("nominal")
                        le = preprocessing.LabelEncoder()
                        le.fit(np.array(elements[2][1:-1].split(",")))
                        self.lencs.update({len(self.names) - 1: le})
                    elif elements[2] == "numeric":
                        self.types.append("numeric")
            elif elements[0] == "@relation":
                self.relation = " ".join(elements[1:])

        self.types = np.array(self.types)
        self.nominal_atts = np.where(self.types == "nominal")[0]
        self.numeric_atts = np.where(self.types == "numeric")[0]
        self.n_attributes = self.types.shape[0]
        self.is_dry_ = False

        # Read first line
        self.a_line = self._f.readline()

    def __str__(self):
        return self.name

    def is_dry(self):
        """Checking if we have reached the end of the stream."""

        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def get_chunk(self):
        """Get Chunk of size."""

        if self.chunk_id == 0:
            self.previous_chunk = None
        else:
            self.previous_chunk = self.current_chunk

        size = self.chunk_size
        X, y = np.zeros((size, self.n_attributes)), []
        for i in range(size):
            # Read pattern and break it into elements
            if not self.a_line[-1] == "\n":
                self.is_dry_ = True
                line = self.a_line
            else:
                line = self.a_line[:-1]
            elements = line.split(",")

            # Get class
            y.append(elements[-2])

            # Read attributes
            attributes = np.array(elements[:-1])

            # Get nominal
            X[i, self.numeric_atts] = attributes[self.numeric_atts]
            X[i, self.nominal_atts] = [
                self.lencs[j].transform([attributes[j]])[0] for j in self.nominal_atts
            ]

            if self.is_dry_:
                self.reset()
                break
            else:
                # Catch dry stream with length dividable by chunk size
                self.a_line = self._f.readline()
                if not self.a_line:
                    self.is_dry_ = True
                    break

        y = self.le.transform(y)
        self.chunk_id += 1
        self.current_chunk = X[: i + 1, :], y[: i + 1]
        return self.current_chunk

    def reset(self):
        self.is_dry_ = False
        self.chunk_id = 0
        self._f.close()
