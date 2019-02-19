"""
ARFF Parser.

A class to parse datasets in ARFF standard.
"""
import numpy as np
from sklearn import preprocessing

ATYPES = ("nominal", "numeric")


class ARFF:
    """Stream-aware parser of datasets in ARFF format."""

    def __init__(self, path, chunk_size=500):
        """Initializer."""
        self.chunk_size = chunk_size
        # Read file.
        self.path = path
        self._f = open(path, "r")

        # Prepare header storage
        self.types = []
        self.names = []
        self.lencs = {}

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
        self.is_dry = False

        # Read filrst line
        self.a_line = self._f.readline()

    def get_chunk(self):
        """Get Chunk of size."""
        size = self.chunk_size
        X, y = np.zeros((size, self.n_attributes)), []
        for i in range(size):
            # Read pattern and break it into elements
            if not self.a_line[-1] == "\n":
                self.is_dry = True
                line = self.a_line
            else:
                line = self.a_line[:-1]
            elements = line.split(",")

            # Get class
            y.append(elements[-1])

            # Read attributes
            attributes = np.array(elements[:-1])

            # Get nominal
            X[i, self.numeric_atts] = attributes[self.numeric_atts]
            X[i, self.nominal_atts] = [
                self.lencs[j].transform([attributes[j]])[0] for j in self.nominal_atts
            ]

            if self.is_dry:
                break
            else:
                # Catch dry stream with length dividable by chunk size
                self.a_line = self._f.readline()
                if not self.a_line:
                    self.is_dry = True
                    break

        y = self.le.transform(y)
        return X[: i + 1, :], y[: i + 1]

    def close(self):
        """Close file."""
        self._f.close()
