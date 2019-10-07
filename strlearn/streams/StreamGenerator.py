"""
Data streams generator.

A class for generating streams with various parameters.
"""

from sklearn.datasets import make_classification
import numpy as np
from scipy.stats import logistic


class StreamGenerator:
    """
    Data streams generator for both stationary
    and drifting data streams.

    Parameters
    ----------
    n_chunks : integer, optional (default=250)
        The number of data chunks, that the stream
        is composed of.
    chunk_size : integer, optional (default=200)
        The number of instances in each data chunk.
    random_state : integer, optional (default=1410)
        The seed used by the random number generator.
    n_drifts : integer, optional (default=4)
        The number of concept changes in the data stream.
    concept_sigmoid_spacing : integer, optional (default=10)
        Value that determines how sudden is the change of concept.
        The higher the value, the more sudden the drift is.
    n_classes : integer, optional (default=2)
        The number of classes in the generated data stream.

    Attributes
    ----------

    """
    def __init__(
        self,
        n_chunks=250,
        chunk_size=200,
        random_state=1410,
        n_drifts=4,
        concept_sigmoid_spacing=10,
        n_classes=2,
        **kwargs,
    ):
        # Wyższy spacing, bardziej nagły
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.n_drifts = n_drifts
        self.concept_sigmoid_spacing = concept_sigmoid_spacing
        self.n_classes = n_classes
        self.make_classification_kwargs = kwargs

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
            # To pomocniczo
            n_samples = self.n_chunks * self.chunk_size
            X_a, y_a = make_classification(
                **self.make_classification_kwargs,
                n_samples=self.n_chunks * self.chunk_size,
                n_classes=self.n_classes,
                random_state=self.random_state,
            )
            X_b, y_b = make_classification(
                **self.make_classification_kwargs,
                n_samples=self.n_chunks * self.chunk_size,
                n_classes=self.n_classes,
                random_state=self.random_state + 1,
            )
            big_X = np.array([X_a, X_b])
            big_y = np.array([y_a, y_b])

            # Okres
            period = (
                int((n_samples) / (self.n_drifts))
                if self.n_drifts > 0
                else int(n_samples)
            )

            # Sigmoid
            self.concept_sigmoid = (
                logistic.cdf(
                    np.concatenate(
                        [
                            np.linspace(
                                -self.concept_sigmoid_spacing
                                if i % 2
                                else self.concept_sigmoid_spacing,
                                self.concept_sigmoid_spacing
                                if i % 2
                                else -self.concept_sigmoid_spacing,
                                period,
                            )
                            for i in range(self.n_drifts)
                        ]
                    )
                )
                if self.n_drifts > 0
                else np.ones(n_samples)
            )
            # Szum
            self.concept_noise = np.random.rand(n_samples)

            # Selekcja klas
            self.concept_selector = (self.concept_sigmoid > self.concept_noise).astype(
                int
            )

            # Przypisanie klas {do naprawy}
            self.X = np.zeros(X_a.shape)
            self.X[self.concept_selector == 0] = big_X[0, self.concept_selector == 0, :]
            self.X[self.concept_selector == 1] = big_X[1, self.concept_selector == 1, :]

            self.y = np.zeros(y_a.shape)
            self.y[self.concept_selector == 0] = big_y[0, self.concept_selector == 0]
            self.y[self.concept_selector == 1] = big_y[1, self.concept_selector == 1]

            self.chunk_id = -1
            self.previous_chunk = None

        self.chunk_id += 1

        if self.chunk_id < self.n_chunks:
            start, end = (
                self.chunk_size * self.chunk_id,
                self.chunk_size * self.chunk_id + self.chunk_size,
            )

            self.current_chunk = (self.X[start:end], self.y[start:end])
            print(self.current_chunk)

            return self.current_chunk
        else:
            return None
