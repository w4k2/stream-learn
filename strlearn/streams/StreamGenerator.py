import numpy as np
import pandas as pd
from scipy.stats import logistic
from sklearn.datasets import make_classification
import pandas as pd

class StreamGenerator:
    """ Data streams generator for both stationary and drifting data streams.

    A key element of the ``stream-learn`` package is a generator that allows to prepare a replicable (according to the given ``random_state`` value) classification dataset with class distribution changing over the course of stream, with base concepts build on a default class distributions for the ``scikit-learn`` package from the ``make_classification()`` function. These types of distributions try to reproduce the rules for generating the ``Madelon`` set. The ``StreamGenerator`` is capable of preparing any variation of the data stream known in the general taxonomy of data streams.

    :param n_chunks: The number of data chunks, that the stream is composed of.
    :param chunk_size: The number of instances in each data chunk.
    :param random_state: The seed used by the random number generator.
    :param n_drifts: The number of concept changes in the data stream.
    :param concept_sigmoid_spacing: Value that determines the shape of sigmoid function and how sudden is the change of concept. The higher the value, the more sudden the drift is.
    :param n_classes: The number of classes in the generated data stream.
    :param y_flip: Label noise for whole dataset or separate classes.
    :param recurring: Determines if the streams can go back to the previously encountered concepts.
    :param weights: If array - class weight for static imbalance, if 3-valued tuple - (n_drifts, concept_sigmoid_spacing, IR amplitude [0-1]) for generation of continous dynamically imbalanced streams, if 2-valued tuple - (mean value, standard deviation) for generation of discreete dynamically imbalanced streams.

    :type n_chunks: integer, optional (default=250)
    :type chunk_size: integer, optional (default=200)
    :type random_state: integer, optional (default=1410)
    :type n_drifts: integer, optional (default=4)
    :type concept_sigmoid_spacing: float, optional (default=10.)
    :type n_classes: integer, optional (default=2)
    :type y_flip: float or tuple (default=0.01)
    :type recurring: boolean, optional (default=False)
    :type weights: array-like, shape (n_classes, ) or tuple (only for 2 classes)

    :Example:

    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator(n_drifts=2, weights=[0.2, 0.8], concept_sigmoid_spacing=5)
    >>> clf = sl.classifiers.AccumulatedSamplesClassifier()
    >>> evaluator = sl.evaluators.PrequentialEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    [[0.955      0.93655817 0.93601827 0.93655817 0.97142857]
     [0.94       0.91397849 0.91275313 0.91397849 0.96129032]
     [0.9        0.85565271 0.85234488 0.85565271 0.93670886]
     ...
     [0.815      0.72584133 0.70447376 0.72584133 0.8802589 ]
     [0.83       0.69522145 0.65223303 0.69522145 0.89570552]
     [0.845      0.67267706 0.61257135 0.67267706 0.90855457]]
    """

    def __init__(
        self,
        n_chunks=250,
        chunk_size=200,
        random_state=1410,
        n_drifts=0,
        concept_sigmoid_spacing=None,
        n_classes=2,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        recurring=False,
        weights=None,
        incremental=False,
        y_flip=0.01,
        **kwargs,
    ):
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.n_drifts = n_drifts
        self.concept_sigmoid_spacing = concept_sigmoid_spacing
        self.n_classes = n_classes
        self.make_classification_kwargs = kwargs
        self.recurring = recurring
        self.n_samples = self.n_chunks * self.chunk_size
        self.weights = weights
        self.incremental = incremental
        self.y_flip = y_flip
        self.classes_ = np.array(range(self.n_classes))
        self.n_features = n_features
        self.n_redundant = n_redundant
        self.n_informative = n_informative
        self.n_repeated = n_repeated
        self.n_clusters_per_class = n_clusters_per_class

    def is_dry(self):

        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def _sigmoid(self, sigmoid_spacing, n_drifts):
        period = (
            int((self.n_samples) / (n_drifts)) if n_drifts > 0 else int(self.n_samples)
        )
        css = sigmoid_spacing if sigmoid_spacing is not None else 9999
        _probabilities = (
            logistic.cdf(
                np.concatenate(
                    [
                        np.linspace(
                            -css if i % 2 else css, css if i % 2 else -css, period
                        )
                        for i in range(n_drifts)
                    ]
                )
            )
            if n_drifts > 0
            else np.ones(self.n_samples)
        )

        # Szybka naprawa, żeby dało się przepuścić podzielną z resztą liczbę dryfów
        probabilities = np.ones(self.n_chunks * self.chunk_size) * _probabilities[-1]
        probabilities[: _probabilities.shape[0]] = _probabilities

        return (period, probabilities)

    def _make_classification(self):
        np.random.seed(self.random_state)
        # To jest dziwna koncepcja z wagami z wierszy macierzy diagonalnej ale działa.
        # Jak coś działa to jest dobre.
        self.concepts = np.array(
            [
                [
                    make_classification(
                        **self.make_classification_kwargs,
                        n_samples=self.n_chunks * self.chunk_size,
                        n_classes=self.n_classes,
                        n_features=self.n_features,
                        n_informative=self.n_informative,
                        n_redundant=self.n_redundant,
                        n_repeated=self.n_repeated,
                        n_clusters_per_class=self.n_clusters_per_class,
                        random_state=self.random_state + i,
                        weights=weights.tolist(),
                    )[0].T
                    for weights in np.diag(
                        np.diag(np.ones((self.n_classes, self.n_classes)))
                    )
                ]
                for i in range(self.n_drifts + 1 if not self.recurring else 2)
            ]
        )

        # Prepare concept sigmoids if there are drifts
        if self.n_drifts > 0:
            # Get period and probabilities
            period, self.concept_probabilities = self._sigmoid(
                self.concept_sigmoid_spacing, self.n_drifts
            )

            # Szum
            self.concept_noise = np.random.rand(self.n_samples)

            # Inkrementalny
            if self.incremental:
                # Something
                self.a_ind = np.zeros(self.concept_probabilities.shape).astype(int)
                self.b_ind = np.ones(self.concept_probabilities.shape).astype(int)

                # Recurring
                if self.recurring is False:
                    for i in range(0, self.n_drifts):
                        start, end = (i * period, (i + 1) * period)
                        self.a_ind[start:end] = i + ((i + 1) % 2)
                        self.b_ind[start:end] = i + (i % 2)

                a = np.choose(self.a_ind, self.concepts)
                b = np.choose(self.b_ind, self.concepts)

                a = a * (1 - self.concept_probabilities)
                b = b * (self.concept_probabilities)
                c = a + b

            # Gradualny
            else:
                # Selekcja klas
                self.concept_selector = (
                    self.concept_probabilities < self.concept_noise
                ).astype(int)

                # Recurring drift
                if self.recurring is False:
                    for i in range(1, self.n_drifts):
                        start, end = (i * period, (i + 1) * period)
                        self.concept_selector[
                            np.where(self.concept_selector[start:end] == 1)[0] + start
                        ] = i + ((i + 1) % 2)
                        self.concept_selector[
                            np.where(self.concept_selector[start:end] == 0)[0] + start
                        ] = i + (i % 2)

        # Selekcja klas na potrzeby doboru balansu
        self.balance_noise = np.random.rand(self.n_samples)

        # Case of same size of all classes
        if self.weights is None:
            self.class_selector = (self.balance_noise * self.n_classes).astype(int)
        # If static balance is given
        elif not isinstance(self.weights, tuple):
            self.class_selector = np.zeros(self.balance_noise.shape).astype(int)
            accumulator = 0.0
            for i, treshold in enumerate(self.weights):
                mask = self.balance_noise > accumulator
                self.class_selector[mask] = i
                accumulator += treshold
        # If dynamic balance is given
        else:
            if len(self.weights) == 3:
                (
                    self.n_balance_drifts,
                    self.class_sigmoid_spacing,
                    self.balance_amplitude,
                ) = self.weights

                period, self.class_probabilities = self._sigmoid(
                    self.class_sigmoid_spacing, self.n_balance_drifts
                )

                # Amplitude correction
                self.class_probabilities -= 0.5
                self.class_probabilities *= self.balance_amplitude
                self.class_probabilities += 0.5

                # Will it work?
                self.class_selector = (
                    self.class_probabilities < self.balance_noise
                ).astype(int)
            elif len(self.weights) == 2:
                (
                    self.mean_prior,
                    self.std_prior
                ) = self.weights

                self.class_probabilities = np.random.normal(
                    self.mean_prior,
                    self.std_prior,
                    self.n_chunks
                )

                self.class_selector = np.random.uniform(size=(self.n_chunks,
                                                              self.chunk_size))

                self.class_selector[:, 0] = 0
                self.class_selector[:,-1] = 1

                self.class_selector = (self.class_selector > self.class_probabilities[:,np.newaxis]).astype(int)

                self.class_selector = np.ravel(self.class_selector)

        # Przypisanie klas i etykiet
        if self.n_drifts > 0:
            # Jeśli dryfy, przypisz koncepty
            if self.incremental:
                self.concepts = c
            else:
                self.concepts = np.choose(self.concept_selector, self.concepts)
        else:
            # Jeśli nie, przecież jest jeden, więc spłaszcz
            self.concepts = np.squeeze(self.concepts)

        # Assign objects to real classes
        X = np.choose(self.class_selector, self.concepts).T

        # Prepare label noise
        y = np.copy(self.class_selector)
        if isinstance(self.y_flip, float):
            # Global label noise
            flip_noise = np.random.rand(self.n_samples)
            y[flip_noise < self.y_flip] += 1
        elif isinstance(self.y_flip, tuple):
            if len(self.y_flip) == self.n_classes:
                for i, val in enumerate(self.y_flip):
                    mask = self.class_selector == i
                    y[(np.random.rand(self.n_samples) < val) & mask] += 1
            else:
                raise Exception(
                    "y_flip tuple should have as many values as classes in problem"
                )
        else:
            raise Exception("y_flip should be float or tuple")

        y = np.mod(y, self.n_classes)
        return X, y

    def reset(self):
        self.previous_chunk = None
        self.chunk_id = -1

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
        else:
            return None

    def __str__(self):
        if type(self.y_flip) == tuple and type(self.weights) != tuple:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_%i_d%i_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip[0] * 100),
                int(self.y_flip[1] * 100),
                50 if self.weights is None else (self.weights[0] * 100),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) != tuple and type(self.weights) != tuple:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_d%i_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip * 100),
                50 if self.weights is None else (self.weights[0] * 100),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) == tuple and type(self.weights) == tuple and len(self.weights) == 3:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_%i_dc%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip[0] * 100),
                int(self.y_flip[1] * 100),
                ("%i_%i_%.0f" % (self.weights[0],self.weights[1],self.weights[2]*100)),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) != tuple and type(self.weights) == tuple and len(self.weights) == 3:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_dc%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip * 100),
                ("%i_%i_%.0f" % (self.weights[0],self.weights[1],self.weights[2]*100)),
                int(self.chunk_size * self.n_chunks)
            )
        elif type(self.y_flip) == tuple and type(self.weights) == tuple and len(self.weights) == 2:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_%i_dd%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip[0] * 100),
                int(self.y_flip[1] * 100),
                ("%.0f_%.0f" % (self.weights[0]*100,self.weights[1]*100)),
                int(self.chunk_size * self.n_chunks),
            )
        elif type(self.y_flip) != tuple and type(self.weights) == tuple and len(self.weights) == 2:
            return "%s_%s_css%i_rs%i_nd%i_ln%i_dd%s_%i" % (
                "gr" if self.incremental is False else "inc",
                "n" if self.recurring is False else "r",
                999
                if self.concept_sigmoid_spacing is None
                else self.concept_sigmoid_spacing,
                self.random_state,
                self.n_drifts,
                int(self.y_flip * 100),
                ("%.0f_%.0f" % (self.weights[0]*100,self.weights[1]*100)),
                int(self.chunk_size * self.n_chunks)
            )

    def save_to_arff(self, filepath):
        """
        Save generated stream to the ARFF format file.

        :param filepath: Path to the file where data will be saved in ARFF format.
        :type filepath: string
        """
        X_array = []
        y_array = []

        for i in range(self.n_chunks):
            X, y = self.get_chunk()
            X_array.extend(X)
            y_array.extend(y)

        X_array = np.array(X_array)
        y_array = np.array(y_array)
        classes = np.unique(y_array)
        data = np.column_stack((X_array, y_array))

        header = "@relation %s %s\n\n" % (
            (filepath.split("/")[-1]).split(".")[0],
            str(self),
        )

        for feature in range(self.n_features):
            header += "@attribute feature" + str(feature + 1) + " numeric \n"

        header += "@attribute class {%s} \n\n" % ",".join(map(str, classes))
        header += "@data\n"

        with open(filepath, "w") as file:
            file.write(str(header))
            np.savetxt(file, data, fmt="%.20g", delimiter=",")
            file.write("\n")

        self.reset()

    def save_to_npy(self, filepath):
        """
        Save generated stream to the numpy format file.

        :param filepath: Path to the file where data will be saved in numpy format.
        :type filepath: string
        """
        X, y = self._make_classification()
        ds = np.concatenate([X, y[:, np.newaxis]], axis=1)
        np.save(filepath, ds)


    def save_to_csv(self, filepath):
        """
        Save generated stream to the csv format file.

        :param filepath: Path to the file where data will be saved in csv format.
        :type filepath: string
        """
        X, y = self._make_classification()

        ds = np.concatenate([X, y[:, np.newaxis]], axis=1)

        pdds = pd.DataFrame(ds)
        pdds.infer_objects()
        pdds.iloc[: , -1] = pdds.iloc[: , -1].astype(int)
        pdds.to_csv(filepath, header=None,index=None)
