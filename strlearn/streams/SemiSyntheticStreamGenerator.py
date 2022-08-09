import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import problexity as px
from scipy.spatial.distance import euclidean

class SemiSyntheticStreamGenerator:
    """ Semi-Synthetic Data streams generator for drifting data streams.

    A generator that allows preparing a replicable classification dataset based on real-world input data. The generator uses one-dimensional interpolation to generate the drifting projections, based on which the final data stream is generated. 

    :param X: Static dataset features.
    :param y: Static dataset labels.
    :param n_chunks: The number of data chunks, that the stream is composed of.
    :param chunk_size: The number of instances in each data chunk.
    :param random_state: The seed used by the random number generator.
    :param n_drifts: The number of concept changes in the data stream.
    :param n_features: The number of features in output stream.
    :param interpolation: Interpolation type.
    :param stabilize_factor: The factor describing the stability of a concept.
    :param binarize: Flag describing if the data should be binarized.
    :param density: The number of possible drift points from which the generated drifts are randomly selected.
    :param base_projection_pool_size: Number of initial projections from which the final ones are selected.
    :param evaluation_measures: Measures based on which the projections are selected.
    
    :type X: array-like, shape (n_samples, n_features)
    :type y: array-like, shape (n_samples, )
    :type n_chunks: integer, optional (default=200)
    :type chunk_size: integer, optional (default=250)
    :type random_state: integer, optional (default=None)
    :type n_drifts: integer, optional (default=2)
    :type n_features: integer, optional (default=10)
    :type interpolation: string, optional (default='nearest')
    :type stabilize_factor: float, optional (default=0.2)
    :type binarize: boolean, optional (default=True)
    :type density: integer, optional (default=150)
    :type base_projection_pool_size: integer, optional (default=50)
    :type evaluation_measures: list, optional (default=[F1, N2])

    :Example:
    >>> import strlearn as sl
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.naive_bayes import GaussianNB
    >>> X, y =  load_breast_cancer(return_X_y=True)
    >>> stream = sl.streams.SemiSyntheticStreamGenerator(X, y, n_drifts=4, interpolation='cubic')
    >>> clf = GaussianNB()
    >>> evaluator = sl.evaluators.TestThenTrain()
    >>> evaluator.process(stream, clf)
    >>> print(stream._get_drifts())
    [ 14  48  89 155]
    """

    def __init__(
        self,
        X, 
        y,
        n_chunks=200,
        chunk_size=250,
        random_state=None,
        n_drifts=2,
        n_features=10,
        interpolation='nearest',
        stabilize_factor=0.2,
        binarize=True,
        density=150,
        base_projection_pool_size=50,
        evaluation_measures=[px.f1, px.n2]
    ):
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.random_state = random_state if random_state is not None else np.random.randint(10000)
        self.n_drifts = n_drifts
        self.n_samples = self.n_chunks * self.chunk_size
        self.n_features = n_features
        self.interpolation = interpolation
        self.stabilize_factor = stabilize_factor
        self.binarize = binarize
        self.density = density
        self.base_projection_pool_size = base_projection_pool_size
        self.evaluation_measures=evaluation_measures

        self.X_base = np.copy(X)
        self.y_base = np.copy(y)

        self.X = np.copy(X)
        self.y = np.copy(y)

        self.drift_basepoints = None
    
    def _get_drifts(self):
        if self.drift_basepoints is None:
            return []

        return (np.array([(int(self.drift_basepoints[bp_id-1]+(self.drift_basepoints[bp_id]-self.drift_basepoints[bp_id-1])/2)) 
                    for bp_id in range(1,len(self.drift_basepoints))])/self.chunk_size).astype(int)
    
    def _get_distances(self):
        if self.drift_basepoints is None:
            return []

        return [(self.drift_basepoints[bp_id]-self.drift_basepoints[bp_id-1]) 
                    for bp_id in range(1,len(self.drift_basepoints))]

    def _make_stream(self):
        np.random.seed(self.random_state)
        
        # Optionally binarize the data
        if self.binarize:
            self.y[self.y!=0] = 1

        self.classes_ = np.unique(self.y)
        class_indexes =[]

        # Perform random resampling
        for c_id, c in enumerate(self.classes_):
            ir = len(np.argwhere(self.y==c))/len(self.y)
            if c_id == len(self.classes_)-1:
                samples = self.n_samples - len(np.concatenate(class_indexes))
            else:
                samples = int(np.rint(ir*self.n_samples))
            indexes = np.random.choice(np.argwhere(self.y==c).flatten(), samples)
            class_indexes.append(indexes)

        X_arrs = [self.X[ind] for ind in class_indexes]
        y_arrs = [self.y[ind] for ind in class_indexes]

        self.X = np.concatenate((X_arrs))
        self.y = np.concatenate((y_arrs))

        # Shuffle the data
        p = np.random.permutation(len(self.y))

        self.X = self.X[p]
        self.y = self.y[p]

        # Generate basepoints
        possible_basepoints = np.linspace(0,self.n_samples,self.density)

        self.drift_basepoints = np.sort(np.random.choice(possible_basepoints, self.n_drifts+1, replace=False).astype(int))
        self.drift_basepoints[0] = 0
        self.drift_basepoints[-1] = self.n_samples

        # Select best projections
        n_concept_features = self.X.shape[1]
        base_projection_pool = np.random.normal(size=(self.base_projection_pool_size,
                                              n_concept_features,
                                              self.n_features))

        base = np.array([m(self.X_base, self.y_base) for m in self.evaluation_measures])
        base[np.isnan(base)] = 1
        projection_scores = []
        for bp in base_projection_pool:
            X_temp = np.sum(self.X_base[:, :, np.newaxis] * bp, axis=1)
            projection_scores.append([m(X_temp, self.y_base) for m in self.evaluation_measures])

        projection_scores = np.array(projection_scores)
        projection_scores[np.isnan(projection_scores)] = 1
        
        dist = [euclidean(projection_scores[i], base) for i in range(self.base_projection_pool_size)]
        proba = -np.array(dist)
        proba -= np.min(proba)
        proba /=np.sum(proba)

        try:
            base_projections_idx = np.random.choice(range(self.base_projection_pool_size), 
                                                p=proba, 
                                                replace=False, 
                                                size=self.n_drifts+1)
        except:
            base_projections_idx = np.random.choice(range(self.base_projection_pool_size), 
                                                replace=False, 
                                                size=self.n_drifts+1)

        base_projections = base_projection_pool[base_projections_idx]

        # Add auxiliary points     
        distances = self._get_distances()

        _drift_basepoints = []
        _base_projections = []
        for p_id, p in enumerate(self.drift_basepoints):
            sep_neg = int(distances[p_id-1]*self.stabilize_factor)
            try:
                sep_pos = int(distances[p_id]*self.stabilize_factor)
            except:
                sep_pos = 0

            _drift_basepoints.append(p - sep_neg)
            _drift_basepoints.append(p)
            _drift_basepoints.append(p + sep_pos)
            
            [_base_projections.append(base_projections[p_id]) for i in range(3)]

        # Remove first and last - out of range
        drift_basepoints_aux = np.array(_drift_basepoints)[1:-1]
        base_projections_aux = np.array(_base_projections)[1:-1]

        # Generate stream
        continous_projections = np.zeros((self.n_samples, n_concept_features, self.n_features))
        stream_basepoints = np.linspace(0, self.n_samples-1, self.n_samples).astype(int)

        for d_s in range(self.n_features): 
            for d_c in range(n_concept_features):
                original_values = base_projections_aux[:, d_c, d_s]
                f = interp1d(drift_basepoints_aux, original_values, kind=self.interpolation)
                continous_projections[:, d_c, d_s] = f(stream_basepoints)
                
        X_s = np.sum(self.X[:, :, np.newaxis] * continous_projections, axis=1)

        return X_s, np.copy(self.y)


    def is_dry(self):
        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

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
        if hasattr(self, "generated_X"):
            self.previous_chunk = self.current_chunk
        else:
            self.generated_X, self.generated_y = self._make_stream()
            self.reset()

        self.chunk_id += 1

        if self.chunk_id < self.n_chunks:
            start, end = (
                self.chunk_size * self.chunk_id,
                self.chunk_size * self.chunk_id + self.chunk_size,
            )

            self.current_chunk = (self.generated_X[start:end], self.generated_y[start:end])
            return self.current_chunk
        else:
            return None
        
    def __next__(self):
        while not self.is_dry():
            yield self.get_chunk()

    def __iter__(self):
        return next(self)

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
        X, y = self._make_stream()
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