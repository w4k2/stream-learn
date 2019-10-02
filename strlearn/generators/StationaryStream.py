from sklearn.datasets import make_classification


class StationaryStream:
    def __init__(
        self,
        n_chunks=250,
        chunk_size=200,
        random_state=1410,
        n_features=8,
        n_classes=2,
        weights=None,
    ):
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = weights

    def is_dry(self):
        return (
            self.chunk_id + 1 >= self.n_chunks if hasattr(self, "chunk_id") else False
        )

    def get_chunk(self):
        if hasattr(self, "X"):
            self.previous_chunk = self.current_chunk
        else:
            self.X, self.y = make_classification(
                n_samples=self.n_chunks * self.chunk_size,
                random_state=self.random_state,
                n_features=self.n_features,
                n_classes=self.n_classes,
                weights=self.weights,
            )
            self.chunk_id = -1
            self.previous_chunk = None

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
