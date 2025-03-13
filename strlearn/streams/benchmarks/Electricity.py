import pandas as pd

from ..CSVParser import CSVParser
from ..utils import download_dataset, get_data_path


class Electricity(CSVParser):
    def __init__(self, chunk_size: int = 'auto', n_chunks: int = 100):
        data_path = get_data_path()
        download_dataset("https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/electricity.csv", data_path / "electricity.csv")
        super().__init__(data_path / "electricity.csv", chunk_size, n_chunks, classes=[0, 1], read_header=True)

    def chunk_iter(self):
        with open(self.path, 'r') as f:
            header = 0 if self.read_header else None
            for chunk in pd.read_csv(f, chunksize=self.chunk_size, delimiter=self.delimiter, header=header):
                chunk['class'] = chunk['class'].map({'UP': 1, 'DOWN': 0})
                chunk = chunk.to_numpy().astype(float)
                X, y = chunk[:, :-1], chunk[:, -1]
                yield X, y

    def __str__(self):
        return f'Electricity(chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'
