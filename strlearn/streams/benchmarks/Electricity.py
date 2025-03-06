import pandas

from ..CSVParser import CSVParser
from ..utils import download_dataset, get_data_path


class Electricity(CSVParser):
    def __init__(self, chunk_size: int = 200, n_chunks: int = 100):
        data_path = get_data_path()
        download_dataset("https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/electricity.csv", data_path / "electricity.csv")
        super().__init__(data_path / "electricity.csv", chunk_size, n_chunks)

    def _make_classification(self):
        csv_content = pandas.read_csv(self.path)
        csv_content['class'] = csv_content['class'].map({'UP': 1, 'DOWN': 0})
        csv_np = csv_content.to_numpy().astype(float)
        return csv_np[:, :-1], csv_np[:, -1]

    def __str__(self):
        return f'Electricity(chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'
