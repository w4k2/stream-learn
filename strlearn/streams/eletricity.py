from .CSVParser import CSVParser
from .utils import download_dataset, get_data_path


class Eletricity(CSVParser):
    def __init__(self, chunk_size: int = 200, n_chunks: int = 250):
        data_path = get_data_path()
        download_dataset("https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/electricity.csv", data_path / "electricity.csv")
        super().__init__(data_path / "electricity.csv", chunk_size, n_chunks)

    def __str__(self):
        return f'Eletricity(chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'
