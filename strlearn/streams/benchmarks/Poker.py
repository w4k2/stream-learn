from ..ARFFParser import ARFFParser
from ..utils import download_dataset, get_data_path


class Poker(ARFFParser):
    def __init__(self, chunk_size: int = 'auto', n_chunks: int = 250):
        data_path = get_data_path()
        download_dataset("https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/poker-lsn-1-2vsAll-pruned.arff", data_path / "poker.arff")
        super().__init__(data_path / "poker.arff", chunk_size, n_chunks)

    def __str__(self):
        return f'Poker(chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'
