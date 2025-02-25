from ..ARFFParser import ARFFParser
from ..utils import download_dataset, get_data_path


class Covtype(ARFFParser):
    def __init__(self, chunk_size: int = 200, n_chunks: int = 250):
        data_path = get_data_path()
        download_dataset("https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/covtypeNorm-1-2vsAll-pruned.arff", data_path / "covtype.arff")
        super().__init__(data_path / "covtype.arff", chunk_size, n_chunks)

    def __str__(self):
        return f'Covtype(chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'
