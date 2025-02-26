from ..ARFFParser import ARFFParser
from ..utils import download_dataset, get_data_path


class Insects(ARFFParser):
    urls = {
        'INSECTS-abrupt_imbalanced_norm.arff': 'https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/INSECTS-abrupt_imbalanced_norm.arff',
        'INSECTS-abrupt_imbalanced_norm_5prc.arff': 'https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/INSECTS-abrupt_imbalanced_norm_5prc.arff',
        'INSECTS-gradual_imbalanced_norm.arff': 'https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/INSECTS-gradual_imbalanced_norm.arff',
        'INSECTS-gradual_imbalanced_norm_5prc.arff': 'https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/INSECTS-gradual_imbalanced_norm_5prc.arff',
        'INSECTS-incremental_imbalanced_norm.arff': 'https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/INSECTS-incremental_imbalanced_norm.arff',
        'INSECTS-incremental_imbalanced_norm_5prc.arff': 'https://raw.githubusercontent.com/w4k2/stream-datasets/refs/heads/main/INSECTS-incremental_imbalanced_norm_5prc.arff',
    }

    def __init__(self, drift_mode: str, subsample: bool = False, chunk_size: int = 200, n_chunks: int = 250):
        if drift_mode not in ['abrupt', 'gradual', 'incremental']:
            raise ValueError(f'drift mode should be abrupt, gradual or incremental, not {drift_mode}')
        self.drift_mode = drift_mode
        self.subsample = subsample

        filename = f'INSECTS-{drift_mode}_imbalanced_norm'
        if subsample:
            filename += "_5prc"
        filename += '.arff'
        data_url = self.urls[filename]

        data_path = get_data_path()
        download_dataset(data_url, data_path / filename)
        super().__init__(data_path / filename, chunk_size, n_chunks)

    def __str__(self):
        return f'Insects(drift_mode={self.subsample}, subsample={self.subsample} chunk_size={self.chunk_size}, n_chunks={self.n_chunks})'
