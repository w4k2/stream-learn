from .stream import Steam
import wget
# TODO: https://github.com/scikit-multiflow/streaming-datasets


def download_dataset(url: str, path: str):
    wget.download(url, out=path)


class Eletricity(Steam):
    def __init__(self):
        super().__init__()

    def get_chunk(self):
        raise NotImplementedError

    def is_dry(self):
        raise NotImplementedError
