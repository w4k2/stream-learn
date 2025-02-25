import wget
import pathlib


def download_dataset(url: str, output_path: str):
    if output_path.exists():
        return
    if type(output_path) != str:
        output_path = str(output_path)

    wget.download(url, str(output_path))


def get_data_path() -> pathlib.Path:
    strlearn_path = pathlib.Path('~/.strlearn').expanduser()  # TODO add path for Windows
    strlearn_path.mkdir(exist_ok=True)
    data_path = strlearn_path / "data"
    data_path.mkdir(exist_ok=True, parents=True)
    return data_path
