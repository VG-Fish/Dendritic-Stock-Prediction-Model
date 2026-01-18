import shutil
from pathlib import Path

import kagglehub


def download_dataset() -> None:
    path: str = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    source_directory: Path = Path(path)
    dataset_directory: Path = Path(".") / "stock_market_dataset"

    if not dataset_directory.exists():
        shutil.copytree(source_directory, dataset_directory)
