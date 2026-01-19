from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from download import NASDAQDatasetInfo, NASDAQDownloader

# Initialize important variables
RANDOM_SEED: int = 1290

device: torch.device = torch.device("cpu")
if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

downloader: NASDAQDownloader = NASDAQDownloader()
dataset_directory_info: NASDAQDatasetInfo = downloader.download_dataset(
    stop_if_dest_dir_exists=True
)
stocks_directory: Path = dataset_directory_info.stocks_directory

SEQUENCE_LENGTH: int = 60


def make_windows(stock: pl.DataFrame) -> pl.DataFrame:
    scaler: StandardScaler = StandardScaler()

    stock = stock.sort("Date")
    close: np.ndarray = scaler.fit_transform(stock["Close"].reshape((-1, 1)))

    # Ignore cases where there's less values than SEQUENCE_LENGTH
    if len(close) < SEQUENCE_LENGTH:
        return pl.DataFrame(
            {"Ticker": [], "Sequence": [], "Sequence Dates": [], "Target": []}
        )

    total_indices: int = len(close) - SEQUENCE_LENGTH
    out: Dict = {
        "Ticker": np.repeat(stock["Ticker"][0], total_indices),
        "Sequence": np.empty((total_indices, SEQUENCE_LENGTH - 1)),
        "Sequence Dates": [],  # Using a Python list because funky stuff happens when converting polars Dates to numpy
        "Target": np.empty(total_indices),
    }
    for end in range(SEQUENCE_LENGTH, len(close)):
        start: int = end - SEQUENCE_LENGTH
        out["Sequence"][start] = close[start : end - 1, 0]
        out["Sequence Dates"].append(stock["Date"][start : end - 1])
        out["Target"][start] = close[end - 1, 0]
    return pl.DataFrame(out)


STOCKS_DATA_PATH: str = "stocks_data.parquet"


def save_stocks_dataset() -> None:
    lazy_frames: List[pl.LazyFrame] = []
    for file in stocks_directory.glob("*.csv"):
        lazy_frames.append(
            pl.scan_csv(file, try_parse_dates=True)
            .with_columns(
                [
                    pl.col("Date").dt.date(),
                    pl.lit(file.stem).alias("Ticker"),
                ]
            )
            .drop_nulls()
            .select(["Date", "Close", "Ticker"])
        )

    stock_data: pl.DataFrame = (
        pl.concat(lazy_frames)
        .group_by("Ticker")
        .map_groups(
            make_windows,
            schema={
                "Ticker": pl.String,
                "Sequence": pl.List(pl.Float64),
                "Sequence Dates": pl.List(pl.Date),
                "Target": pl.Float64,
            },
        )
    ).collect()
    stock_data.write_parquet(dataset_directory_info.parent_directory / STOCKS_DATA_PATH)


save_stocks_dataset()


def create_dataset_from(
    path: Path, train_fraction: float = 0.8, val_fraction: float = 0.1
) -> pl.DataFrame:
    stocks_data: pl.DataFrame = pl.read_parquet(path)
    print(stocks_data.head())
    # unique_dates = stocks_data.select("Date").unique().sort("Date").to_series()

    # num_dates = len(unique_dates)
    # if train_fraction > 1.0:
    #     raise ValueError("train_fraction must be < 1.0")
    # elif train_fraction + val_fraction > 1.0:
    #     raise ValueError("train_fraction + val_fraction must be < 1.0")

    # train_end_idx = int(train_fraction * num_dates)
    # val_end_idx = int((train_fraction + val_fraction) * num_dates)

    # train_last_date = unique_dates[train_end_idx - 1]
    # val_last_date = unique_dates[val_end_idx - 1]

    # train_dataset: Dataset = stocks_data.filter(
    #     pl.col("Date") <= train_last_date
    # ).to_torch(return_type="dataset")
    # print(train_dataset.schema)
    # train_dataloader: DataLoader = DataLoader(
    #     train_dataset, batch_size=64, shuffle=True
    # )

    # val_dataset: Dataset = stocks_data.filter(
    #     (pl.col("Date") > train_last_date) & (pl.col("Date") <= val_last_date)
    # ).to_torch(return_type="dataset")
    # val_dataloader: DataLoader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # test_dataset: Dataset = stocks_data.filter(pl.col("Date") > val_last_date).to_torch(
    #     return_type="dataset"
    # )
    # test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return stocks_data


stocks_data: pl.DataFrame = create_dataset_from(
    dataset_directory_info.parent_directory / STOCKS_DATA_PATH
)
print(stocks_data.head())
