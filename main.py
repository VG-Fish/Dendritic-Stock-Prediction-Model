import os
from pathlib import Path
from tkinter import SE
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from download import NASDAQDatasetInfo, NASDAQDownloader

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

scaler: StandardScaler = StandardScaler()

SEQUENCE_LENGTH: int = 60


def make_windows(stock: pl.DataFrame) -> pl.DataFrame:
    stock = stock.sort("Date")
    close: np.ndarray = stock["Close"].to_numpy()

    total_indices: int = len(close) - SEQUENCE_LENGTH
    out: Dict = {
        "Ticker": np.repeat(stock["Ticker"][0], total_indices),
        "Sequence": np.empty((total_indices, SEQUENCE_LENGTH)),
        "Target": np.empty(total_indices),
    }
    for end in range(SEQUENCE_LENGTH, len(close)):
        start: int = end - SEQUENCE_LENGTH
        out["Sequence"][start] = close[start:end]
        out["Target"][start] = close[end]
    return pl.DataFrame(out)


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
        .filter(pl.col("Close").len() > SEQUENCE_LENGTH)
    )

stock_data: pl.DataFrame = (
    pl.concat(lazy_frames)
    .group_by("Ticker")
    .map_groups(
        make_windows,
        {
            "Ticker": pl.String,
            "Sequence": pl.List(pl.Float64),
            "Target": pl.Float64,
        },
    )
).collect()
print(stock_data.head())
print(stock_data.columns)
print(stock_data.schema)
print(stock_data.shape)
