import os
from pathlib import Path
from typing import List

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

lazy_frames: List[pl.LazyFrame] = []
for file in stocks_directory.glob("*.csv"):
    lazy_frames.append(
        pl.scan_csv(file, try_parse_dates=True)
        .with_columns(
            [
                pl.col("Date").dt.date(),
                pl.col("Volume").cast(pl.Float64),
                pl.lit(file.stem).alias("Ticker"),
            ]
        )
        .drop_nulls()
    )
stock_data: pl.DataFrame = pl.concat(lazy_frames).collect()
