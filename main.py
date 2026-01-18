import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from download import NASDAQDownloader

device: torch.device = torch.device("cpu")
if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

# Uncomment to download dataset
downloader: NASDAQDownloader = NASDAQDownloader()
downloader.download_dataset(stop_if_dest_dir_exists=False)

DATASET_DIRECTORY: str = "nasdaq_dataset"
t_df: pl.DataFrame = pl.read_csv(f"{DATASET_DIRECTORY}/stocks/AACB.csv")

fig, ax = plt.subplots()
ax.scatter(
    x=t_df["Date"],
    y=t_df["Close"],
)
fig.show()
