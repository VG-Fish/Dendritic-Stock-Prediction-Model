from pathlib import Path

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

scaler: StandardScaler = StandardScaler()

SEQUENCE_LENGTH: int = 60
print(dataset_directory_info)
