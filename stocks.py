from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


@dataclass
class StocksDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


class StocksDataset(Dataset):
    def __init__(self: Self, stock_df: pl.DataFrame, device: torch.device):
        X: np.ndarray = stock_df["Sequence"].to_numpy().astype(np.float32)
        y: np.ndarray = stock_df["Target"].to_numpy().astype(np.float32)

        self.X: torch.Tensor = torch.from_numpy(X).to(device)
        self.y: torch.Tensor = torch.from_numpy(y).to(device)

    def __len__(self: Self) -> int:
        return self.X.shape[0]

    def __getitem__(self: Self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
