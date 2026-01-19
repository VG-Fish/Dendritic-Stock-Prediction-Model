from dataclasses import dataclass
from typing import Self, Tuple

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
        self.sequences: torch.Tensor = torch.tensor(
            stock_df["Sequence"], dtype=torch.float32
        )
        self.targets: torch.Tensor = torch.tensor(stock_df["Target"], torch.float32)

    def __len__(self: Self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self: Self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]
