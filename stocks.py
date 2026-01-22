from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


@dataclass
class NASDAQDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


class NASDAQDataset(Dataset):
    def __init__(self: Self, df: pl.DataFrame) -> None:
        self.features = torch.tensor(
            np.stack(
                [
                    np.stack(df["Close"].to_numpy()),  # pyright: ignore[reportCallIssue, reportArgumentType]
                    np.stack(df["Open"].to_numpy()),  # pyright: ignore[reportCallIssue, reportArgumentType]
                    np.stack(df["Volume"].to_numpy()),  # pyright: ignore[reportCallIssue, reportArgumentType]
                    np.stack(df["Range"].to_numpy()),  # pyright: ignore[reportCallIssue, reportArgumentType]
                    np.stack(df["Rolling STD"].to_numpy()),  # pyright: ignore[reportCallIssue, reportArgumentType]
                ],
                axis=2,
            ),
            dtype=torch.float32,
        )

        self.targets = torch.tensor(
            df["Target"].to_numpy(), dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self: Self) -> int:
        return len(self.features)

    def __getitem__(self: Self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
