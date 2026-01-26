from dataclasses import dataclass
from typing import List, Self, Tuple

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
        self.stock_ids: torch.Tensor = torch.tensor(
            df["Stock ID"].to_numpy().astype(np.float64), dtype=torch.long
        )
        self.feature_cols: List[str] = [
            c for c in df.columns if c not in ("Target", "Stock ID")
        ]

        n_samples: int = len(df)
        sequence_length: int = len(df[self.feature_cols[0]][0])

        flat_features: np.ndarray = (
            df.select(self.feature_cols).explode(self.feature_cols).to_numpy()
        )

        # Converting the flat numpy array to a tensor, then 'viewing' it as a 3D tensor.
        self.features: torch.Tensor = torch.tensor(
            flat_features, dtype=torch.float
        ).view(
            n_samples, sequence_length, -1
        )  # -1 means PyTorch automatically calculates the size

        self.targets: torch.Tensor = torch.tensor(
            df["Target"].to_numpy(), dtype=torch.float
        ).unsqueeze(1)

    def __len__(self: Self) -> int:
        return len(self.features)

    def __getitem__(
        self: Self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.stock_ids[idx], self.targets[idx]
