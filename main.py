from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from download import NASDAQDatasetInfo, NASDAQDownloader
from model import StockPredictionModel
from stocks import StocksDataLoaders, StocksDataset

# Initialize important variables
RANDOM_SEED: int = 1290
SEQUENCE_LENGTH: int = 60
STOCKS_DATA_PATH: str = "stocks_data.parquet"
TRAIN_FRACTION: float = 0.8
VAL_FRACTION: float = 0.1

torch.manual_seed(RANDOM_SEED)
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


def make_windows(stock: pl.DataFrame) -> pl.DataFrame:
    stock = stock.sort("Date")
    close: np.ndarray = scaler.transform(stock["Close"].to_numpy().reshape(-1, 1))  # pyright: ignore[reportAssignmentType]

    if len(close) < SEQUENCE_LENGTH:
        return pl.DataFrame({"Sequence": [], "Target": []})

    total_indices: int = len(close) - SEQUENCE_LENGTH
    out: Dict = {
        "Sequence": np.empty((total_indices, SEQUENCE_LENGTH - 1)),
        "Target": np.empty(total_indices),
    }

    for end in range(SEQUENCE_LENGTH, len(close)):
        start: int = end - SEQUENCE_LENGTH
        out["Sequence"][start] = close[start : end - 1, 0]
        out["Target"][start] = close[end - 1, 0]

    return pl.DataFrame(out)


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
            .select("Date", "Close", "Ticker")
        )

    raw_data: pl.DataFrame = pl.concat(lazy_frames).collect()
    train_slice: pl.DataFrame = raw_data[: int(len(raw_data) * TRAIN_FRACTION)]
    scaler.fit(train_slice["Close"].reshape((-1, 1)))

    stock_data: pl.DataFrame = (
        pl.concat(lazy_frames)
        .group_by("Ticker")
        .map_groups(
            make_windows,
            schema={
                "Sequence": pl.List(pl.Float64),
                "Target": pl.Float64,
            },
        )
    ).collect()

    stock_data.write_parquet(dataset_directory_info.parent_directory / STOCKS_DATA_PATH)


# save_stocks_dataset()


def create_datasets_from(
    path: Path,
) -> StocksDataLoaders:
    if TRAIN_FRACTION > 1.0:
        raise ValueError("train_fraction must be < 1.0")
    elif TRAIN_FRACTION + VAL_FRACTION > 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0")

    stocks_data: pl.DataFrame = pl.read_parquet(path)

    num_data_points: int = len(stocks_data)
    train_end_idx = int(TRAIN_FRACTION * num_data_points)
    val_end_idx = int((TRAIN_FRACTION + VAL_FRACTION) * num_data_points)

    batch_size: int = 256
    train_dataset: Dataset = StocksDataset(stocks_data[:train_end_idx], device=device)
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
    )

    val_dataset: Dataset = StocksDataset(
        stocks_data[train_end_idx:val_end_idx], device=device
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
    )

    test_dataset: Dataset = StocksDataset(stocks_data[val_end_idx:], device=device)
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
    )

    return StocksDataLoaders(train_dataloader, val_dataloader, test_dataloader)


data_loaders: StocksDataLoaders = create_datasets_from(
    dataset_directory_info.parent_directory / STOCKS_DATA_PATH
)

model: StockPredictionModel = StockPredictionModel(
    input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, device=device
).to(device)
criterion: nn.MSELoss = nn.MSELoss().to(device)
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.01)

num_epochs: int = 100

for epoch in tqdm(range(num_epochs), desc="Number of Epochs Left"):
    model.train()
    for X_train, y_train in tqdm(data_loaders.train, desc="Number of Batches Left"):
        X_train = X_train.unsqueeze(-1).to(device)
        y_train = y_train.unsqueeze(-1).to(device)

        y_train_pred: torch.Tensor = model(X_train)
        loss: torch.Tensor = criterion(y_train_pred, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        torch.mps.empty_cache()

    model.eval()
    all_predictions: List = []
    all_targets: List = []
    with torch.no_grad():
        for X_val, y_val in data_loaders.val:
            X_val = X_val.unsqueeze(-1).to(device)
            y_val = y_val.unsqueeze(-1).to(device)

            y_val_pred = model(X_val)

            all_predictions.append(y_val_pred.cpu())
            all_targets.append(y_val.cpu().reshape((-1, 1)))

            torch.mps.empty_cache()

    final_predictions_scaled: torch.Tensor = torch.vstack(all_predictions)
    final_targets_scaled: torch.Tensor = torch.vstack(all_targets)

    final_predictions: np.ndarray = scaler.inverse_transform(
        final_predictions_scaled.numpy()
    )
    final_targets: np.ndarray = scaler.inverse_transform(final_targets_scaled.numpy())

    val_rmse: float = root_mean_squared_error(final_targets, final_predictions)
    print(f"Epoch {epoch} | Val RMSE: ${val_rmse:.2f}")
