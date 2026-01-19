from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from download import NASDAQDatasetInfo, NASDAQDownloader
from model import StockPredictionModel
from stocks import StocksDataLoaders, StocksDataset

# Initialize important variables
RANDOM_SEED = 1290
SEQUENCE_LENGTH = 60
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10

torch.manual_seed(RANDOM_SEED)
device: torch.device = torch.device("cpu")
if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")


@dataclass
class ProcessedData:
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray


def make_windows_per_ticker(stock: pl.DataFrame) -> ProcessedData:
    stock = stock.sort("Date")

    prices: np.ndarray = stock["Close"].to_numpy()
    log_returns: np.ndarray = np.diff(np.log(prices))  # Calculates log return

    n: int = len(log_returns)
    train_idx: int = int(TRAIN_FRACTION * n)
    val_idx: int = int((TRAIN_FRACTION + VAL_FRACTION) * n)

    def windowing(data: np.ndarray):
        if len(data) <= SEQUENCE_LENGTH:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i : i + SEQUENCE_LENGTH - 1])
            y.append(data[i + SEQUENCE_LENGTH - 1])
        return np.array(X), np.array(y)

    train_X, train_y = windowing(log_returns[:train_idx])
    val_X, val_y = windowing(log_returns[train_idx:val_idx])
    test_X, test_y = windowing(log_returns[val_idx:])

    return ProcessedData(train_X, train_y, val_X, val_y, test_X, test_y)


def prepare_data(stocks_dir: Path) -> Tuple[StocksDataLoaders, StandardScaler]:
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []

    for file in tqdm(stocks_dir.glob("*.csv")):
        df: pl.DataFrame = pl.read_csv(file).drop_nulls()
        # Ensures that model has enough context to learn something
        if len(df) < SEQUENCE_LENGTH + 10:
            continue

        processed_data: ProcessedData = make_windows_per_ticker(df)

        if processed_data.train_X.size > 0:
            all_train_X.append(processed_data.train_X)
            all_train_y.append(processed_data.train_y)
        if processed_data.val_X.size > 0:
            all_val_X.append(processed_data.val_X)
            all_val_y.append(processed_data.val_y)
        if processed_data.test_X.size > 0:
            all_test_X.append(processed_data.test_X)
            all_test_y.append(processed_data.test_y)

    train_X: np.ndarray = np.vstack(all_train_X)
    train_y: np.ndarray = np.concatenate(all_train_y)

    val_X: np.ndarray = np.vstack(all_val_X)
    val_y: np.ndarray = np.concatenate(all_val_y)

    test_X: np.ndarray = np.vstack(all_test_X)
    test_y: np.ndarray = np.concatenate(all_test_y)

    scaler: StandardScaler = StandardScaler()
    # Flatten to scale, then reshape back
    train_X_shape: Tuple = train_X.shape
    train_X = scaler.fit_transform(train_X.reshape(-1, 1)).reshape(train_X_shape)

    val_X_shape: Tuple = val_X.shape
    val_X = scaler.transform(val_X.reshape(-1, 1)).reshape(val_X_shape)  # pyright: ignore[reportAttributeAccessIssue]

    test_X_shape: Tuple = test_X.shape
    test_X = scaler.transform(test_X.reshape(-1, 1)).reshape(test_X_shape)  # pyright: ignore[reportAttributeAccessIssue]

    # Scale targets
    train_y = scaler.transform(train_y.reshape(-1, 1)).flatten()  # pyright: ignore[reportAttributeAccessIssue]
    val_y = scaler.transform(val_y.reshape(-1, 1)).flatten()  # pyright: ignore[reportAttributeAccessIssue]
    test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()  # pyright: ignore[reportAttributeAccessIssue]

    train_loader: DataLoader = DataLoader(
        StocksDataset(train_X, train_y),  # pyright: ignore[reportArgumentType]
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader: DataLoader = DataLoader(
        StocksDataset(val_X, val_y),  # pyright: ignore[reportArgumentType]
        batch_size=BATCH_SIZE,
    )
    test_loader: DataLoader = DataLoader(
        StocksDataset(test_X, test_y),  # pyright: ignore[reportArgumentType]
        batch_size=BATCH_SIZE,
    )
    return StocksDataLoaders(train_loader, val_loader, test_loader), scaler


def train_step(
    model: StockPredictionModel,
    train_loader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.Adam,
    epoch: int,
) -> float:
    model.train()
    train_loss: float = 0
    for X_train, y_train in tqdm(
        train_loader, desc=f"Number of Batches Left for Epoch - {epoch}"
    ):
        X_train = X_train.unsqueeze(-1)
        y_train = y_train.unsqueeze(-1)

        y_train_pred: torch.Tensor = model(X_train)
        loss: torch.Tensor = criterion(y_train_pred, y_train)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss


def val_step(model: StockPredictionModel, val_loader: DataLoader, epoch: int) -> float:
    model.eval()
    val_predictions: List = []
    val_targets: List = []
    with torch.no_grad():
        for X_val, y_val in tqdm(
            val_loader, desc=f"Number of Val Batches Left for Epoch - {epoch}"
        ):
            X_val = X_val.unsqueeze(-1)
            y_val = y_val.unsqueeze(-1)

            y_val_pred: torch.Tensor = model(X_val)

            val_predictions.append(y_val_pred.cpu())
            val_targets.append(y_val.cpu().reshape((-1, 1)))

    final_predictions_scaled: torch.Tensor = torch.vstack(val_predictions)
    final_targets_scaled: torch.Tensor = torch.vstack(val_targets)

    final_predictions: np.ndarray = scaler.inverse_transform(
        final_predictions_scaled.numpy()
    )
    final_targets: np.ndarray = scaler.inverse_transform(final_targets_scaled.numpy())

    val_rmse: float = root_mean_squared_error(final_targets, final_predictions)

    return val_rmse


if __name__ == "__main__":
    downloader: NASDAQDownloader = NASDAQDownloader()
    info: NASDAQDatasetInfo = downloader.download_dataset(stop_if_dest_dir_exists=True)

    data_loaders, scaler = prepare_data(info.stocks_directory)

    model: StockPredictionModel = StockPredictionModel(
        input_dim=1, hidden_dim=32, num_layers=2, output_dim=1, device=device
    ).to(device)
    criterion: nn.MSELoss = nn.MSELoss().to(device)
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.01)

    num_epochs: int = 10
    all_losses: List[float] = []
    all_rmse: List[float] = []
    for epoch in tqdm(range(num_epochs), desc="Number of Epochs Left"):
        losses: float = train_step(
            model, data_loaders.train, criterion, optimizer, epoch
        )
        all_losses.append(losses)

        rsme: float = val_step(model, data_loaders.val, epoch)
        all_rmse.append(rsme)
