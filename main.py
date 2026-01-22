import argparse
import os
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.axes import Axes
from sklearn.metrics import root_mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from create_training_data import NormalizationData, create_data_loaders_from
from download import NASDAQDatasetInfo, NASDAQDownloader
from model import DirectionalMSELoss, StockPredictionModel

# Initialize important variables
RANDOM_SEED: int = 1290
SEQUENCE_LENGTH: int = 30
TRAIN_FRACTION: float = 0.8
VAL_FRACTION: float = 0.1
BATCH_SIZE: int = 256
LEARNING_RATE: float = 0.0002
EPOCHS: int = 200  # Early stopping will most likely be triggered before this is reached
MODEL_INFO_DIR: Path = Path("dimensional_bidirectional_lstm_model_info")

# ANSI escape codes
RED: str = "\033[31m"
RESET: str = "\033[0m"

torch.manual_seed(RANDOM_SEED)
device: torch.device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")


def inverse_transform(data: np.ndarray, stats: NormalizationData) -> np.ndarray:
    return (data * stats.std) + stats.mean


def train_step(
    model: StockPredictionModel,
    train_loader: DataLoader,
    criterion: DirectionalMSELoss,
    optimizer: optim.Adam,
    epoch: int,
) -> float:
    model.train()
    train_loss: float = 0
    for X_train, y_train in tqdm(
        train_loader, desc=f"Number of Train Batches Left for Epoch - {epoch}"
    ):
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        y_train_pred: torch.Tensor = model(X_train)
        loss: torch.Tensor = criterion(y_train_pred, y_train)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        # This prevents model weights from exploding and turning into NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    epoch_loss: float = train_loss / len(train_loader)
    print(f"\nCurrent Training Loss: {epoch_loss}")

    return epoch_loss


def val_step(
    model: StockPredictionModel,
    val_loader: DataLoader,
    price_stats: NormalizationData,
    scheduler: ReduceLROnPlateau,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    val_predictions: List[torch.Tensor] = []
    val_targets: List[torch.Tensor] = []
    dim_accuracy: float = 0.0
    with torch.no_grad():
        for X_val, y_val in tqdm(
            val_loader, desc=f"Number of Val Batches Left for Epoch - {epoch}"
        ):
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            y_val_pred: torch.Tensor = model(X_val)

            val_predictions.append(y_val_pred.cpu())
            val_targets.append(y_val.cpu())

    final_predictions_scaled: torch.Tensor = torch.vstack(val_predictions)
    final_targets_scaled: torch.Tensor = torch.vstack(val_targets)

    final_predictions: np.ndarray = inverse_transform(
        final_predictions_scaled.numpy(), price_stats
    )
    final_targets: np.ndarray = inverse_transform(
        final_targets_scaled.numpy(), price_stats
    )

    dim_correct: np.ndarray = np.sign(final_predictions) == np.sign(final_targets)
    dim_accuracy = np.mean(dim_correct).item()
    print(f"\nCurrent Dimensional Accuracy: {dim_accuracy:.3f}%")

    val_rmse: float = root_mean_squared_error(final_targets, final_predictions)
    print(f"\nCurrent RSME: {val_rmse}")

    scheduler.step(val_rmse)

    return val_rmse, dim_accuracy


def plot_model_performance(
    all_losses: List[float], all_rsme: List[float], all_dim_accuracies: List[float]
) -> None:
    print("Saving model performance...")

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss (MSE)", color=color)
    ax1.plot(all_losses, color=color, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2: Axes = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Val RMSE (Log Returns)", color=color)
    ax2.plot(all_rsme, color=color, label="Val RMSE")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.set_title("Loss and Error Over Epochs")
    plt.legend()

    ax3.plot(all_dim_accuracies, color="tab:green", label="Directional Accuracy")
    ax3.axhline(
        y=0.50, color="black", linestyle="--", alpha=0.5, label="Random Guess (50%)"
    )
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Model Directional Accuracy")
    plt.legend()

    fig.tight_layout()
    plt.savefig(MODEL_INFO_DIR / "baseline_model_performance.png")


def main() -> None:
    model_save_dir: Path = MODEL_INFO_DIR / "models"
    os.makedirs(model_save_dir, exist_ok=True)

    downloader: NASDAQDownloader = NASDAQDownloader()
    info: NASDAQDatasetInfo = downloader.download_dataset(stop_if_dest_dir_exists=True)

    data_loaders, price_stats = create_data_loaders_from(
        info.stocks_directory, load_datasets_from_memory=True
    )

    model: StockPredictionModel = StockPredictionModel(
        input_dim=8,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
        device=device,
    ).to(device)
    criterion: DirectionalMSELoss = DirectionalMSELoss(penalty_factor=2.0).to(device)
    optimizer: optim.Adam = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,  # Weight decay penalizes large weights
    )
    scheduler: ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    all_losses: List[float] = []
    all_rmse: List[float] = []
    all_dim_accuracies: List[float] = []

    best_val_rmse: float = float("inf")
    patience: int = 15
    counter: int = 0
    for epoch in tqdm(range(EPOCHS), desc="Number of Epochs Left"):
        losses: float = train_step(
            model, data_loaders.train, criterion, optimizer, epoch
        )
        all_losses.append(losses)

        rsme, dim_accuray = val_step(
            model, data_loaders.val, price_stats, scheduler, epoch
        )
        all_rmse.append(rsme)
        all_dim_accuracies.append(dim_accuray)

        if rsme < best_val_rmse:
            best_val_rmse = rsme
            counter = 0
            torch.save(model.state_dict(), model_save_dir / "best_model.pt")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")
            if counter >= patience:
                print("Early stopping triggered!")
                break

        torch.save(model.state_dict(), model_save_dir / f"model_{epoch}.pt")
        plot_model_performance(all_losses, all_rmse, all_dim_accuracies)
    plot_model_performance(all_losses, all_rmse, all_dim_accuracies)
    print("Model training complete!")


def float_in_range(low: float, high: float) -> Callable:
    def checker(value: str) -> float:
        try:
            f_value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{RED}'{value}' is not a valid float.{RESET}"
            )

        f_value = float(value)
        if f_value < low or f_value >= high:
            raise argparse.ArgumentTypeError(
                f"{RED}Value must be in [{low}, {high}){RESET}"
            )
        return f_value

    return checker


def parse_args() -> argparse.Namespace:
    global \
        RANDOM_SEED, \
        SEQUENCE_LENGTH, \
        TRAIN_FRACTION, \
        VAL_FRACTION, \
        BATCH_SIZE, \
        LEARNING_RATE, \
        EPOCHS, \
        MODEL_INFO_DIR

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Dendritic LSTM Stock Prediction Model",
        description="This program trains a Dendritic LSTM Stock Prediction Model on data from thousands of companies from the NASDAQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_seed", help="Set the random seed.", type=int, default=RANDOM_SEED
    )
    parser.add_argument(
        "--sequence_length",
        help="How many days the model should consider to predict the price for the next day.",
        type=int,
        default=SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--train_fraction",
        help="How big the training dataset should be. This parameter should be in [0, 1).",
        type=float_in_range(0.0, 1.0),
        default=TRAIN_FRACTION,
    )
    parser.add_argument(
        "--val_fraction",
        help="How big the validation dataset should be. This parameter should be in [0, 1).",
        type=float_in_range(0.0, 1.0),
        default=VAL_FRACTION,
    )
    parser.add_argument(
        "--batch_size",
        help="How big each batch size should be. This parameter should ideally be a power of 2.",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--learning_rate",
        help="The model learning rate.",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--epochs",
        help="The number of epochs.",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--model_info_dir",
        help="The directory to save all model related stuff to.",
        type=str,
        default=MODEL_INFO_DIR,
    )
    args: argparse.Namespace = parser.parse_args()

    if args.train_fraction + args.val_fraction >= 1.0:
        parser.error(
            f"{RED}The sum of --train_fraction ({args.train_fraction}) and "
            f"--val_fraction ({args.val_fraction}) must be less than 1.0 "
            f"to leave room for the test set.{RESET}"
        )
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()

    RANDOM_SEED = args.random_seed
    torch.manual_seed(RANDOM_SEED)
    SEQUENCE_LENGTH = args.sequence_length
    TRAIN_FRACTION = args.train_fraction
    VAL_FRACTION = args.val_fraction
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    MODEL_INFO_DIR = Path(args.model_info_dir)

    main()
