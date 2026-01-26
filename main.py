import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from create_training_data import DatasetLoadingConfig, TrainingDataCreator
from download import (
    NASDAQDatasetCreationOptions,
    NASDAQDatasetInfo,
    NASDAQDownloader,
    SecurityType,
)
from model import StockPredictionModel
from parse_config import ModelConfig, get_config_from_json
from stocks import NASDAQDataLoaders

# ANSI escape codes
RED: str = "\033[31m"
GREEN: str = "\033[32m"
RESET: str = "\033[0m"

device: torch.device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")

# Saves resources by using a different backend for rendering graphs
matplotlib.use("Agg")


@dataclass
class CalculatedValMetrics:
    avg_loss: float
    rsme: float
    mae: float
    r2: float


def train_step(
    model: StockPredictionModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Adam,
    # Dynamically scales loss values, thus keeps the model stable when using AMP
    scaler: torch.GradScaler,
    epoch: int,
) -> float:
    model.train()

    train_loss: float = 0
    for X_train, stock_id, y_train in tqdm(
        train_loader, desc=f"Num Train Batches left for Epoch - {epoch}"
    ):
        X_train = X_train.to(device)
        stock_id = stock_id.to(device)
        y_train = y_train.to(device)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, cache_enabled=True
        ):
            predictions: torch.Tensor = model(X_train, stock_id)
            loss: torch.Tensor = criterion(predictions, y_train)

        # Scale loss up so gradients don't vanish
        scaler.scale(loss).backward()
        # Must unscale before clipping gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        optimizer.zero_grad()

    epoch_loss: float = train_loss / len(train_loader)
    print(f"\nCurrent Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate(
    model: StockPredictionModel,
    loader: DataLoader,
    criterion: nn.Module,
    desc: str = "Evaluating",
) -> CalculatedValMetrics:
    model.eval()

    total_samples: int = 0
    total_loss: float = 0.0

    all_targets: List[NDArray] = []
    all_predictions: List[NDArray] = []

    with torch.no_grad():
        for X, stock_id, y in tqdm(loader, desc=desc):
            X = X.to(device)
            stock_id = stock_id.to(device)
            y = y.to(device)

            predictions: torch.Tensor = model(X, stock_id)

            batch_loss: torch.Tensor = criterion(predictions, y)
            # Multiply by batch size to get total loss for the batch
            total_loss += batch_loss.item() * X.size(0)
            total_samples += X.size(0)

            all_targets.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss: float = total_loss / total_samples

    # Regression metrics
    rsme: float = root_mean_squared_error(all_targets, all_predictions)
    mae: float = mean_absolute_error(all_targets, all_predictions)
    r2: float = r2_score(all_targets, all_predictions)

    return CalculatedValMetrics(avg_loss, rsme, mae, r2)


def val_step(
    model: StockPredictionModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    epoch: int,
) -> CalculatedValMetrics:
    metrics: CalculatedValMetrics = evaluate(
        model,
        val_loader,
        criterion,
        desc=f"Num Val Batches Left for Epoch - {epoch}",
    )

    print(f"\nCurrent Val MAE: {metrics.mae:.6f}")
    print(f"Current Val R2: {metrics.r2:.4f}")
    print(f"Current Val Loss: {metrics.avg_loss:.6f}")

    scheduler.step(metrics.avg_loss)

    return metrics


# Gemini coded plotting
def save_and_plot_model_performance(
    model_config: ModelConfig,
    all_training_losses: List[float],
    all_val_metrics: List[CalculatedValMetrics],
) -> None:
    print("Saving model performance to CSV...")

    val_losses: List[float] = [m.avg_loss for m in all_val_metrics]
    val_maes: List[float] = [m.mae for m in all_val_metrics]
    val_r2s: List[float] = [m.r2 for m in all_val_metrics]
    model_performance_df: pl.DataFrame = pl.DataFrame(
        {
            "Training Loss": all_training_losses,
            "Val Loss": val_losses,
            "Val MAE": val_maes,
            "Val R2": val_r2s,
        }
    )
    model_performance_df.write_csv(
        model_config.model_info_dir / "model_performance.csv"
    )

    print("Saving model performance to graphs...")
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    color: str = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color=color)
    l1: List = ax1.plot(all_training_losses, color=color, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2: Axes = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Val Loss", color=color)
    l2: List = ax2.plot(val_losses, color=color, label="Val Loss")
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.set_title("Loss and Error Over Epochs")

    # Combining legends
    lines: List = l1 + l2
    labs: List = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc="upper right")

    ax3.plot(val_r2s, color="tab:green", label="R² Score")
    ax3.axhline(
        y=0.0, color="black", linestyle="--", alpha=0.5, label="Mean Baseline (0.0)"
    )
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("R² Score")
    ax3.set_title("Model Fit (R² Score)")
    ax3.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(model_config.model_info_dir / "baseline_model_performance.png")

    plt.close(fig)


def plot_model_test_performance(
    model: StockPredictionModel,
    test_dataset: Dataset,
    epoch: int,
) -> None:
    model.eval()

    sampler: RandomSampler = RandomSampler(test_dataset, replacement=False)  # pyright: ignore[reportArgumentType]
    dataloader: DataLoader = DataLoader(test_dataset, batch_size=2048, sampler=sampler)

    X_sample, stock_id, y_sample = next(iter(dataloader))
    X_sample = X_sample.to(device)
    stock_id = stock_id.to(device)
    y_sample = y_sample.to(device).float()

    with torch.no_grad():
        predictions: NDArray = model(X_sample, stock_id).cpu().numpy().flatten()
        targets: NDArray = y_sample.cpu().numpy().flatten()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    num_samples_to_plot: int = 100
    subset_targets: NDArray = targets[:num_samples_to_plot]
    subset_predictions: NDArray = predictions[:num_samples_to_plot]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        subset_targets,
        label="Actual Target",
        marker="o",
        linestyle="-",
        color="tab:blue",
        alpha=0.7,
    )

    ax.plot(
        subset_predictions,
        label="Model Prediction",
        marker="x",
        linestyle="--",
        color="tab:red",
        alpha=0.9,
    )

    for i in range(num_samples_to_plot):
        ax.vlines(
            x=i,
            ymin=min(subset_targets[i], subset_predictions[i]),
            ymax=max(subset_targets[i], subset_predictions[i]),
            colors="gray",
            linestyles=":",
            alpha=0.3,
        )

    ax.set_title(
        f"Actual vs Predicted Comparison (Epoch {epoch}) - First {num_samples_to_plot} Random Samples"
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Stock Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    debug_directory: Path = model.config.model_info_dir / "debug_predictions"
    os.makedirs(debug_directory, exist_ok=True)
    plt.savefig(debug_directory / f"epoch_{epoch}.png")
    plt.close()


def clean_dataset_directory(directory: Path) -> None:
    debug_directory: Path = directory / "debug_predictions"
    if debug_directory.exists():
        print(f"Cleaning {debug_directory}...")
        shutil.rmtree(debug_directory)

    model_directory: Path = directory / "models"
    if model_directory.exists():
        print(f"Cleaning {model_directory}...")
        shutil.rmtree(model_directory)

    model_performance_csv: Path = directory / "model_performance.csv"
    if model_performance_csv.exists():
        model_performance_csv.unlink()

    model_performance_plots: Path = directory / "baseline_model_performance.png"
    if model_performance_plots.exists():
        model_performance_plots.unlink()


def main(model_config: ModelConfig) -> None:
    clean_dataset_directory(model_config.model_info_dir)

    model_save_dir: Path = model_config.model_info_dir / "models"
    os.makedirs(model_save_dir, exist_ok=True)

    downloader: NASDAQDownloader = NASDAQDownloader(model_config)
    info: NASDAQDatasetInfo = downloader.download_dataset(
        save_directory="nasdaq_dataset",
        security_type=SecurityType.STOCK,
        dataset_creation_option=NASDAQDatasetCreationOptions.REUSE,
        target=model_config.num_training_files,
    )

    training_data_creator: TrainingDataCreator = TrainingDataCreator(model_config)
    data_loaders: NASDAQDataLoaders = training_data_creator.create_data_loaders_from(
        info.stocks_directory,
        model_config.model_info_dir,
        dataset_loading_config=DatasetLoadingConfig.REUSE,
    )

    # Get 'Close' column dynamically
    feature_names: List[str] = data_loaders.train.dataset.feature_cols  # pyright: ignore[reportAttributeAccessIssue]
    try:
        target_idx = feature_names.index("Close")
    except ValueError:
        raise ValueError(f"{RED}'Close' column not found in dataset features!{RESET}")

    model: StockPredictionModel = StockPredictionModel(
        input_dim=9,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.2,
        target_feature_idx=target_idx,
        embedding_dim=16,
        device=device,
        model_config=model_config,
    ).to(device)

    criterion: nn.MSELoss = nn.MSELoss().to(device)

    optimizer: optim.AdamW = optim.AdamW(
        model.parameters(),
        lr=model.config.learning_rate,
        weight_decay=1e-4,
    )

    scheduler: ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )
    scaler: torch.GradScaler = torch.GradScaler(device=device.type)

    all_training_losses: List[float] = []
    all_val_metrics: List[CalculatedValMetrics] = []

    best_loss: float = float("inf")
    counter: int = 0
    for epoch in tqdm(range(model.config.epochs), desc="Number of Epochs Left"):
        losses: float = train_step(
            model,
            data_loaders.train,
            criterion,
            optimizer,
            scaler,
            epoch,
        )
        all_training_losses.append(losses)

        val_metrics: CalculatedValMetrics = val_step(
            model,
            data_loaders.val,
            criterion,
            scheduler,
            epoch,
        )
        all_val_metrics.append(val_metrics)

        if val_metrics.avg_loss < best_loss:
            best_loss = val_metrics.avg_loss
            counter = 0
            torch.save(model.state_dict(), model_save_dir / "best_model.pt")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")
            if counter >= model.config.early_stopping_patience:
                print("Early stopping triggered!")
                break

        torch.save(model.state_dict(), model_save_dir / f"model_{epoch}.pt")

        save_and_plot_model_performance(
            model.config, all_training_losses, all_val_metrics
        )

        plot_model_test_performance(model, data_loaders.test.dataset, epoch)

    print("Model training complete!")

    print("=" * 30)
    print("Running final test evaluation...")
    print("=" * 30)

    best_model_path = model_save_dir / "best_model.pt"
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    test_metrics: CalculatedValMetrics = evaluate(
        model, data_loaders.test, criterion, desc="Testing Model"
    )

    print(f"\nFinal Test Loss: {test_metrics.avg_loss:.5f}")
    print(f"Final Test MAE: {test_metrics.mae:.5f}")
    print(f"Final Test R2: {test_metrics.r2:.4f}")

    # For r2 metric, 0.0 means guessing the mean, 1.0 means a perfect model
    if test_metrics.r2 > 0.0:
        print(f"{GREEN}SUCCESS: The model beats the baseline mean prediction!{RESET}")
    else:
        print(f"{RED}FAILURE: The model failed to generalize (R2 <= 0).{RESET}")


if __name__ == "__main__":
    model_config = get_config_from_json("lstm_model_config.json")  # pyright: ignore[reportAssignmentType]
    if model_config is not None:
        torch.manual_seed(model_config.random_seed)

        main(model_config)
