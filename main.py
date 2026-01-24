import os
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from create_training_data import TrainingDataCreator
from download import (
    NASDAQDatasetCreationOptions,
    NASDAQDatasetInfo,
    NASDAQDownloader,
    SecurityType,
)
from model import DirectionalMSELoss, StockPredictionModel
from parse_config import ModelConfig, get_config_from_json
from stocks import NASDAQDataLoaders

model_config: ModelConfig

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


def train_step(
    model: StockPredictionModel,
    train_loader: DataLoader,
    criterion: nn.Module,
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

        # This adds mixed amp precision for faster training
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # Gemini suggested outputs for stable Z-score
            pred_norm, mean, std = model(X_train)
            y_train_norm: torch.Tensor = (y_train - mean) / std
            loss: torch.Tensor = criterion(pred_norm, y_train_norm)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        # This prevents model weights from exploding and turning into NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    epoch_loss: float = train_loss / len(train_loader)
    print(f"\nCurrent Training Loss: {epoch_loss}")

    return epoch_loss


def evaluate(
    model: StockPredictionModel,
    loader: DataLoader,
    desc: str = "Evaluating",
) -> Tuple[float, float]:
    model.eval()
    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for X, y in tqdm(loader, desc=desc):
            X = X.to(device)
            y = y.to(device)

            # Gemini suggested z-score stuff
            pred_norm, mean, std = model(X)
            pred_real: torch.Tensor = (pred_norm * std) + mean

            all_predictions.append(pred_real.cpu())
            all_targets.append(y.cpu())

    final_predictions: torch.Tensor = torch.vstack(all_predictions)
    final_targets: torch.Tensor = torch.vstack(all_targets)

    dim_correct: torch.Tensor = torch.sign(final_predictions) == torch.sign(
        final_targets
    )
    dim_accuracy: float = torch.mean(dim_correct.float()).item()
    rmse: float = torch.sqrt(
        torch.nn.functional.mse_loss(final_targets, final_predictions)
    ).item()

    return rmse, dim_accuracy


def val_step(
    model: StockPredictionModel,
    val_loader: DataLoader,
    scheduler: ReduceLROnPlateau,
    epoch: int,
) -> Tuple[float, float]:
    val_rmse, dim_accuracy = evaluate(model, val_loader, desc=f"Val Epoch {epoch}")

    print(f"\nCurrent Dimensional Accuracy: {dim_accuracy:.3%}")
    print(f"Current RMSE: {val_rmse}")

    scheduler.step(val_rmse)

    return val_rmse, dim_accuracy


def plot_model_performance(
    model_config: ModelConfig,
    all_losses: List[float],
    all_rsme: List[float],
    all_dim_accuracies: List[float],
) -> None:
    print("Saving model performance...")

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color=color)
    l1 = ax1.plot(all_losses, color=color, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Val RMSE (Log Returns)", color=color)
    l2 = ax2.plot(all_rsme, color=color, label="Val RMSE")
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.set_title("Loss and Error Over Epochs")

    # Combining legends
    lines = l1 + l2
    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc="upper right")

    ax3.plot(all_dim_accuracies, color="tab:green", label="Directional Accuracy")
    ax3.axhline(
        y=0.50, color="black", linestyle="--", alpha=0.5, label="Random Guess (50%)"
    )
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Model Directional Accuracy")
    ax3.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(model_config.model_info_dir / "baseline_model_performance.png")

    plt.close(fig)


def plot_model_predictions_over_targets(
    model: StockPredictionModel,
    model_config: ModelConfig,
    test_dataset: Dataset,
    epoch: int,
) -> None:
    model.eval()

    sampler: RandomSampler = RandomSampler(
        test_dataset,  # pyright: ignore[reportArgumentType]
        replacement=False,
    )
    dataloader_with_sampler: DataLoader = DataLoader(
        test_dataset, batch_size=model_config.batch_size, sampler=sampler
    )
    X_sample, y_sample = next(iter(dataloader_with_sampler))

    with torch.no_grad():
        predictions_normalized, mean, std = model(X_sample.to(device))
        # Denormalize the prediction (Z-Score -> Actual Log Return)
        predictions_real: torch.Tensor = (predictions_normalized * std) + mean
        predictions: torch.Tensor = predictions_real.cpu()

    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:100], label="Predictions")
    plt.plot(y_sample[:100], label="Actual Target", alpha=0.5)
    plt.title("Predictions vs Actuals")
    plt.legend()

    debug_directory: Path = model_config.model_info_dir / "debug_predictions"
    os.makedirs(debug_directory, exist_ok=True)
    plt.savefig(debug_directory / f"epoch_{epoch}.png")

    plt.close()


def clean_debug_predictions() -> None:
    debug_directory: Path = model_config.model_info_dir / "debug_predictions"
    if debug_directory.exists():
        print(f"Cleaning {debug_directory}...")
        shutil.rmtree(debug_directory)


def main(model_config: ModelConfig) -> None:
    model_save_dir: Path = model_config.model_info_dir / "models"
    os.makedirs(model_save_dir, exist_ok=True)
    clean_debug_predictions()

    downloader: NASDAQDownloader = NASDAQDownloader()
    info: NASDAQDatasetInfo = downloader.download_dataset(
        save_directory="nasdaq_dataset",
        security_type=SecurityType.STOCK,
        dataset_creation_option=NASDAQDatasetCreationOptions.REPLACE,
        target=500,
    )

    training_data_creator: TrainingDataCreator = TrainingDataCreator(model_config)
    data_loaders: NASDAQDataLoaders = training_data_creator.create_data_loaders_from(
        info.stocks_directory,
        model_config.model_info_dir,
        load_datasets_from_memory=model_config.load_dataset_from_memory,
    )

    # Get close column dynamically
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
        dropout=0.5,
        target_feature_idx=target_idx,
        device=device,
    ).to(device)
    # huber_loss: nn.HuberLoss = nn.HuberLoss(reduction="mean").to(device)
    directional_mse_loss: DirectionalMSELoss = DirectionalMSELoss(
        penalty_factor=5.0
    ).to(device)
    optimizer: optim.Adam = optim.Adam(
        model.parameters(),
        lr=model_config.learning_rate,
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
    patience: int = 20
    counter: int = 0
    # loss_epoch_switch: int = 20
    for epoch in tqdm(range(model_config.epochs), desc="Number of Epochs Left"):
        losses: float = train_step(
            model,
            data_loaders.train,
            directional_mse_loss,
            optimizer,
            epoch,
        )
        all_losses.append(losses)

        rsme, dim_accuracy = val_step(model, data_loaders.val, scheduler, epoch)
        all_rmse.append(rsme)
        all_dim_accuracies.append(dim_accuracy)

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

        plot_model_performance(model_config, all_losses, all_rmse, all_dim_accuracies)
        plot_model_predictions_over_targets(
            model, model_config, data_loaders.test.dataset, epoch
        )

    print("Model training complete!")

    print("\n" + "=" * 30)
    print("Running final test evaluation...")
    print("=" * 30)

    best_model_path = model_save_dir / "best_model.pt"
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    test_rmse, test_acc = evaluate(model, data_loaders.test, desc="Testing model")

    print(f"\nFinal Test RMSE: {test_rmse:.5f}")
    print(f"Final Test Directional Accuracy: {test_acc:.3%}")

    if test_acc > 0.5:
        print(f"{GREEN}SUCCESS: The model beats random guessing!{RESET}")
    else:
        print(f"{RED}FAILURE: The model failed to generalize.{RESET}")


if __name__ == "__main__":
    model_config = get_config_from_json("lstm_model_config.json")  # pyright: ignore[reportAssignmentType]
    torch.manual_seed(model_config.random_seed)

    main(model_config)
