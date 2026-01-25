import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
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
    for X_train, y_train in tqdm(
        train_loader, desc=f"Num Train Batches left for Epoch - {epoch}"
    ):
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, cache_enabled=True
        ):
            logits: torch.Tensor = model(X_train)
            loss: torch.Tensor = criterion(logits, y_train)

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
    epoch: Optional[int] = None,
    save_model_classification_report: bool = True,
    desc: str = "Evaluating",
) -> Tuple[float, float]:
    model.eval()

    correct: int = 0
    total_samples: int = 0
    total_loss: float = 0.0

    all_targets: List[NDArray] = []
    all_predictions: List[NDArray] = []

    with torch.no_grad():
        for X, y in tqdm(loader, desc=desc):
            X = X.to(device)
            y = y.to(device)

            # Gemini suggested z-score stuff
            logits: torch.Tensor = model(X)

            batch_loss: torch.Tensor = criterion(logits, y)
            # Multiply by batch size to get total loss for the batch
            total_loss += batch_loss.item() * X.size(0)

            probabilities: torch.Tensor = torch.sigmoid(logits)
            # If prob > 0.5, predict 1 (Up), else 0 (Down)
            predictions: torch.Tensor = (probabilities > 0.5).float()
            correct += (predictions == y).sum().item()
            total_samples += y.size(0)

            all_targets.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss: float = total_loss / total_samples
    accuracy: float = correct / total_samples

    if not save_model_classification_report:
        return avg_loss, accuracy

    report: Dict = classification_report(  # pyright: ignore[reportAssignmentType]
        all_targets,
        all_predictions,
        target_names=["Down", "Up"],
        output_dict=True,
        digits=4,
    )
    classification_report_directory: Path = (
        model.config.model_info_dir / "classification_reports"
    )
    os.makedirs(classification_report_directory, exist_ok=True)
    ending: str = f"_{epoch}" if epoch else ""
    with open(
        classification_report_directory / f"report{ending}.json",
        "w",
    ) as f:
        json.dump(report, f, indent=2)

    return avg_loss, accuracy


def val_step(
    model: StockPredictionModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    save_model_classification_report: bool = True,
) -> Tuple[float, float]:
    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        epoch,
        save_model_classification_report=save_model_classification_report,
        desc=f"Num Val Batches Left for Epoch - {epoch}",
    )

    print(f"\nCurrent Accuracy: {val_accuracy:.3%}")
    print(f"Current Val Loss: {val_loss}")

    scheduler.step(val_loss)

    return val_loss, val_accuracy


def save_and_plot_model_performance(
    model_config: ModelConfig,
    all_training_losses: List[float],
    all_val_losses: List[float],
    all_accuracies: List[float],
) -> None:
    print("Saving model performance to CSV...")
    model_performance_df: pl.DataFrame = pl.DataFrame(
        {
            "Training Loss": all_training_losses,
            "Val Loss": all_val_losses,
            "Dimensional Accuracy": all_accuracies,
        }
    )
    model_performance_df.write_csv(
        model_config.model_info_dir / "model_performance.csv"
    )

    print("Saving model performance to graphs...")
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color=color)
    l1 = ax1.plot(all_training_losses, color=color, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Val Loss", color=color)
    l2 = ax2.plot(all_val_losses, color=color, label="Val Loss")
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.set_title("Loss and Error Over Epochs")

    # Combining legends
    lines = l1 + l2
    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc="upper right")

    ax3.plot(all_accuracies, color="tab:green", label="Accuracy")
    ax3.axhline(
        y=0.50, color="black", linestyle="--", alpha=0.5, label="Random Guess (50%)"
    )
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Model Accuracy")
    ax3.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(model_config.model_info_dir / "baseline_model_performance.png")

    plt.close(fig)


def plot_model_test_performance(
    model: StockPredictionModel,
    test_dataset: Dataset,
    epoch: int,
) -> None:
    model.eval()

    sampler = RandomSampler(test_dataset, replacement=False)  # pyright: ignore[reportArgumentType]
    dataloader = DataLoader(test_dataset, batch_size=2048, sampler=sampler)

    X_sample, y_sample = next(iter(dataloader))
    X_sample, y_sample = X_sample.to(device), y_sample.to(device)

    with torch.no_grad():
        logits = model(X_sample)
        # Convert logits to probabilities (0.0 to 1.0)
        probabilities: NDArray = torch.sigmoid(logits).cpu().numpy().flatten()
        targets: NDArray = y_sample.cpu().numpy().flatten()
        predictions: NDArray = (probabilities > 0.5).astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Green (up) bars should ideally be clustered tightly on the right.
    # Red (down) bars clustered tightly on the left
    ax1.hist(
        probabilities[targets == 1],
        bins=50,
        alpha=0.6,
        color="green",
        label="Actual: UP",
        range=(0, 1),
    )
    ax1.hist(
        probabilities[targets == 0],
        bins=50,
        alpha=0.6,
        color="red",
        label="Actual: DOWN",
        range=(0, 1),
    )
    ax1.axvline(0.5, color="black", linestyle="--", alpha=0.3, label="Threshold")
    ax1.set_title(f"Prediction Confidence Distribution (Epoch {epoch})")
    ax1.set_xlabel("Predicted Probability (0=Down, 1=Up)")
    ax1.set_ylabel("Count")
    ax1.legend()

    cm: NDArray = confusion_matrix(targets, predictions, normalize="all")
    display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Down", "Up"]
    )
    display.plot(cmap="Blues", ax=ax2, colorbar=True)
    threshold: float = cm.max() / 2.0

    # Add labels to confusion matrix
    text_labels: List[Tuple[int, int, str]] = [
        (0, 0, f"True Neg\n{cm[0, 0]}"),
        (0, 1, f"False Neg\n{cm[0, 1]}"),
        (1, 0, f"False Pos\n{cm[1, 0]}"),
        (1, 1, f"True Pos\n{cm[1, 1]}"),
    ]

    # Clear existing text (optional, or just draw over it)
    for text in display.text_.ravel():  # pyright: ignore[reportOptionalMemberAccess]
        text.set_text("")

    for x, y, label in text_labels:
        color = "white" if cm[y, x] > threshold else "black"
        ax2.text(x, y, label, ha="center", va="center", color=color, fontweight="bold")

    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    plt.tight_layout()
    debug_directory = model.config.model_info_dir / "debug_predictions"
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

    classification_report_directory: Path = directory / "classification_reports"
    if classification_report_directory.exists():
        print(f"Cleaning {classification_report_directory}...")
        shutil.rmtree(classification_report_directory)

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
        dropout=0.3,
        target_feature_idx=target_idx,
        device=device,
        model_config=model_config,
    ).to(device)

    # This helps the model adjust to class imbalance in the dataset dynamically
    counts: pl.DataFrame = training_data_creator.counts.sort("Target")
    negative_count: int = counts["count"][0]
    positive_count: int = counts["count"][1]
    weight_value: float = negative_count / positive_count
    positive_weight: torch.Tensor = torch.tensor([weight_value]).to(device)
    criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    ).to(device)

    optimizer: optim.Adam = optim.Adam(
        model.parameters(),
        lr=model.config.learning_rate,
        weight_decay=1e-5,
    )

    scheduler: ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )
    scaler: torch.GradScaler = torch.GradScaler(device=device.type)

    all_training_losses: List[float] = []
    all_val_losses: List[float] = []
    all_accuracies: List[float] = []

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

        loss, accuracy = val_step(
            model,
            data_loaders.val,
            criterion,
            scheduler,
            epoch,
            save_model_classification_report=True,
        )
        all_val_losses.append(loss)
        all_accuracies.append(accuracy)

        if loss < best_loss:
            best_loss = loss
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
            model.config, all_training_losses, all_val_losses, all_accuracies
        )

        plot_model_test_performance(model, data_loaders.test.dataset, epoch)

    print("Model training complete!")

    print("\n" + "=" * 30)
    print("Running final test evaluation...")
    print("=" * 30)

    best_model_path = model_save_dir / "best_model.pt"
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    test_rmse, test_acc = evaluate(
        model, data_loaders.test, criterion, desc="Testing Model"
    )

    print(f"\nFinal Test Loss: {test_rmse:.5f}")
    print(f"Final Test Accuracy: {test_acc:.3%}")

    if test_acc > 0.5:
        print(f"{GREEN}SUCCESS: The model beats random guessing!{RESET}")
    else:
        print(f"{RED}FAILURE: The model failed to generalize.{RESET}")


if __name__ == "__main__":
    model_config = get_config_from_json("lstm_model_config.json")  # pyright: ignore[reportAssignmentType]
    if model_config is not None:
        torch.manual_seed(model_config.random_seed)

        main(model_config)
