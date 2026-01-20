import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from perforatedai import globals_perforatedai as GPA
from perforatedai import library_perforatedai as LPA
from perforatedai import utils_perforatedai as UPA
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from download import NASDAQDatasetInfo, NASDAQDownloader
from main import parse_args, plot_model_performance, prepare_data
from model import StockPredictionModel

# Initialize important variables
RANDOM_SEED: int = 1290
SEQUENCE_LENGTH: int = 30
TRAIN_FRACTION: float = 0.8
VAL_FRACTION: float = 0.1
BATCH_SIZE: int = 256
LEARNING_RATE: float = 0.0005
EPOCHS: int = 10
MODEL_INFO_DIR: Path = Path("model_info")


torch.manual_seed(RANDOM_SEED)
device: torch.device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")


# Clipped train step suggested by Google Gemini
def clipped_train_step(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.Adam,
    epoch: int,
) -> float:
    model.train()
    train_loss: float = 0

    # Enable anomaly detection to find where gradients die
    torch.autograd.set_detect_anomaly(True)

    for X_train, y_train in tqdm(
        train_loader, desc=f"Number of Train Batches Left for Epoch - {epoch}"
    ):
        X_train = X_train.unsqueeze(-1).to(device)
        y_train = y_train.unsqueeze(-1).to(device)

        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    return train_loss / len(train_loader)


def main() -> None:
    model_save_dir: Path = MODEL_INFO_DIR / "models"
    os.makedirs(model_save_dir, exist_ok=True)

    downloader: NASDAQDownloader = NASDAQDownloader()
    info: NASDAQDatasetInfo = downloader.download_dataset(stop_if_dest_dir_exists=True)

    data_loaders, scaler = prepare_data(info.stocks_directory)

    model: StockPredictionModel = StockPredictionModel(
        input_dim=1,
        hidden_dim=64,
        num_layers=1,
        output_dim=1,
        dropout=0.2,
        device=device,
    ).to(device)

    UPA.initialize_pai(
        model,
        save_name=str(MODEL_INFO_DIR),
        maximizing_score=False,  # We're trying to minimize the loss
    )
    model.lstm.set_this_output_dimensions([-1, -1, 0])  # pyright: ignore[reportCallIssue]
    model.layer_norm.set_this_output_dimensions([-1, 0])  # pyright: ignore[reportCallIssue]
    model.fc.set_this_output_dimensions([-1, 0])  # pyright: ignore[reportCallIssue]

    criterion: nn.MSELoss = nn.MSELoss().to(device)

    optimArgs: Dict = {
        "params": model.parameters(),
        "lr": LEARNING_RATE,
    }
    GPA.pai_tracker.set_optimizer(optim.Adam)

    schedArgs: Dict = {
        "mode": "min",
        "patience": 5,
        "factor": 0.5,
        "threshold": 0.001,
    }
    GPA.pai_tracker.set_scheduler(optim.lr_scheduler.ReduceLROnPlateau)

    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(
        model, optimArgs, schedArgs
    )

    all_losses: List[float] = []
    all_rmse: List[float] = []
    all_dim_accuracies: List[float] = []
    epochs: int = -1
    while True:
        epochs += 1

        loss: float = clipped_train_step(
            model, data_loaders.train, criterion, optimizer, epochs
        )
        all_losses.append(loss)
        GPA.pai_tracker.add_extra_score(loss, "Train Loss")

        print()

        # Validation step
        model.eval()
        val_predictions: List[torch.Tensor] = []
        val_targets: List[torch.Tensor] = []
        dim_accuracy: float = 0.0
        with torch.no_grad():
            for X_val, y_val in tqdm(
                data_loaders.val,
                desc=f"Number of Val Batches Left for Epoch - {epochs}",
            ):
                X_val = X_val.unsqueeze(-1).to(device)
                y_val = y_val.unsqueeze(-1)

                y_val_pred: torch.Tensor = model(X_val)

                val_predictions.append(y_val_pred.cpu())
                val_targets.append(y_val)

        final_predictions_scaled: torch.Tensor = torch.vstack(val_predictions)
        final_targets_scaled: torch.Tensor = torch.vstack(val_targets)

        final_predictions: np.ndarray = scaler.inverse_transform(
            final_predictions_scaled.numpy()
        )
        final_targets: np.ndarray = scaler.inverse_transform(
            final_targets_scaled.numpy()
        )

        dim_correct: np.ndarray = np.sign(final_predictions) == np.sign(final_targets)
        dim_accuracy = np.mean(dim_correct).item()

        val_rmse: float = root_mean_squared_error(final_targets, final_predictions)
        GPA.pai_tracker.add_extra_score(val_rmse, "Val RSME")

        val_loss: float = criterion(
            final_predictions_scaled.to(device), final_targets_scaled.to(device)
        ).item()

        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_loss, model
        )
        model.to(device)
        if training_complete:
            print("PAI Training Complete.")
            break
        elif restructured:
            print("Model restructured. Adding dendrites and resetting optimizer...")
            model.to(device)

            optimArgs = {
                "params": model.parameters(),
                "lr": LEARNING_RATE,
            }
            schedArgs = {
                "mode": "min",
                "patience": 5,
                "factor": 0.5,
                "threshold": 0.001,
            }

            optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(
                model, optimArgs, schedArgs
            )

        all_rmse.append(val_rmse)
        all_dim_accuracies.append(dim_accuracy)

        print()

        torch.save(model.state_dict(), model_save_dir / f"model_{epochs}.pt")
    plot_model_performance(all_losses, all_rmse, all_dim_accuracies)
    print("Model training complete!")


if __name__ == "__main__":
    load_dotenv()

    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_cap_at_n(True)
    GPA.pc.set_improvement_threshold(1e-5)

    GPA.pc.append_modules_to_convert([nn.LSTM])
    GPA.pc.append_modules_to_convert([nn.LayerNorm])
    GPA.pc.append_module_names_with_processing(["LSTM"])
    # This processor lets the dendrites keep track of their own hidden state
    GPA.pc.append_module_by_name_processing_classes([LPA.LSTMProcessor])

    args: Namespace = parse_args()

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
