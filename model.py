from typing import Self

import torch
import torch.nn as nn


class StockPredictionModel(nn.Module):
    def __init__(
        self: Self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.hidden_dim: int = hidden_dim
        self.device: torch.device = device

        self.lstm: nn.LSTM = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            device=device,
        )
        # Doing bidirectional = True doubles the amount of hidden units
        # This layer stabilizes/speeds up training by preventing vanishing/exploding gradients
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim, device=device)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.fc_1: nn.Linear = nn.Linear(hidden_dim, hidden_dim // 2, device=device)
        self.relu: nn.ReLU = nn.ReLU()
        self.fc_2: nn.Linear = nn.Linear(hidden_dim // 2, output_dim, device=device)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Doing instance normalization for each input tensor to force the model to generalize
        x_mean: torch.Tensor = x.mean(dim=1, keepdim=True)
        x_std: torch.Tensor = (
            x.std(dim=1, keepdim=True) + 1e-5
        )  # epsilon to avoid div/0

        x_norm: torch.Tensor = (x - x_mean) / x_std
        x_norm_flat: torch.Tensor = x_norm.view(batch_size, seq_len, -1)

        out, _ = self.lstm(x_norm_flat)

        out = out[:, -1, :]
        out = self.layer_norm(out)

        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        # Denormalize out and rescale to match the original magnitude
        target_std: torch.Tensor = x_std[:, :, 0]
        target_mean: torch.Tensor = x_mean[:, :, 0]
        out_rescaled: torch.Tensor = (out * target_std) + target_mean

        return out_rescaled


class DirectionalMSELoss(nn.Module):
    def __init__(self: Self, penalty_factor: float = 5.0):
        super().__init__()
        self.mse: nn.MSELoss = nn.MSELoss()
        self.penalty_factor: float = penalty_factor

    def forward(self: Self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse_loss: torch.Tensor = self.mse(y_pred, y_true)

        # - (pred * true) = positive when signs of pred & true are different, causing ReLU to be +
        direction_penalty: torch.Tensor = torch.mean(torch.relu(-y_pred * y_true))

        return mse_loss + (self.penalty_factor * direction_penalty)
