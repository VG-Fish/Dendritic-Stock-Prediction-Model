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
        # This layer stabilizes/speeds up training by preventing vanishing/exploding gradients
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim, device=device)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.fc: nn.Linear = nn.Linear(
            hidden_dim,
            output_dim,
            device=device,
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        # We must turn 4d tensor to 3d tensor by flattening all the features into one tensor
        x = x.view(batch_size, seq_len, -1)

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.layer_norm(out)
        out = self.fc(self.dropout(out))

        return out


class DirectionalMSELoss(nn.Module):
    def __init__(self: Self, penalty_factor: float = 5.0):
        super().__init__()
        self.mse: nn.MSELoss = nn.MSELoss()
        self.penalty_factor: float = penalty_factor

    def forward(self: Self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse_loss: float = self.mse(y_pred, y_true)

        # - (pred * true) = positive when signs of pred & true are different, causing ReLU to be +
        direction_penalty: torch.Tensor = torch.mean(torch.relu(-y_pred * y_true))

        return mse_loss + (self.penalty_factor * direction_penalty)
