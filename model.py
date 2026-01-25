from typing import Self

import torch
import torch.nn as nn

from parse_config import ModelConfig


class StockPredictionModel(nn.Module):
    def __init__(
        self: Self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        target_feature_idx: int,
        device: torch.device,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.hidden_dim: int = hidden_dim
        self.target_feature_idx: int = target_feature_idx
        self.device: torch.device = device
        # This variable is used outside of this script
        self.config: ModelConfig = model_config

        self.lstm: nn.LSTM = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            device=device,
        )
        # Doing bidirectional = True doubles the amount of hidden units
        # This layer stabilizes/speeds up training by preventing vanishing/exploding gradients
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim, device=device)
        self.dropout: nn.Dropout = nn.Dropout(0.2)

        # Projection Head
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            # ReLU kills negative neurons, but having negative neurons could be valuable as they can be correlated to bearish trends
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, output_dim, device=device),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        # Doing instance normalization for each input tensor to force the model to generalize
        # This technique is called RevIN
        # x shape: [Batch, Sequence, Features]
        x_mean: torch.Tensor = x.mean(dim=1, keepdim=True)
        x_std: torch.Tensor = (
            x.std(dim=1, keepdim=True) + 1e-5
        )  # epsilon to avoid div/0

        x_norm: torch.Tensor = (x - x_mean) / x_std
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x_norm_flat: torch.Tensor = x_norm.view(batch_size, sequence_length, -1)

        out, _ = self.lstm(x_norm_flat)

        # Take the last time step, which is the standard for many-to-one tasks, because the last step should
        # theoretically contain all of the information of the previous steps
        out = out[:, -1, :]
        out = self.layer_norm(out)

        # We're doing binary classification now
        out_logits: torch.Tensor = self.fc(out)

        return out_logits
