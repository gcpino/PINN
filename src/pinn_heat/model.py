import torch
from torch import nn


class MLP(nn.Module):
    """Small fully connected network that maps (x, t) to u(x, t)."""

    def __init__(self, hidden_layers: int = 3, hidden_width: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.net(x_t)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

