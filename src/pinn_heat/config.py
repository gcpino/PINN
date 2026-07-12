from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HeatConfig:
    alpha: float = 0.01
    x_min: float = 0.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    hidden_layers: int = 3
    hidden_width: int = 32
    learning_rate: float = 1.0e-3
    epochs: int = 2_000
    n_collocation: int = 4_096
    n_initial: int = 512
    n_boundary: int = 512
    seed: int = 1234
    output_dir: str = "outputs"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

