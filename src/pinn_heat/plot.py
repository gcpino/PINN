from pathlib import Path

import torch
from torch import nn

from pinn_heat.config import HeatConfig
from pinn_heat.physics import analytical_solution


@torch.no_grad()
def save_comparison_plot(model: nn.Module, config: HeatConfig, path: Path, grid_size: int = 120) -> None:
    import matplotlib.pyplot as plt

    device = config.device
    x = torch.linspace(config.x_min, config.x_max, grid_size, device=device)
    t = torch.linspace(config.t_min, config.t_max, grid_size, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    x_flat = xx.reshape(-1, 1)
    t_flat = tt.reshape(-1, 1)

    prediction = model(torch.cat([x_flat, t_flat], dim=1)).reshape(grid_size, grid_size).cpu()
    target = analytical_solution(x_flat, t_flat, config.alpha).reshape(grid_size, grid_size).cpu()
    error = torch.abs(prediction - target)

    extent = [config.t_min, config.t_max, config.x_min, config.x_max]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    panels = [
        (prediction, "PINN prediction"),
        (target, "Analytical solution"),
        (error, "Absolute error"),
    ]
    for axis, (values, title) in zip(axes, panels):
        image = axis.imshow(values, origin="lower", aspect="auto", extent=extent)
        axis.set_title(title)
        axis.set_xlabel("t")
        axis.set_ylabel("x")
        fig.colorbar(image, ax=axis)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)

