import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pinn_heat.config import HeatConfig
from pinn_heat.model import MLP
from pinn_heat.physics import analytical_solution, heat_residual, initial_condition


def sample_uniform(
    n: int,
    lower: float,
    upper: float,
    device: torch.device,
) -> torch.Tensor:
    return lower + (upper - lower) * torch.rand((n, 1), device=device)


def pinn_losses(model: nn.Module, config: HeatConfig) -> dict[str, torch.Tensor]:
    device = config.device

    x_f = sample_uniform(config.n_collocation, config.x_min, config.x_max, device)
    t_f = sample_uniform(config.n_collocation, config.t_min, config.t_max, device)
    residual = heat_residual(model, x_f, t_f, config.alpha)
    pde_loss = torch.mean(residual**2)

    x_i = sample_uniform(config.n_initial, config.x_min, config.x_max, device)
    t_i = torch.full_like(x_i, config.t_min)
    u_i_pred = model(torch.cat([x_i, t_i], dim=1))
    u_i_true = initial_condition(x_i)
    initial_loss = torch.mean((u_i_pred - u_i_true) ** 2)

    t_b = sample_uniform(config.n_boundary, config.t_min, config.t_max, device)
    x_left = torch.full_like(t_b, config.x_min)
    x_right = torch.full_like(t_b, config.x_max)
    u_left = model(torch.cat([x_left, t_b], dim=1))
    u_right = model(torch.cat([x_right, t_b], dim=1))
    boundary_loss = torch.mean(u_left**2) + torch.mean(u_right**2)

    total = pde_loss + initial_loss + boundary_loss
    return {
        "total": total,
        "pde": pde_loss,
        "initial": initial_loss,
        "boundary": boundary_loss,
    }


@torch.no_grad()
def validation_mse(model: nn.Module, config: HeatConfig, grid_size: int = 80) -> float:
    device = config.device
    x = torch.linspace(config.x_min, config.x_max, grid_size, device=device)
    t = torch.linspace(config.t_min, config.t_max, grid_size, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    x_flat = xx.reshape(-1, 1)
    t_flat = tt.reshape(-1, 1)
    prediction = model(torch.cat([x_flat, t_flat], dim=1))
    target = analytical_solution(x_flat, t_flat, config.alpha)
    return torch.mean((prediction - target) ** 2).item()


def train(config: HeatConfig) -> tuple[MLP, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = MLP(config.hidden_layers, config.hidden_width).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = pinn_losses(model, config)
        losses["total"].backward()
        optimizer.step()

        should_log = epoch == 1 or epoch == config.epochs or epoch % max(1, config.epochs // 10) == 0
        if should_log:
            row = {"epoch": float(epoch)}
            row.update({name: value.item() for name, value in losses.items()})
            row["validation_mse"] = validation_mse(model, config)
            history.append(row)
            print(
                f"epoch={epoch:5d} "
                f"loss={row['total']:.6e} "
                f"pde={row['pde']:.6e} "
                f"ic={row['initial']:.6e} "
                f"bc={row['boundary']:.6e} "
                f"val_mse={row['validation_mse']:.6e}"
            )

    return model, history


def save_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def maybe_plot(model: nn.Module, config: HeatConfig, output_dir: Path) -> None:
    try:
        from pinn_heat.plot import save_comparison_plot
        save_comparison_plot(model, config, output_dir / "comparison.png")
    except ImportError:
        print("matplotlib is not installed; skipping comparison plot.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PINN for the 1D heat equation.")
    parser.add_argument("--epochs", type=int, default=HeatConfig.epochs)
    parser.add_argument("--alpha", type=float, default=HeatConfig.alpha)
    parser.add_argument("--hidden-layers", type=int, default=HeatConfig.hidden_layers)
    parser.add_argument("--hidden-width", type=int, default=HeatConfig.hidden_width)
    parser.add_argument("--learning-rate", type=float, default=HeatConfig.learning_rate)
    parser.add_argument("--n-collocation", type=int, default=HeatConfig.n_collocation)
    parser.add_argument("--n-initial", type=int, default=HeatConfig.n_initial)
    parser.add_argument("--n-boundary", type=int, default=HeatConfig.n_boundary)
    parser.add_argument("--output-dir", type=str, default=HeatConfig.output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        HeatConfig(),
        epochs=args.epochs,
        alpha=args.alpha,
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
        learning_rate=args.learning_rate,
        n_collocation=args.n_collocation,
        n_initial=args.n_initial,
        n_boundary=args.n_boundary,
        output_dir=args.output_dir,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {config.device}")
    model, history = train(config)

    torch.save(model.state_dict(), output_dir / "model.pt")
    save_history(history, output_dir / "history.csv")
    maybe_plot(model, config, output_dir)


if __name__ == "__main__":
    main()
