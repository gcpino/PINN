import torch
from torch import nn

from pinn_heat.physics import analytical_solution, heat_residual, initial_condition


class AnalyticalHeatModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        return analytical_solution(x_t[:, 0:1], x_t[:, 1:2], self.alpha)


def test_initial_and_boundary_conditions() -> None:
    alpha = 0.01
    x = torch.linspace(0.0, 1.0, 25).reshape(-1, 1)
    t_zero = torch.zeros_like(x)

    assert torch.allclose(analytical_solution(x, t_zero, alpha), initial_condition(x), atol=1.0e-6)

    t = torch.linspace(0.0, 1.0, 25).reshape(-1, 1)
    x_left = torch.zeros_like(t)
    x_right = torch.ones_like(t)

    assert torch.allclose(analytical_solution(x_left, t, alpha), torch.zeros_like(t), atol=1.0e-6)
    assert torch.allclose(analytical_solution(x_right, t, alpha), torch.zeros_like(t), atol=1.0e-6)


def test_analytical_solution_has_small_heat_residual() -> None:
    alpha = 0.01
    model = AnalyticalHeatModel(alpha)
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64).reshape(-1, 1)
    t = torch.linspace(0.0, 1.0, 9, dtype=torch.float64).reshape(-1, 1)

    residual = heat_residual(model, x, t, alpha)

    assert torch.max(torch.abs(residual)).item() < 1.0e-10

