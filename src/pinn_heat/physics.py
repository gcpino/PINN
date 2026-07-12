import math

import torch
from torch import nn


def analytical_solution(x: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.exp(-alpha * math.pi**2 * t) * torch.sin(math.pi * x)


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x)


def heat_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = model(torch.cat([x, t], dim=1))

    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_t - alpha * u_xx

