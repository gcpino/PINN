"""Physics-Informed Neural Network for the 1D heat equation."""

from pinn_heat.config import HeatConfig
from pinn_heat.model import MLP
from pinn_heat.physics import analytical_solution, heat_residual

__all__ = ["HeatConfig", "MLP", "analytical_solution", "heat_residual"]

