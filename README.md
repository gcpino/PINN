# PINN Heat Equation

A small Physics-Informed Neural Network project for the 1D heat equation:

```text
u_t = alpha * u_xx
```

with domain `x in [0, 1]`, `t in [0, 1]`, initial condition
`u(x, 0) = sin(pi x)`, and boundary conditions `u(0, t) = u(1, t) = 0`.

The analytical solution used for validation is:

```text
u(x, t) = exp(-alpha * pi^2 * t) * sin(pi x)
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[plot,test]"
```

## Train

```powershell
python -m pinn_heat.train --epochs 2000
```

For a quick smoke run:

```powershell
python -m pinn_heat.train --epochs 50 --n-collocation 512 --n-initial 128 --n-boundary 128
```

Training writes artifacts to `outputs/`:

- `model.pt`
- `history.csv`
- `comparison.png`, if `matplotlib` is installed

## Test

```powershell
python -m pytest
```
