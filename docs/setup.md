# Setup

## Requirements

Python 3.10+ and PyTorch 2.0+. Optional: CUDA 11.8+ (NVIDIA) or macOS 12.3+ for MPS (Apple Silicon).

## Install

```bash
# Recommended
uv add pinnrl

# pip
pip install pinnrl

# With optional extras
pip install "pinnrl[viz]"       # Dash dashboard + seaborn
pip install "pinnrl[notebooks]" # Jupyter support
pip install "pinnrl[all]"       # Everything
```

## Development setup

```bash
git clone https://github.com/josegarciav/PINNs-RL-PDE.git
cd PINNs-RL-PDE
uv sync --all-extras
uv run pre-commit install
```

## Run tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src --cov-report=term-missing  # with coverage
```

## Device

Set in `config.yaml` or override via `--device` flag:

```yaml
device: "mps"   # mps | cuda | cpu
```

The library falls back to CPU automatically if the requested device is unavailable.

## Configuration

All behavior is controlled by `config.yaml`. Key sections:

- **`architectures`** — hyperparameters per architecture (`mapping_size`, `omega_0`, `num_heads`, etc.)
- **`pde_configs`** — domain, parameters, BCs, ICs, and default architecture per PDE
- **`training`** — epochs, batch size, loss weights, optimizer, scheduler, early stopping
- **`rl`** — enable/disable RL-based adaptive sampling (`enabled: false` by default)

> **dropout is 0.0 everywhere** — stochastic dropout disrupts the PDE residual gradient. Don't change this for PINN training.

## Verify

```bash
uv run python -c "
import torch
from src.config import ModelConfig
from src.neural_networks import PINNModel
cfg = ModelConfig(input_dim=2, hidden_dim=64, output_dim=1, num_layers=3, activation='tanh', architecture='fourier')
model = PINNModel(cfg)
print('OK —', model(torch.rand(4, 2)).shape)
"
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'torch'` | `pip install torch` before installing pinnrl |
| MPS not available | macOS 12.3+ required; run `pip install --upgrade torch` |
| `NaN` loss from epoch 0 | Lower `learning_rate` to `0.001` |
| Loss explodes after ~500 epochs | Add `gradient_clipping: 1.0` to training config |
| Slow convergence on Burgers/KdV | Use `resnet` for Burgers, `siren` for KdV |
| `uv sync` fails on Apple Silicon | Run `xcode-select --install` first |
