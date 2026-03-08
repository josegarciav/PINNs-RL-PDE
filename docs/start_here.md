# Start Here — pinnrl in 5 Minutes

## What is pinnrl?

`pinnrl` is an open-source Python library for solving partial differential equations (PDEs) using Physics-Informed Neural Networks (PINNs), with an optional reinforcement learning layer that adaptively selects collocation points to improve training efficiency. It supports nine PDEs out of the box, six neural network architectures, and runs on CPU, CUDA, and Apple Silicon (MPS) with a single configuration change.

---

## What you can do in 5 minutes

1. Install the library
2. Run a training job from the CLI
3. Inspect the loss curve and solution plot saved to `experiments/`

### Step 1 — Install

Using `uv` (recommended):

```bash
uv add pinnrl
```

Using `pip`:

```bash
pip install pinnrl
```

### Step 2 — Run the CLI

Train a PINN on the heat equation using the Fourier Features architecture for 3000 epochs:

```bash
python scripts/train.py --pde heat --arch fourier --epochs 3000
```

You will see a live progress bar. Training output is saved automatically to a timestamped directory under `experiments/`.

### Step 3 — Inspect results

After training completes, the experiment directory contains:

```
experiments/
  20240101_120000_Heat Equation_fourier_no_rl/
    config.yaml          # full config snapshot
    loss_curve.png       # training and validation loss
    solution_plot.png    # predicted vs exact solution
    metrics.json         # L2 error, max error, mean error
```

A successful run on the heat equation typically reaches a total loss below `1e-3` within 3000 epochs.

---

## Hello World — Solving the Heat Equation

The heat equation describes how heat (or any diffusing quantity) evolves over space and time:

```
u_t = alpha * u_xx
```

where `u(x, t)` is the temperature field, `alpha` is the thermal diffusivity, and subscripts denote partial derivatives.

### How the physics loss works

A PINN learns to satisfy the PDE by minimizing three residual terms simultaneously:

**1. PDE residual** — the network must satisfy the governing equation at interior collocation points:

```
L_residual = mean( (u_t - alpha * u_xx)^2 )
```

**2. Boundary condition (BC) loss** — the network must match the prescribed boundary values:

```
L_bc = mean( (u_network(x_boundary, t) - u_boundary)^2 )
```

**3. Initial condition (IC) loss** — the network must reproduce the initial state at `t=0`:

```
L_ic = mean( (u_network(x, 0) - u_0(x))^2 )
```

The total training loss combines all three terms with configurable weights:

```
L_total = w_residual * L_residual + w_bc * L_bc + w_ic * L_ic
```

Default weights in `config.yaml`: `residual=15.0`, `boundary=20.0`, `initial=10.0`.

### Python API example

```python
import torch
import yaml
from src.config import Config, ModelConfig, TrainingConfig
from src.neural_networks import PINNModel
from src.trainer import PDETrainer
from src.pdes.heat_equation import HeatEquation
from src.pdes.pde_base import PDEConfig

# Load default config
from src.config import DEFAULT_CONFIG_PATH
with open(DEFAULT_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

pde_cfg = cfg["pde_configs"]["heat"]

# Build PDE configuration
pde_config = PDEConfig(
    name="Heat Equation",
    domain=[[0, 2]],
    time_domain=[0, 10],
    parameters={"alpha": 0.01},
    boundary_conditions={"periodic": {}},
    initial_condition={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 2.0},
    exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 2.0},
    dimension=1,
)

# Instantiate PDE
pde = HeatEquation(pde_config)

# Build model — Fourier Features architecture
model_config = ModelConfig(
    input_dim=2,           # (x, t)
    output_dim=1,          # u(x, t)
    architecture="fourier",
    mapping_size=512,
    hidden_dims=[512, 512, 512, 512],
    scale=4.0,
)
model = PINNModel(model_config)

# Train
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
trainer = PDETrainer(model=model, pde=pde, optimizer_config={"name": "adam", "learning_rate": 0.005}, config=None, device=device)
trainer.train(num_epochs=3000)
```

### Expected training output

```
Epoch    0 | Loss: 4.2351 | Residual: 2.1104 | BC: 1.3847 | IC: 0.7400
Epoch  100 | Loss: 0.8712 | Residual: 0.3204 | BC: 0.2917 | IC: 0.2591
Epoch  500 | Loss: 0.0341 | Residual: 0.0089 | BC: 0.0147 | IC: 0.0105
Epoch 1000 | Loss: 0.0047 | Residual: 0.0011 | BC: 0.0021 | IC: 0.0015
Epoch 3000 | Loss: 0.0003 | Residual: 0.0001 | BC: 0.0001 | IC: 0.0001
```

The loss curve descends monotonically (with minor fluctuations from cosine LR annealing). After training, `solution_plot.png` shows the PINN prediction overlaid with the analytic solution — the two should be visually indistinguishable for the heat equation at `alpha=0.01`.

---

## Next steps

| Goal | Where to go |
|---|---|
| Full installation, GPU setup, troubleshooting | [docs/setup.md](setup.md) |
| Architecture selection and training loop internals | [docs/architecture/overview.md](architecture/overview.md) |
| Interactive examples and comparisons | `notebooks/PINN_intro_workshop.ipynb` |
| Sampling strategies (RL vs uniform vs Latin hypercube) | `notebooks/sampling_techniques.ipynb` |
| Roadmap and future directions | [docs/roadmap.md](roadmap.md) |

---

## Quick reference card

| PDE | Recommended arch | Key parameters | CLI flag |
|---|---|---|---|
| Heat equation | `fourier` | `alpha` (diffusivity), periodic BC | `--pde heat` |
| Wave equation | `siren` | `c` (wave speed), Dirichlet BC | `--pde wave` |
| Burgers equation | `resnet` | `viscosity`, shock-forming | `--pde burgers` |
| Convection equation | `fourier` | `velocity`, periodic BC | `--pde convection` |
| KdV equation | `siren` | `alpha`, `beta`, soliton IC | `--pde kdv` |
| Allen-Cahn equation | `fourier` | `epsilon` (interface width) | `--pde allen_cahn` |
| Cahn-Hilliard equation | `resnet` | `mobility`, `kappa` | `--pde cahn_hilliard` |
| Black-Scholes equation | `feedforward` | `sigma` (vol), `r` (rate) | `--pde black_scholes` |
| Pendulum equation | `resnet` | `g`, `L`, `damping` | `--pde pendulum` |

**Epochs guidance:** Start with 3000 for linear PDEs (heat, wave, convection). Use 5000+ for nonlinear PDEs (Burgers, KdV, Allen-Cahn, Cahn-Hilliard). Black-Scholes and Pendulum typically converge well within 3000 epochs.
