# Getting started

## Install

```bash
pip install pinnrl            # or: uv add pinnrl
```

Python 3.10–3.13. PyTorch is a dependency and runs on CPU, Apple GPU (`mps`) or CUDA. Optional extras:

| Extra | Adds |
|---|---|
| `pinnrl[well]` | Loaders for [The Well](datasets.md) benchmark datasets |
| `pinnrl[notebooks]` | Jupyter, for the tutorial notebooks in the repository |
| `pinnrl[docs]`, `pinnrl[dev]` | Building these docs; tests and linters |

## First run from the command line

```bash
pinnrl-train --pde burgers --arch fourier --epochs 2000 --device cpu
```

- `--pde` takes a key (`heat`, `heat_2d`, `wave`, `burgers`, `convection`, `kdv`, `allen_cahn`, `cahn_hilliard`, `black_scholes`, `pendulum`) or the display name (`"Burgers Equation"`).
- Every other setting comes from the packaged `config.yaml`. Override it with flags (see [Training](training.md)), or pass `--config my.yaml`.
- A device that isn't available falls back to CPU.

Each run writes a folder under `./runs/` in the current directory:

| File | Contents |
|---|---|
| `config.yaml` | The exact configuration used |
| `metadata.json` | Run info and status, final and best validation loss, true and identified parameters (inverse mode) |
| `history.json` (and an identical `metrics.json`) | Per-epoch losses, learning rate and, in inverse mode, parameter values |
| `final_model.pt` | Model weights (reload with `pinnrl.training.train.load_run`) |
| `experiment.log`, `live_snapshot.npz` | The training log; the prediction and residual grid the dashboard plots |
| `visualizations/final_training_history.png`, `final_solution_comparison.png`, `collocation_evolution.png` | Plots |

## Dashboard

```bash
pinnrl-dashboard                 # opens http://127.0.0.1:8050/
pinnrl-dashboard --port 8051 --no-browser
```

The dashboard lets you configure and launch runs, watch the solution and residual surfaces update as training proceeds, and compare finished runs. See [Dashboard](dashboard.md).

## Python API

The shortest route is your equation as text. `CustomPDE` builds the problem and `build_trainer` a trainer with the packaged defaults:

```python
from pinnrl import CustomPDE

pde = CustomPDE("u_t - alpha*u_xx", domain={"x": (0, 1), "t": (0, 1)}, parameters={"alpha": 0.1},
                initial_condition="sin(2*pi*x)", boundary_conditions={"periodic": {}},
                exact_solution="exp(-4*alpha*pi^2*t)*sin(2*pi*x)")
trainer = pde.build_trainer("fourier", num_epochs=150, num_collocation_points=1024, batch_size=1024,
                            learning_rate=2e-3, early_stopping={"enabled": False})
trainer.train(progress=False)
print(f"relative L2 error: {pde.validate(trainer.model)['l2_error']:.2e}")
```

See [Custom equations](custom_equations.md) for systems, ODEs, callables and the YAML form. The same pieces, spelled out with a shipped PDE class, a `PINNModel` and a `PDETrainer`:

```python
import torch

from pinnrl.config import Config, ModelConfig, TrainingConfig
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes import HeatEquation, PDEConfig
from pinnrl.training import PDETrainer

device = torch.device("cpu")
training = TrainingConfig.from_dict({
    "num_epochs": 200,
    "batch_size": 500,
    "num_collocation_points": 500,
    "num_boundary_points": 100,
    "num_initial_points": 100,
    "optimizer_config": {"learning_rate": 5e-3, "weight_decay": 0.0},
    "scheduler_type": "constant",
    "early_stopping": {"enabled": False},
})

# u_t = alpha * u_xx on [0, 1] x [0, 1], periodic in x, u(x, 0) = sin(2 pi x)
pde = HeatEquation(PDEConfig(
    name="Heat Equation",
    domain=[[0.0, 1.0]],
    time_domain=[0.0, 1.0],
    parameters={"alpha": 0.1},
    boundary_conditions={"periodic": {}},
    initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
    exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
    dimension=1,
    device=device,
    training=training,
))

config = Config()
config.device, config.training = device, training
config.model = ModelConfig.from_architecture(
    "fourier",
    {"hidden_dims": [64, 64, 64], "mapping_size": 32, "scale": 4.0, "activation": "tanh"},
    input_dim=2,
    output_dim=1,
)
model = PINNModel(config, device=device)

PDETrainer(model, pde, config=config, device=device).train(progress=False)
metrics = pde.validate(model)
print(f"relative L2 error: {metrics['l2_error']:.2e}")
```

`validate` returns the relative L2 error $\lVert u - u^* \rVert / \lVert u^* \rVert$ as `l2_error`, plus `mse`, `max_error` and `mean_error`.

## Next

- [Concepts](concepts.md) explains the loss, sampling and architecture choices behind these settings.
- The [tutorial notebooks](https://github.com/josegarciav/PINNs-RL-PDE/tree/main/notebooks) walk through a first PINN, an architecture comparison, the RL sampler, The Well, and inverse problems; notebooks 06, 07 and 08 review the identification-baseline, conditioning, and stability-and-uncertainty experiments from their raw results.
