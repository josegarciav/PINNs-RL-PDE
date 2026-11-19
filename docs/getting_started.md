# Getting started

This page takes a new user from installation to a first trained PINN, on the command line and in Python. Read it before the guides.

## Install

```bash
pip install pinnrl            # or: uv add pinnrl
```

- **Python**: 3.10–3.13.
- **PyTorch**: installed as a dependency. It runs on CPU, Apple GPU (`mps`) or CUDA.
- **Compiled engine**: the default training step is compiled C++ and needs a C++ compiler at first use. Without one, everything runs on autograd. See [The native engine](engine.md).
- **Autograd**: PyTorch's automatic differentiation, the fallback engine.
- **Extra**: an optional group of dependencies, installed as `pip install 'pinnrl[name]'`.

| Extra | Adds |
|---|---|
| `pinnrl[well]` | Loaders for [The Well](datasets.md) benchmark datasets |
| `pinnrl[notebooks]` | Jupyter, for the tutorial notebooks in the repository |
| `pinnrl[docs]`, `pinnrl[dev]` | Building these docs; tests and linters |

## First run from the command line

```bash
pinnrl-train --pde burgers --arch fourier --epochs 2000 --device cpu
```

- **`--pde`**: the equation to solve. It takes a key (`heat`, `heat_2d`, `wave`, `burgers`, `convection`, `kdv`, `allen_cahn`, `cahn_hilliard`, `black_scholes`, `pendulum`) or the display name (`"Burgers Equation"`).
- **`--arch`**: the network architecture, here Fourier features.
- **`--epochs`**: the number of passes of the optimizer over the sampled points.
- **`--device`**: where PyTorch runs. A device that is not available falls back to CPU.
- **Other settings**: they come from the packaged `config.yaml`. Override them with flags (see [Training](training.md)), or pass `--config my.yaml`.
- **Engine**: `--derivatives auto` is the default and uses the compiled engine where it applies. `--derivatives autograd` switches it off.

Each run writes a folder under `./runs/` in the current directory:

| File | Contents |
|---|---|
| `config.yaml` | The exact configuration used |
| `metadata.json` | Run info and status, final and best validation loss, true and identified parameters (inverse mode) |
| `history.json` (and an identical `metrics.json`) | Per-epoch losses, learning rate and, in inverse mode, parameter values |
| `final_model.pt` | Model weights (reload with `pinnrl.training.train.load_run`) |
| `experiment.log`, `live_snapshot.npz` | The training log; the prediction and residual grid the dashboard plots |
| `visualizations/final_training_history.png`, `final_solution_comparison.png`, `collocation_evolution.png` | Plots |

- **Inverse mode**: training that also identifies parameters of the equation from measurements.
- **Residual**: what is left when the network is put into the equation.

## Dashboard

```bash
pinnrl-dashboard                 # opens http://127.0.0.1:8050/
pinnrl-dashboard --port 8051 --no-browser
```

The dashboard does three things:

1. It configures and launches runs.
2. It shows the solution and residual surfaces as training proceeds.
3. It compares finished runs.

See [Dashboard](dashboard.md).

## Python API

The shortest route is your equation as text. Here it is the heat equation with a known exact solution:

$$
u_t - \alpha\, u_{xx} = 0, \qquad u(x, 0) = \sin(2\pi x), \qquad u^*(x, t) = e^{-4\alpha\pi^2 t}\sin(2\pi x)
$$

- **$u_t$, $u_{xx}$**: the time derivative and the second space derivative of $u$.
- **$\alpha$**: the diffusivity, here 0.1.
- **$u^*$**: the exact solution, used to measure the error.
- **`CustomPDE`**: builds the problem from the text.
- **`build_trainer`**: builds a trainer with the packaged defaults.
- **Collocation point**: a sampled point where the residual is evaluated.

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

See [Custom equations](custom_equations.md) for systems, ODEs, callables and the YAML form.

The same pieces, spelled out:

1. A shipped PDE class (`HeatEquation`) with its `PDEConfig`.
2. A `PINNModel`.
3. A `PDETrainer`.

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

`validate` returns four metrics:

$$
\text{relative L2 error} = \frac{\lVert u - u^* \rVert}{\lVert u^* \rVert}
$$

- **`l2_error`**: the relative L2 error of the network $u$ against the exact solution $u^*$.
- **`mse`, `max_error`, `mean_error`**: the mean squared, maximum and mean errors.

## Next

- [Concepts](concepts.md) explains the loss, sampling and architecture choices behind these settings.
- [When to use a PINN](when_to_use.md) says which problems suit the method.
- The [tutorial notebooks](https://github.com/josegarciav/PINNs-RL-PDE/tree/main/notebooks) walk through:
    1. a first PINN;
    2. an architecture comparison;
    3. the RL sampler;
    4. The Well;
    5. inverse problems;
    6. notebooks 06, 07 and 08: the identification-baseline, conditioning, and stability-and-uncertainty experiments, reviewed from their raw results.
