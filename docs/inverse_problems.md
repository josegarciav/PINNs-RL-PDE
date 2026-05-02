# Inverse Problems — Parameter Identification

Forward PINN training answers "given a PDE and its parameters, what is `u(x, t)`?" An **inverse problem** flips that around: given noisy or sparse measurements of `u`, recover the *unknown parameters* of the PDE that generated them. `pinnrl` supports inverse problems as a first-class training mode.

Real-world examples include:

- **Materials science** — recover thermal diffusivity `α` of a sample from infrared sensor traces.
- **Geophysics** — identify subsurface wave speed from seismic surface measurements.
- **Biomechanics** — infer tissue elasticity from displacement scans.
- **Climate / fluid dynamics** — estimate effective viscosity from particle-imaging velocimetry data.
- **Quantitative finance** — calibrate Black-Scholes volatility from observed option prices.

---

## How it works

In inverse mode, pinnrl jointly trains the network weights `θ` *and* a small set of PDE parameters that you mark trainable. The total loss combines the standard PINN terms with a data-fitting term:

```
L = w_pde · L_residual + w_bc · L_boundary + w_ic · L_initial + w_data · L_data
```

with

```
L_data = mean( | u_θ(x_obs, t_obs) − u_obs |² )
```

The trainable parameters live as `nn.Parameter` tensors registered on the PDE object, so PyTorch's autograd flows gradients through both the network and the parameters in a single backward pass. After training, `pde.get_trainable_parameter_values()` returns a `{name: float}` snapshot of the recovered values.

The PDE's analytical residual still uses these parameters, so the residual loss *anchors* them to physical solutions while the data loss *pulls* them toward the observations. The two terms together regularise an otherwise ill-posed inversion.

---

## Quickstart — recover thermal diffusivity from synthetic observations

The fastest way to see inverse mode work is to ask pinnrl to recover `α` of the heat equation from synthetic noisy observations of its own analytical solution.

### From the dashboard

1. Open **New Training**.
2. Pick **Heat Equation** as the PDE.
3. Set **Mode** to **Inverse (identify)** — the inverse panel reveals.
4. Tick **alpha** under "Parameters to identify" and set its initial guess (e.g. `0.5` while the truth is `0.05`).
5. Choose **Synthetic** as the observation source, leave Observation points = `200` and Noise std = `0.01`.
6. Click **Start Training**. The Monitor sub-tab will plot `alpha`'s trajectory toward the truth in real time.

### From the CLI

```bash
pinnrl-train \
  --pde "Heat Equation" --arch fourier \
  --epochs 3000 --device cpu \
  --mode inverse \
  --identify alpha \
  --initial-guess alpha=0.5 \
  --obs-points 200 --obs-noise 0.01
```

### From the Python API

```python
import torch
from pinnrl.config import (
    AdaptiveWeightsConfig, EarlyStoppingConfig,
    LearningRateSchedulerConfig, ModelConfig, TrainingConfig, Config,
)
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEConfig
from pinnrl.training.trainer import PDETrainer

device = torch.device("cpu")
TRUE_ALPHA, INITIAL_GUESS = 0.05, 0.5

training = TrainingConfig(
    num_epochs=3000, batch_size=2048,
    num_collocation_points=5000, num_boundary_points=500, num_initial_points=500,
    learning_rate=5e-3, weight_decay=0.0, gradient_clipping=1.0,
    early_stopping=EarlyStoppingConfig(enabled=False, patience=999, min_delta=1e-7),
    learning_rate_scheduler=LearningRateSchedulerConfig(
        type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3,
    ),
    adaptive_weights=AdaptiveWeightsConfig(enabled=False),
    loss_weights={"residual": 1.0, "boundary": 1.0, "initial": 1.0, "data": 50.0},
    mode="inverse",
)

pde_config = PDEConfig(
    name="Heat Equation",
    domain=[[0.0, 1.0]], time_domain=[0.0, 1.0],
    parameters={"alpha": TRUE_ALPHA},
    boundary_conditions={"type": {"value": 0.0}},
    initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
    exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
    dimension=1, device=device, training=training,
    trainable_parameters=["alpha"],
    parameter_initial_guesses={"alpha": INITIAL_GUESS},
)
pde = HeatEquation(config=pde_config)
pde.generate_synthetic_observations(n_points=200, noise_std=0.01, seed=0)

config = Config.__new__(Config)
config.device = device
config.model = ModelConfig(
    input_dim=2, hidden_dim=128, output_dim=1, num_layers=4,
    activation="tanh", architecture="fourier",
)
config.training = training

model = PINNModel(config=config, device=device)
trainer = PDETrainer(
    model=model, pde=pde,
    optimizer_config={"learning_rate": 5e-3, "weight_decay": 0.0},
    config=config, device=device,
)
trainer.train(num_epochs=3000)

print("Recovered alpha:", pde.get_trainable_parameter_values()["alpha"])
print("True alpha:     ", TRUE_ALPHA)
```

A successful run pulls `alpha` from `0.5` toward something close to `0.05` (typically within 5–10% under mild noise).

---

## Bringing your own observations

When you have real measurements, replace the synthetic source with an `.npz` file containing three arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `x` | `(N, d_space)` | Spatial coordinates of each observation |
| `t` | `(N, 1)` | Time of each observation |
| `u` | `(N, d_fields)` | Measured field values |

Load via the dashboard ("Observation source" → **File**, paste the absolute path) or the CLI flag `--obs-path /absolute/path/to/observations.npz`.

You can also feed a Well dataset as the observation source — see the [datasets guide](datasets.md). When mode is `data_augmented` and the PDE matches a Well dataset, the residual + IC/BC + data terms train together against real simulation snapshots.

---

## Dashboard panel reference

When **Mode = Inverse (identify)** is selected, the dashboard reveals an inverse panel with:

- **Parameters to identify** — checklist of the PDE's parameters that can be made trainable. The list is auto-populated from the selected PDE config.
- **Initial guesses** — one numeric input per checked parameter; a far-from-truth guess shows that the inversion actually moves the value.
- **Observation source** — radio:
  - `Synthetic` generates points from the analytical solution + Gaussian noise (controlled by Observation points + Noise std).
  - `File` loads a user-provided `.npz` from the path field.
- **Live parameter trajectory** — once training starts, the Monitor sub-tab plots each identified parameter's value at every logged epoch, so you can watch the convergence in real time.

---

## When inverse mode is well-posed

- **At least one observation per identified parameter.** Heuristically, plan for 50–200 noisy observations per scalar parameter.
- **Observations should span the dynamic regime where the parameter matters.** Trying to recover thermal diffusivity from a snapshot of stationary equilibrium fails because the diffusivity has no signature there.
- **Initial guesses within an order of magnitude of the truth converge faster.** Far guesses still recover under enough observations + epochs but may need a higher data weight (`loss_weights.data = 50–500`).
- **The data weight matters.** If the data term is too small, the residual will dominate and freeze parameters near the initial guess; if too large, observations override physics and the network overfits noise. The default `data = 1.0` works for matched PDE/observation scales; increase it 10× for synthetic experiments where the residual is computed on many more points than the observations.

---

## Combining inverse mode with the dataset modes

When you select a Well dataset for which pinnrl has an analytical solver (e.g. acoustic-scattering variants ↔ wave equation), three loss recipes are sensible:

| Goal | Mode | Behaviour |
|---|---|---|
| Train a surrogate that reproduces the Well snapshots | `data_only` | Skip residual / IC / BC; pure regression on the dataset. |
| Train a physics-constrained model on Well data | `data_augmented` | Residual + IC + BC + data fit; PDE parameters fixed. |
| Recover the dataset's underlying parameters | `inverse` | Residual + IC + BC + data fit; selected parameters trainable. |

Inverse mode + Well datasets is the natural setting for *parameter calibration* — fitting an analytical model to a real or high-fidelity reference simulation.

---

## See also

- [Dashboard guide](dashboard.md) — full New Training form reference.
- [Benchmark datasets](datasets.md) — Well dataset registry and usage.
- [`tests/unit_tests/test_inverse_heat.py`](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/tests/unit_tests/test_inverse_heat.py) — end-to-end inverse-heat test that recovers `alpha` from synthetic observations.
- API: [`PDEConfig.trainable_parameters`](api/pdes.md), [`PDEBase.get_trainable_parameter_values`](api/pdes.md), [`PDEBase.generate_synthetic_observations`](api/pdes.md).
