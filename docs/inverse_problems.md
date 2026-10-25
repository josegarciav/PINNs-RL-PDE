# Inverse problems

A forward problem asks for `u(x, t)` given the PDE and its parameters. An **inverse problem** asks for the parameters given measurements of `u`. Examples include a material's thermal diffusivity from temperature traces, a wave speed from surface seismograms, a fouling resistance from evaporator temperatures, or a volatility from option prices.

Classical approaches need a forward solver run many times, plus its adjoint for gradient-based methods (ensemble Kalman filters avoid the adjoint at the cost of an ensemble of solves). A PINN learns the solution and the parameters in one optimization. The parameters become trainable tensors, and a data term joins the physics loss:

$$
\mathcal{L} = w_r\,\mathcal{L}_\text{residual} + w_b\,\mathcal{L}_\text{boundary} + w_i\,\mathcal{L}_\text{initial}
+ w_d \operatorname{mean} \lvert u_\theta(x_\text{obs}, t_\text{obs}) - u_\text{obs} \rvert^2
$$

The residual ties the parameters to physically consistent solutions; the data term pulls the solution toward the measurements.

## Command line

```bash
pinnrl-train --pde heat --arch fourier --epochs 3000 --device cpu \
  --mode inverse --identify alpha --initial-guess alpha=0.5 \
  --obs-points 200 --obs-noise 0.01
```

Without `--obs-path`, observations are sampled from the PDE's exact solution at the configured (true) parameters, with Gaussian noise. That makes it a recovery test whose answer is known. The run's `metadata.json` records the true and recovered values. Any scalar parameter of the PDE can be identified, and `--identify` can be repeated.

## Python

```python
import torch

from pinnrl.config import Config, ModelConfig, TrainingConfig
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes import HeatEquation, PDEConfig
from pinnrl.training import PDETrainer

device = torch.device("cpu")
training = TrainingConfig.from_dict({
    "mode": "inverse",
    "num_epochs": 1500,
    "batch_size": 256,
    "num_collocation_points": 256,
    "optimizer_config": {"learning_rate": 5e-3, "weight_decay": 0.0},
    "scheduler_type": "constant",
    "early_stopping": {"enabled": False},
    "gradient_clipping": 0.0,
    "loss_weights": {"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 50.0},
})

pde = HeatEquation(PDEConfig(
    name="Heat Equation",
    domain=[[0.0, 1.0]],
    time_domain=[0.0, 1.0],
    parameters={"alpha": 0.05},                      # the truth, used for synthetic data
    boundary_conditions={"periodic": {}},
    initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
    exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
    dimension=1,
    device=device,
    training=training,
    trainable_parameters=["alpha"],
    parameter_initial_guesses={"alpha": 0.5},        # start 10x off
))
pde.generate_synthetic_observations(n_points=200, noise_std=0.005, seed=0)

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
print(pde.get_trainable_parameter_values())          # notebook 05 recovers about 0.0499
```

## Your own measurements

Provide an `.npz` file with arrays `x` of shape `(N, d)`, `t` of shape `(N, 1)` and `u` of shape `(N, 1)`. Pass it with `--obs-path`, in the dashboard (Observation source → File), or as `PDEConfig(observation_data={"path": ...})`. A [Well dataset](datasets.md) can also serve as the observation source.

## When the inversion is well posed

- **The data must excite the parameter.** Diffusivity can't be recovered from a steady state, because it leaves no signature there. Observations must cover the regime where the parameter acts.
- **Balance the data weight.** If it's too small, the parameter barely moves from its initial guess. If it's too large, the network fits the noise and the physics stops constraining it. The example above (256 collocation points, 200 observations, noise 0.005) uses 50; 10–100 is a reasonable range to search.
- **Check identifiability before trusting a number.** Several parameters, or a parameter field, may not be uniquely determined by the sensors you have. The [theory pages](theory/identifiability.md) state the conditions, and `pinnrl.analysis` computes the Fisher bound when a closed form exists.
