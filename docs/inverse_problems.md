# Inverse problems

This page is for anyone who wants to estimate parameters or functions of an equation from measurements. It states the loss, the workflow, the settings an inverse run needs, and the conditions under which the answer can be trusted.

Examples: a thermal diffusivity from temperature traces, a wave speed from surface seismograms, a fouling resistance from evaporator temperatures, a volatility from option prices.

A PINN learns the solution and the parameters in one optimization:

$$
\min_{\theta,\,\lambda}\; \mathcal{L} = w_r\,\mathcal{L}_\text{residual}(\theta, \lambda) + w_b\,\mathcal{L}_\text{boundary} + w_i\,\mathcal{L}_\text{initial}
+ w_d \operatorname{mean} \lvert u_\theta(x_\text{obs}, t_\text{obs}) - u_\text{obs} \rvert^2
$$

- **$\lambda$**: the unknown parameters of the equation. They are trainable tensors, updated with the network weights $\theta$.
- **$\mathcal{L}_\text{residual}$**: the mean squared residual $\mathcal{N}[u_\theta; \lambda]$ of the equation at the collocation points (sampled points inside the domain). It ties $\lambda$ to physically consistent solutions.
- **Data term**: the last term, the misfit at the observations $(x_\text{obs}, t_\text{obs}, u_\text{obs})$. It pulls the solution toward the measurements.
- **$w_r, w_b, w_i, w_d$**: the loss weights `residual`, `boundary`, `initial` and `data`.

Classical methods run a forward solver many times, plus its adjoint or an ensemble of solves. A PINN needs neither: one optimization returns $u_\theta$ and $\lambda$.

## Command line

```bash
pinnrl-train --pde heat --arch fourier --epochs 3000 --device cpu \
  --mode inverse --identify alpha --initial-guess alpha=0.5 \
  --obs-points 200 --obs-noise 0.01
```

What the run does:

1. It makes `alpha` trainable, starting from 0.5.
2. Without `--obs-path`, it samples observations from the PDE's exact solution at the configured (true) parameters and adds Gaussian noise. This is a recovery test whose answer is known.
3. It trains the network and `alpha` together.
4. It records the true and recovered values in the run's `metadata.json`.

| Flag | Default | Meaning |
|---|---|---|
| `--identify NAME` | none | A scalar parameter of the PDE to identify. Any one can be chosen, and the flag can be repeated |
| `--initial-guess NAME=VALUE` | the configured value | Starting value of that parameter |
| `--obs-points` | 200 | Number of synthetic observations |
| `--obs-noise` | 0.01 | Standard deviation of their Gaussian noise |
| `--obs-path file.npz` | none | Your own observations (see [Your own measurements](#your-own-measurements)) |

The command above keeps the packaged weight decay and gradient clipping. Add `--weight-decay 0 --gradient-clipping 0` for an accurate estimate (see [below](#regularisation-defaults-for-inverse-runs)).

## Python

1. Build a `TrainingConfig` with `mode="inverse"`, no weight decay and no gradient clipping.
2. Build the PDE with `trainable_parameters` and `parameter_initial_guesses`.
3. Attach observations (synthetic here).
4. Build the model and train.
5. Read the estimate with `pde.get_trainable_parameter_values()`.

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

### Regularisation defaults for inverse runs

The packaged defaults are forward-fit defaults. An inverse run must start from `weight_decay=0` and `gradient_clipping=0`.

| Setting | Packaged default | Inverse runs | Why |
|---|---|---|---|
| `optimizer_config.weight_decay` | 0.0005 | 0 | It pulls the network toward zero and the parameter absorbs the bias (F01, D01) |
| `gradient_clipping` | 1.0 (the default of `TrainingConfig`; not in `config.yaml`) | 0 | Clipping at norm 1 slows the network and leaves a field error the parameter absorbs (F01, D01) |
| Weight decay on the unknowns $\lambda$ | always 0 | always 0 | The trainer puts identified parameters and unknown functions in their own Adam group without decay (found in F09) |
| `optimizer` | `adam` | Adam, then an L-BFGS finish (`optimizer="adam_lbfgs"`, or `lbfgs_steps` in `train_windows`; see [Training](training.md)) | The finish halves the error and the bias again (D01) |
| `loss_weights.data` | 10 | search 10–100 | See [When the inversion is well posed](#when-the-inversion-is-well-posed) |

The evidence, on the heat diffusivity $\alpha$ ([F01](benchmarks.md#identification-baseline-experiment-f01), [D01](benchmarks.md#optimizers-and-the-floor-experiments-d01-d02)):

- **Packaged defaults**: the relative error of $\hat\alpha$ stays at 1.5–2 % whatever the number of observations (F01).
- **Both off**: 0.1–0.2 % on clean data (F01). With noise 0.01 the error falls with the number of observations along the Cramér–Rao bound, the smallest standard deviation any unbiased estimate can have for the given sensors and noise.
- **Both off, with an L-BFGS finish**: 0.2 %, against 0.4 % for Adam alone (D01). D01's ablation attributes about half of the old floor to each of the two settings.
- F11 (change point) saw the same effect: under the defaults a 3.6 % field error was absorbed into a smaller jump placed early, and three times the training did not help.
- The packaged defaults are unchanged, so set both to 0 yourself. [Troubleshooting](troubleshooting.md) lists the symptom under "An identified parameter stalls 1–2 % from the truth".

## Unknown functions

When the unknown is a function, replace the scalar by an `unknown_functions` entry of a [`CustomPDE`](custom_equations.md#unknown-functions):

$$
u_t = \big(\alpha(x)\,u_x\big)_x, \qquad \alpha(\cdot) \approx \text{a second network}
$$

- **Unknown function**: a coefficient that depends on the coordinates or on the solution. Examples: a diffusivity that drifts as a surface fouls, a coefficient that varies along the equipment, a closure $f(u)$.
- **Closure**: a relation the equation needs and the physics does not supply.
- **Constraints**: declare the function positive, bounded or monotone. Its network trains with the data under them.
- **Grading**: `pde.unknown_function_error(name)` compares it with the truth the synthetic data came from.

What this recovers on the heat equation ([F09](benchmarks.md#field-valued-unknowns-experiment-f09)):

- a linear drift $\alpha(t)$, to a few percent;
- a step in $\alpha(t)$, as a steep ramp;
- a two-layer field $\alpha(x)$, to its shape;
- not a centred bump in $\alpha(x)$: the solution hardly depends on it.

## Your own measurements

Point observations:

1. Save an `.npz` file with arrays `x` of shape `(N, d)`, `t` of shape `(N, 1)` and `u` of shape `(N, 1)`.
2. Pass it with `--obs-path`, in the dashboard (Observation source → File), or as `PDEConfig(observation_data={"path": ...})`.

A [Well dataset](datasets.md) can also serve as the observation source.

Plant records are better described as [sensors](custom_equations.md#sensors). Each sensor $s$ adds its own misfit term to the data loss:

$$
\mathcal{L}_\text{data} \mathrel{+}= w_s \operatorname{mean} \left\lvert \frac{H_s[u_\theta] - y_s}{c_s} \right\rvert^2
$$

- **Sensor**: records $y_s$ with an operator $H_s$: a probe at a fixed place read over time, a wall flux, an integral. Each has its noise and delay.
- **$w_s$, $c_s$**: its `weight` (default 1) and `scale` (default 1, or the field's scale when the problem is normalised).
- **Fisher information**: how much a sensor layout says about the unknowns, computed from the sensitivities of the readings.

Steps:

1. Declare the sensors: `CustomPDE(sensors=[PointSensor(x0, times), ...])`.
2. Attach the readings: `sensor.set_values(records)`.
3. Before training, compute `sensor_jacobian` to get the Fisher information of the layout and choose the sensors.

## When the inversion is well posed

- **The data must excite the parameter.** Diffusivity cannot be recovered from a steady state, because it leaves no signature there. Observations must cover the regime where the parameter acts.
- **Balance the data weight.** Too small, and the parameter barely moves from its initial guess. Too large, and the network fits the noise and the physics stops constraining it. The example above (256 collocation points, 200 observations, noise 0.005) uses 50; 10–100 is a reasonable range to search.
- **Check identifiability before trusting a number.** Several parameters, or a parameter field, may not be uniquely determined by the sensors you have. The [theory pages](theory/identifiability.md) state the conditions, and `pinnrl.analysis` computes the Fisher bound when a closed form exists.
- **A field is identified only where the solution feels it.** For $u_t = (\alpha(x) u_x)_x$ the sensitivity to a local relative change $\delta\alpha / \alpha$ is proportional to the flux there:

$$
\frac{\delta u}{\delta \alpha / \alpha}(x) \;\propto\; \alpha\,u_x
$$

- $\alpha$ is invisible where $u_x$ vanishes.
- A localized perturbation is smoothed by diffusion before it reaches the sensors.
- The check: compare the truth's solution with the one of the best constant (`heat_variable_reference` makes this one line). If they differ by less than the trained network's own error, the data cannot resolve the field.
