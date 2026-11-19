# Custom equations

This page shows how to write your own PDE, ODE or system as text and train it with `CustomPDE`. Read it when the equation you need is not one of the [shipped ones](equations.md).

- **Residual**: the equation with every term on the left-hand side. You write it as text; [SymPy](https://www.sympy.org) parses it.
- **`PDEBase`**: the class of every problem. A `CustomPDE` is one, so the training loop, samplers, optimizers, inverse-problem machinery, `pinnrl-train` and the dashboard treat it like a shipped equation.
- **Derivatives**: taken by the [compiled kernel](#a-faster-step-the-compiled-kernel) where it applies, by autograd elsewhere.

## A first equation

$$
u_t - \alpha\, u_{xx} = 0, \qquad u(x, 0) = \sin \pi x, \qquad u(0, t) = u(1, t) = 0, \qquad u_{\text{exact}} = e^{-\alpha \pi^2 t} \sin \pi x
$$

```python
from pinnrl import CustomPDE

pde = CustomPDE(
    "u_t - alpha*u_xx",
    domain={"x": (0, 1), "t": (0, 1)},
    parameters={"alpha": 0.1},
    initial_condition="sin(pi*x)",
    boundary_conditions={"dirichlet": {"value": 0.0}},
    exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",   # optional, enables validate()
)
trainer = pde.build_trainer("fourier", num_epochs=150, num_collocation_points=1024,
                            batch_size=1024, learning_rate=2e-3,
                            early_stopping={"enabled": False})
trainer.train(progress=False)
print(f"relative L2 error: {pde.validate(trainer.model)['l2_error']:.2e}")
```

`CustomPDE`:

- **First argument**: the residual, the left-hand side of `expression = 0`.
- **`parameters`**: a value for every parameter in the text.

`build_trainer` returns a `PDETrainer`:

- **First argument**: an architecture of the packaged defaults (`config.yaml`).
- **Keyword arguments**: override keys of the `training` block (`num_epochs`, `batch_size`, `num_collocation_points`, `num_boundary_points`, `num_initial_points`, `learning_rate`, `weight_decay`, `optimizer`, `loss_weights`, `mode`, ...).
- **`model`**: overrides the architecture block, e.g. `model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False}`.
- **Collocation point**: a point where the residual is evaluated.

You can equally build a `PINNModel` and `PDETrainer` yourself, as in [Getting started](getting_started.md#python-api).

## Notation

| Written as | Meaning |
|---|---|
| `u`, `v`, `theta` | a field (network output), as listed in `fields` (default `["u"]`) |
| `x`, `y`, `z`, `r`, `t` | a coordinate (network input), as listed in `domain`; `t` is time |
| `u_t`, `u_xx`, `u_xy`, `theta_tt` | partial derivatives: field, underscore, one coordinate name per differentiation, any order |
| `nu`, `alpha`, `D`, `k` | a parameter: any other name, supplied in `parameters` |
| `sin cos tan exp log sqrt tanh sinh cosh abs sign erf min max atan2 sigmoid relu`, `pi` | functions and the constant π; `^` or `**` is a power; `2*x`, not `2 x` |

- Coordinates with several letters work (`u_theta`), as long as the subscript splits into coordinate names in one way only.
- A derivative with an unknown coordinate is an error (`u_zz` when `z` is not in the domain).
- A parameter the equations use but `parameters` does not define is an error.

## Domains and kinds of problem

`domain` maps each coordinate to `(lo, hi)`. Spatial coordinates come first, in the order given; `t` is the time axis.

- **Time-dependent PDE:** `{"x": (0, 1), "t": (0, 1)}`, with an `initial_condition`.
- **Second order in time** (wave, oscillators): detected from `_tt` in the equations; `initial_velocity` gives `u_t(x, t0)` (default 0).
- **ODE:** a domain with only `t`, e.g. `{"t": (0, 10)}`. The network receives a dummy spatial input that the solution does not depend on.
- **Steady problem** (Poisson, Helmholtz): a domain without `t`. There is no initial term and `t` is fixed at 0.
- **Higher dimension:** `{"x": (0, 1), "y": (0, 1), "t": (0, 2)}`; `u_xx + u_yy` is the Laplacian.

## Initial, boundary and exact solutions

`initial_condition`, `initial_velocity` and `exact_solution` take:

- **A string**, for a scalar problem: an expression in the coordinates and parameters. Only the exact solution may use `t`.
- **A list or `{field: expression}`**, for a system. Fields left out are 0.
- **A callable**, `u0(x)` or `u(x, t)` returning an `(N, k)` tensor, when there is no closed form (a numerical reference, interpolated data).

`boundary_conditions` are the generic ones of `PDEBase`, applied to every field on every face:

| Kind | Written as |
|---|---|
| Dirichlet, $u = g$ | `{"dirichlet": {"value": 0.0}}` or `{"dirichlet": {"left": 1.0, "right": 0.0}}` |
| Neumann, $u_x = g$ along the face's axis | `{"neumann": {"value": 0.0}}` |
| Periodic | `{"periodic": {}}` |
| Values from the exact solution | `{"dirichlet": {"type": "exact"}}` |

- **Face**: one end of a spatial axis; `left` is the lower end, `right` the upper.
- Conditions per field are keyed by field name; see [Systems](#systems).

## Systems

Give one equation per field, in the order of `fields`. A Gray–Scott reaction–diffusion system in 2-D:

$$
\begin{aligned}
u_t &= D_u \nabla^2 u - u v^2 + F (1 - u) \\
v_t &= D_v \nabla^2 v + u v^2 - (F + k)\, v
\end{aligned}
$$

<!-- doctest: skip -->
```python
pde = CustomPDE(
    ["u_t - Du*(u_xx + u_yy) + u*v^2 - F*(1 - u)",
     "v_t - Dv*(v_xx + v_yy) - u*v^2 + (F + k)*v"],
    domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
    fields=["u", "v"],
    parameters={"Du": 0.16, "Dv": 0.08, "F": 0.035, "k": 0.065},
    initial_condition={"u": "1 - 0.5*exp(-((x-0.5)^2 + (y-0.5)^2)/0.01)",
                       "v": "0.25*exp(-((x-0.5)^2 + (y-0.5)^2)/0.01)"},
    boundary_conditions={"periodic": {}},
)
```

- **Residual**: shape `(N, 2)`. The loss averages over both equations; residual-based sampling uses their norm at each point.
- **Conditions per field**: `boundary_conditions={"u": {"dirichlet": {"value": 1.0}}, "v": {"neumann": {"value": 0.0}}}`. A field left out has none. `{kind: options}` applies to every field.
- **`field_weights`**: `field_weights={"v": 10.0}` multiplies that field's residual, initial and boundary errors in the loss. Use it when one component is small or slow and the optimizer would ignore it.
- **Test case for both**: the catalogue entry `reaction_diffusion_pair`, two heat equations exchanging mass at rate `k`, with a closed form.

## A residual in Python

When the expression language is not enough (a lookup table, a learned closure, a conditional), pass a callable `residual(f, p)`.

$$
u_t - \alpha\, u_{xx} - s(x) = 0, \qquad s(x) = \begin{cases} 1 & x < 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

- **`f`**: a `Fields` accessor with the notation above as attributes (`f.u`, `f.u_xx`, `f.x`, `f.t`, `f.d("u", "x", "t")`, `f.laplacian("u")`). Each derivative is computed once and cached.
- **`p`**: the parameter dict, with `nn.Parameter` values for the ones being identified.
- **Return**: an `(N, k)` tensor or a list of `(N, 1)` tensors.
- **`time_order=2`**: pass it with a callable that is second order in time; for text it is detected.

<!-- doctest: skip -->
```python
import torch

def residual(f, p):
    source = torch.where(f.x < 0.5, 1.0, 0.0)          # a piecewise forcing
    return f.u_t - p["alpha"] * f.u_xx - source

pde = CustomPDE(residual, domain={"x": (0, 1), "t": (0, 1)}, parameters={"alpha": 0.1},
                initial_condition="0", boundary_conditions={"dirichlet": {"value": 0.0}})
```

## Inverse problems

Any parameter can be identified from data.

$$
\min_{w,\, \alpha} \; \mathcal{L}_{\text{physics}}(w, \alpha) + \lambda_{\text{data}} \, \frac{1}{N} \sum_{i=1}^{N} \big( u_w(x_i, t_i) - y_i \big)^2
$$

- $w$: the network weights. $\alpha$: the parameter to identify.
- $\mathcal{L}_{\text{physics}}$: the residual, initial and boundary terms.
- $y_i$: an observation at $(x_i, t_i)$. $\lambda_{\text{data}}$: its weight, `loss_weights={"data": 50}` below.

Steps:

1. List the parameter in `trainable_parameters`, with a starting value in `parameter_initial_guesses`.
2. Attach observations: `observation_data={"x": ..., "t": ..., "u": ...}` or `{"path": "obs.npz"}`. Without observations and with an exact solution, `generate_synthetic_observations` creates noisy ones.
3. Train in `mode="inverse"`; `build_trainer` selects it when there are trainable parameters.

See [Inverse problems](inverse_problems.md).

<!-- doctest: skip -->
```python
pde = CustomPDE(
    "u_t - alpha*u_xx",
    domain={"x": (0, 1), "t": (0, 1)},
    parameters={"alpha": 0.05},                 # the true value, used for synthetic data
    trainable_parameters=["alpha"],
    parameter_initial_guesses={"alpha": 0.5},
    initial_condition="sin(pi*x)",
    boundary_conditions={"dirichlet": {"value": 0.0}},
    exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",
)
pde.generate_synthetic_observations(n_points=200, noise_std=0.01, seed=0)
trainer = pde.build_trainer("feedforward", num_epochs=500, loss_weights={"data": 50})
trainer.train(progress=False)
print(pde.get_trainable_parameter_values())
```

- **Log parameterization**: the optimizer takes the same absolute step for a PDE parameter as for a weight. For a small positive parameter such as a viscosity of $10^{-3}$, write `exp(log_nu)` in the residual and list `log_nu` as trainable. The step becomes relative and the viscosity stays positive.
- **Cramér–Rao bound**: the floor any unbiased estimate can reach. With a closed-form solution, `pinnrl.analysis.closed_form_bound(pde, ["alpha"], x, t, sigma)` gives it for observations at `(x, t)` with noise `sigma`.

## Unknown functions

Use one when the unknown physics is a function: a coefficient that drifts in time, a heterogeneous field, a closure of the solution.

$$
u_t - \alpha(t)\, u_{xx} = 0, \qquad \alpha(t) = 0.05 + 0.1\, t \ \text{ to be recovered}
$$

- **[`UnknownFunction`](api/pdes.md#pdes.unknowns.UnknownFunction)**: a small network of the inputs you name (coordinates or fields), trained with the data in inverse mode.
- **Closure**: a term that depends on the solution itself, such as a reaction rate $f(u)$.
- **Use**: give it as an `unknown_functions` entry and write its name in the equation like a parameter.
- **Subscripts**: `alpha_x` is the derivative of the learned function. The conservative form of a heterogeneous diffusion is `u_t - alpha_x*u_x - alpha*u_xx`.

Keys of an entry. The constraints are built into the output, so training never leaves the admissible set.

| Key | Effect |
|---|---|
| `positive: true` | a softplus |
| `bounds: [lo, hi]` | a sigmoid into the interval |
| `monotone: {u: increasing}` | non-negative weights on every path from that input, for a closure such as a reaction rate that grows with concentration |
| `anchor: {at: {X: 0.05}, value: 0.0}` | a value the function is known to take at one point, held exactly by subtracting the network's own value there (a drying rate that vanishes at the equilibrium moisture, a fouling that starts at zero) |
| `initial` | the constant the function starts at; an anchored function starts at its anchor value |
| `truth` | an expression in the inputs: the function the synthetic data came from |

- **Anchor with `monotone`**: fixes the sign everywhere. A closure anchored at zero and increasing is non-negative above its anchor, without a softplus that would keep it from reaching zero.
- **`truth`**: used by `generate_synthetic_observations` and by an `exact_solution` that mentions the unknown. `pde.unknown_function_error("alpha")` grades the recovery as a relative $L^2$ error over the input range.

<!-- doctest: skip -->
```python
pde = CustomPDE(
    "u_t - alpha*u_xx",
    domain={"x": (0, 1), "t": (0, 1)},
    initial_condition="sin(pi*x)",
    boundary_conditions={"dirichlet": {"value": 0.0}},
    exact_solution="exp(-pi^2*(0.05*t + 0.05*t^2))*sin(pi*x)",   # for alpha = 0.05 + 0.1 t
    unknown_functions={
        "alpha": {"inputs": ["t"], "positive": True, "initial": 0.2, "truth": "0.05 + 0.1*t"},
    },
)
pde.generate_synthetic_observations(n_points=200, noise_std=0.01, seed=0)
trainer = pde.build_trainer("feedforward", num_epochs=2000)
trainer.train(progress=False)
print(pde.unknown_function_error("alpha"))          # relative L2 of the recovered alpha(t)
alpha_hat = pde.evaluate_unknown("alpha", x, t)     # the function itself, at any points
```

- **Fixed basis**: `{"alpha": {"inputs": ["x"], "basis": {"type": "cosine", "modes": 6}}}` (or `legendre`) replaces the network. The unknown is then the coefficient vector `fn.coefficients`, so each mode's identifiability can be analysed like a parameter and the coefficients can be sampled by the swarm (K48).
- **Closure of the field**: takes the field as input and a range to normalise it, `{"f": {"inputs": ["u"], "input_ranges": {"u": [-1, 1]}, "monotone": "increasing"}}`, used as `u_t - D*u_xx - f`.
- **No closed form**: pass a numerical reference as `exact_solution`. `pinnrl.benchmarks.heat_variable_reference(alpha)` solves `u_t = (alpha(x, t) u_x)_x` by the method of lines and `.as_exact_solution()` interpolates it.
- **Experiment [F09](benchmarks.md#field-valued-unknowns-experiment-f09)**: recovers a drifting `alpha(t)` and a two-layer `alpha(x)` this way, and shows which fields the data cannot determine.
- **Storage**: the weights are saved next to the model (`unknown_functions.pt`) and restored by `load_run`.
- **No weight decay**: on unknown functions or on scalar unknowns. Decay is a prior toward zero: a regularizer for network weights, a bias for physics.

## Measured inputs

A plant model is driven by what was recorded: an inlet temperature, a feed rate, the ambient conditions.

$$
u_t - \alpha\, u_{xx} - q(t) = 0, \qquad u(0, t) = T_{\text{in}}(t), \qquad u(1, t) = 0
$$

- $q(t)$, $T_{\text{in}}(t)$: records, given as `inputs` entries and used by name in the equations or in boundary values.
- **[`TimeSeries`](api/pdes.md#pdes.inputs.TimeSeries)**: a record, built from a CSV or NPZ file (`{"path": "left.csv", "time_column": "t", "value_column": "T_left"}`) or from arrays.
- **Reading**: by linear interpolation at the time of every collocation or boundary point; held constant outside the record.
- **Boundary values**: accept expressions in the coordinates, parameters and inputs, so `dirichlet: {left: "q", right: "0.2*t"}` and `neumann: {value: "sin(2*pi*t)"}` work alongside plain numbers.

<!-- doctest: skip -->
```python
pde = CustomPDE(
    "u_t - alpha*u_xx - q",                                   # a measured source
    domain={"x": (0, 1), "t": (0, 1)},
    parameters={"alpha": 0.1},
    inputs={"q": {"path": "source.csv", "time_column": "t", "value_column": "q"},
            "T_in": TimeSeries(t_samples, values)},
    boundary_conditions={"dirichlet": {"left": "T_in", "right": 0.0}},   # a measured boundary
    initial_condition="0",
)
```

- **YAML**: relative `path` entries are taken from the file's own folder.
- **Snapshot**: the run records the inputs by path (or inline when they came from arrays), so `load_run` rebuilds the driven problem.

## Sensors

A plant does not deliver the field at random points of space-time. A sensor reports

$$
y_k = h(u)\big|_{t_k - \tau} + \varepsilon_k
$$

- **Observation operator** $h$: the map from the field to the reading, written as an expression.
- $t_k$: the timestamps. $\varepsilon_k$: noise, with the sensor's own `noise` level as standard deviation.
- **`delay`** $\tau$: the record stamped `t` reflects the state at `t - delay`.
- **`weight`**: the sensor's weight in the loss.

| Sensor | Reports | Example |
|---|---|---|
| [`PointSensor`](api/pdes.md#pdes.sensors.PointSensor) | a sequence of readings at one place | a probe |
| [`BoundarySensor`](api/pdes.md#pdes.sensors.BoundarySensor) | an expression on a wall | a flux meter, `-alpha*u_x` |
| [`IntegralSensor`](api/pdes.md#pdes.sensors.IntegralSensor) | an integral over the domain | a level, a tank temperature |

- **Data term**: one mean-squared misfit per sensor, next to any scattered `observation_data`.
- **Scale**: in a [normalized](#non-dimensionalization) problem, the misfit of a sensor that reports a field is divided by that field's scale. Set `scale=` for any other expression.

<!-- doctest: skip -->
```python
import numpy as np
from pinnrl.pdes import PointSensor, BoundarySensor, IntegralSensor, sensor_jacobian
from pinnrl.pdes.catalog import from_catalog
from pinnrl.analysis import fisher_information
from pinnrl.pdes.sensors import sensor_noise_vector

times = np.linspace(0.05, 1.0, 40)
pde = from_catalog(
    "heat",
    trainable_parameters=["alpha"], parameter_initial_guesses={"alpha": 0.5},
    sensors=[
        PointSensor(0.3, times, noise=0.01, name="probe"),                         # u(0.3, t)
        BoundarySensor("left", times, "-alpha*u_x", noise=0.02, delay=0.05),       # wall flux, delayed
        IntegralSensor(times, "u", normalize=True, noise=0.01),                    # the mean over x
    ],
)
pde.observe(seed=0)                     # records from the true solution, through the operators
trainer = pde.build_trainer("feedforward", num_epochs=2000)

# Which layout determines alpha best? The Fisher information of the records, from the closed form.
J = sensor_jacobian(pde, ["alpha"])
print(fisher_information(J, sensor_noise_vector(pde.sensors), ["alpha"]).std)
```

- **Fisher information**: how much the records say about a parameter. The last two lines print the standard deviation of `alpha` this layout allows.
- `pde.evaluate_expression("-alpha*u_x", model, x, t)` is the same machinery for your own derived quantities.
- `sensor.set_values(records)` attaches real measurements.
- `pinnrl-train --pde-file` records synthetic ones when a spec's sensors carry none.
- Experiment [K02](benchmarks.md#sensor-layout-vs-conditioning-experiment-k02) uses `sensor_jacobian` to rank layouts.

After training, `pde.sensor_misfit(trainer.model)` is the one test the run can make on itself:

$$
\chi^2 = \sum_{k} \left( \frac{y_k - \hat{y}_k}{\sigma_k} \right)^2 \le \chi^2_{0.999}
$$

- **Chi-square** $\chi^2$: the misfit of the records $y_k$ under the network's readings $\hat{y}_k$, in units of each sensor's noise, against its 99.9 % limit ([`SensorMisfit`](api/pdes.md#pdes.sensors.SensorMisfit)).
- A twin whose fields miss the records has also missed the unknown: every failed run of J06 and J07b fails this test.

[`train_with_restarts`](api/training.md#training.restarts.train_with_restarts) trains and tests; while the test fails it starts again from the next seed, and it keeps the best of the attempts.

<!-- doctest: skip -->
```python
from pinnrl.training.restarts import train_with_restarts

result = train_with_restarts(
    pde,
    lambda seed: pde.build_trainer("feedforward", num_epochs=2000),
    lambda trainer: trainer.train(progress=False),
    attempts=3,
)
print(result.misfit, result.accepted, result.n_attempts)
```

## Parametric surrogates

- **Parametric surrogate**: a network with the parameter as an extra input. One training covers a range of it, and any value in the range costs one forward pass.

1. List the parameter under `parametric` with its range: `from_catalog("heat", parameters={}, parametric={"alpha": (0.02, 0.5)})`.
2. Train. `alpha` is sampled uniformly with the collocation points and used in the expressions like a coordinate.
3. Call `pde.surrogate_misfit(model, x, t, y, thetas)` for the data misfit of a batch of parameter values.

- Boundary and initial conditions apply to the physical axes only.
- `surrogate_misfit` is the vectorized forward model a swarm (`pinnrl.uq.sample_equivalence_region`) or any sampler needs.
- Experiment K45 measures what such a surrogate loses against per-value training, and uses it as the swarm's forward model.

## Non-dimensionalization

A network expects inputs and outputs of order one. A problem in metres, seconds and kelvin does not supply them.

$$
\hat{x} = 2\, \frac{x - x_{\text{lo}}}{x_{\text{hi}} - x_{\text{lo}}} - 1, \qquad u = u_0 + s \cdot \text{net}(\hat{x}, \hat{t})
$$

- $\hat{x}, \hat{t}$: the domain mapped to $[-1, 1]$.
- $u_0$, $s$: the **offset** and **scale** of a field, from `scales` (a scale, or an `(offset, scale)` pair).

1. Give each field its magnitude with `scales`.
2. Pass `normalize=True` to `build_trainer` (`--normalize` on the command line).

The loss terms are taken in those units too:

- Initial, boundary, data and sensor errors are divided by $s$.
- The residual is multiplied by the time span over $s$ (the magnitude of $u_t$).
- The equations stay as written, in physical units, and the trained model answers in them.

A heat problem on a 1,000 mm slab over an hour, with temperatures around 300 K:

<!-- doctest: skip -->
```python
pde = CustomPDE(
    "u_t - alpha*u_xx",                                      # mm^2/s, mm, s, K
    domain={"x": (0, 1000.0), "t": (0, 3600.0)},
    parameters={"alpha": 50.0},
    initial_condition="300 + 50*sin(pi*x/1000)",
    boundary_conditions={"dirichlet": {"value": 300.0}},
    scales={"u": (300.0, 50.0)},                             # around 300 K, varying by 50 K
)
trainer = pde.build_trainer("feedforward", normalize=True, num_epochs=1000)
```

- Result (`test_normalization.py`): after 1,000 Adam steps the maximum error is 248 K for the raw model and 2 K for the normalized one. The same problem on the unit square trains to a few percent.
- The run snapshot keeps the bounds and scales in its `model` block, so `load_run` rebuilds the scaled model.

## A faster step: the compiled kernel

[`pinnrl.native`](api/native.md) is a small C++ kernel, compiled on first use, that takes the residual's derivatives and the weight gradient without autograd. The mechanism is in [The native engine](engine.md).

**What it is**

- **General kernel**: carries the derivatives forward through the layers next to the value, then takes the weight gradient by one reverse pass.
- **Compiled step**: C++ written for your equations, compiled once (about 1.5 s per equation and machine) and cached. It takes over at evaluation 32, so a short run never waits for the compiler.
- **Default**: `build_trainer` and `pinnrl-train` use it wherever it applies and autograd everywhere else, without notice. Loss and gradients are the same to rounding.

**When it applies**

- **General kernel**: equations given as text (or a registry class such as `HeatEquation`), derivatives of any order, no unknown functions; a feed-forward network of `Linear` and `tanh`, with or without `LayerNorm`, float32 on the CPU; macOS (Accelerate) or Linux (OpenBLAS, `apt install libopenblas-dev`) with a C++ compiler.
- **Compiled step**: also needs a time-dependent problem and the squared loss. It covers trainable parameters, measured inputs, unknown functions of the coordinates, `normalize=True` and layer normalization.
- **Stays on autograd**: an unknown function of a field, or a derivative of one.

The full list is in [engine.md](engine.md#where-it-does-not-apply).

**How to switch it off, and how to check**

<!-- doctest: skip -->
```python
trainer = pde.build_trainer("feedforward")
print(pde.native_active(trainer.model))      # True where the kernel runs; autograd otherwise

trainer = pde.build_trainer("feedforward", derivatives="autograd")   # never the kernel
```

- `derivatives="autograd"`: never the kernel. From the command line: `pinnrl-train --pde-file spec.yaml --derivatives autograd`.
- `derivatives="native"`: the earlier name of the default, still accepted.
- `PINNRL_NATIVE_SPECIALIZE=off`: keeps the general kernel. An integer sets the number of evaluations before the switch.
- [engine.md](engine.md#which-engine-ran) shows how to tell the compiled step apart.

**Measured step times**

- [I11](benchmarks.md#the-cost-of-a-training-step-pytorch-c-and-rust-experiment-i11): heat with the packaged network (layer normalization on), 32 ms by autograd, 10.4 ms by the kernel.
- [I17](benchmarks.md#the-step-against-the-alternatives-experiment-i17): heat with a 4 × 64 network, 4.8 ms by the general kernel, 3.7 ms by the compiled step. On an Apple M4 the compiled step is ahead of JAX on every equation measured; on Linux x86 JAX's forward mode is a few percent ahead on plain networks at low order.

**What the rest of the package sees**

Nothing downstream has to know which engine ran. A gradient that is itself differentiated (`create_graph=True`, as in the Hessian spectrum and the sensitivity analyses of `pinnrl.analysis`) is taken by autograd on the same function.

## From the command line

1. Write the same specification as YAML.
2. Pass it with `--pde-file` in place of `--pde`.
3. Use every other flag as usual, including `--mode inverse --identify r`.

The Fisher–KPP equation:

```yaml
# fisher_kpp.yaml
name: Fisher-KPP
equations: "u_t - D*u_xx - r*u*(1 - u)"
domain:
  x: [-10, 10]
  t: [0, 4]
parameters: {D: 1.0, r: 1.0}
initial_condition: "1/(1 + exp(x))"
boundary_conditions:
  dirichlet: {left: 1.0, right: 0.0}
```

```bash
pinnrl-train --pde-file fisher_kpp.yaml --arch fourier --epochs 2000
```

- The run folder's `config.yaml` keeps the specification, so `load_run` and the dashboard rebuild the problem.
- `CustomPDE.from_yaml(path)` and `pde.to_dict()` convert in both directions.

## Examples of every kind

Each is a complete, trainable problem and an entry of the shipped catalogue (next section), so `from_catalog(key)` gives the same object.

**A nonlinear scalar PDE** (Burgers; shock formation for small ν):

$$
u_t + u\, u_x - \nu\, u_{xx} = 0
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("u_t + u*u_x - nu*u_xx", domain={"x": (-1, 1), "t": (0, 1)},
                parameters={"nu": 0.01 / 3.14159}, initial_condition="-sin(pi*x)",
                boundary_conditions={"dirichlet": {"value": 0.0}})
```

**Second order in time, with an initial velocity** (telegraph equation, exact solution known):

$$
u_{tt} + 2 \gamma\, u_t - c^2 u_{xx} = 0
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("u_tt + 2*gamma*u_t - c^2*u_xx", domain={"x": (0, 1), "t": (0, 1)},
                parameters={"c": 1.0, "gamma": 0.5},
                initial_condition="sin(pi*x)", initial_velocity="-gamma*sin(pi*x)",
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="exp(-gamma*t)*cos(sqrt(c^2*pi^2 - gamma^2)*t)*sin(pi*x)")
```

**A third-order dispersive equation** (KdV soliton):

$$
u_t + 6\, u\, u_x + u_{xxx} = 0
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("u_t + 6*u*u_x + u_xxx", domain={"x": (-15, 15), "t": (0, 2)},
                parameters={"c": 1.0}, initial_condition="c/2/cosh(sqrt(c)/2*x)^2",
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="c/2/cosh(sqrt(c)/2*(x - c*t))^2")
```

**A steady boundary-value problem** (Helmholtz in 2-D; no `t`, no initial condition):

$$
u_{xx} + u_{yy} + k^2 u = (k^2 - 2 \pi^2) \sin \pi x \, \sin \pi y
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("u_xx + u_yy + k^2*u - (k^2 - 2*pi^2)*sin(pi*x)*sin(pi*y)",
                domain={"x": (0, 1), "y": (0, 1)}, parameters={"k": 4.0},
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="sin(pi*x)*sin(pi*y)")
```

**An ODE** (logistic growth; the field is `N`, the only coordinate is `t`):

$$
\dot{N} = r\, N \left( 1 - N / K \right)
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("N_t - r*N*(1 - N/K)", domain={"t": (0, 10)}, fields=["N"],
                parameters={"r": 1.0, "K": 10.0, "N0": 1.0}, initial_condition="N0",
                exact_solution="K/(1 + (K/N0 - 1)*exp(-r*t))")
```

**A system of ODEs** (SIR epidemic; one equation per field, initial values per field):

$$
\dot{S} = -\beta S I, \qquad \dot{I} = \beta S I - \gamma I, \qquad \dot{R} = \gamma I
$$

<!-- doctest: skip -->
```python
pde = CustomPDE(["S_t + beta*S*I", "I_t - beta*S*I + gamma*I", "R_t - gamma*I"],
                domain={"t": (0, 50)}, fields=["S", "I", "R"],
                parameters={"beta": 0.3, "gamma": 0.1},
                initial_condition={"S": "0.99", "I": "0.01", "R": "0"})
```

**A process model** (a continuous stirred-tank reactor with an Arrhenius rate $k$, which grows with temperature):

$$
\dot{C} = \frac{C_{\text{in}} - C}{\tau} - k\, C, \qquad \dot{T} = \frac{T_{\text{in}} - T}{\tau} + \Delta H\, k\, C - UA\, (T - T_c), \qquad k = k_0\, e^{-E_a / (R T)}
$$

<!-- doctest: skip -->
```python
pde = CustomPDE(["C_t - (Cin - C)/tau + k0*exp(-Ea/(R*max(T, 0.1)))*C",
                 "T_t - (Tin - T)/tau - dH*k0*exp(-Ea/(R*max(T, 0.1)))*C + UA*(T - Tc)"],
                domain={"t": (0, 5)}, fields=["C", "T"],
                parameters={"Cin": 1.0, "Tin": 1.0, "tau": 1.0, "k0": 5.0, "Ea": 2.0, "R": 1.0,
                            "dH": 0.5, "UA": 0.5, "Tc": 1.0},
                initial_condition={"C": "1", "T": "1"})
```

**A coupled PDE system with a constraint** (unsteady Stokes flow; the third equation has no time derivative):

$$
u_t + p_x = \nu \nabla^2 u, \qquad v_t + p_y = \nu \nabla^2 v, \qquad u_x + v_y = 0
$$

<!-- doctest: skip -->
```python
pde = CustomPDE(["u_t + p_x - nu*(u_xx + u_yy)", "v_t + p_y - nu*(v_xx + v_yy)", "u_x + v_y"],
                domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)}, fields=["u", "v", "p"],
                parameters={"nu": 0.1},
                initial_condition={"u": "sin(2*pi*x)*cos(2*pi*y)", "v": "-cos(2*pi*x)*sin(2*pi*y)"},
                boundary_conditions={"periodic": {}})
```

**Many dimensions** (heat in five space dimensions, six network inputs; name the coordinates and the subscripts follow):

$$
u_t - \alpha \sum_{i=1}^{5} u_{x_i x_i} = 0
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("u_t - alpha*(u_x1x1 + u_x2x2 + u_x3x3 + u_x4x4 + u_x5x5)",
                domain={f"x{i}": (0, 1) for i in range(1, 6)} | {"t": (0, 1)},
                parameters={"alpha": 0.02},
                initial_condition="sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4)*sin(pi*x5)",
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="exp(-5*alpha*pi^2*t)*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4)*sin(pi*x5)")
```

**Finance** (Black–Scholes in time to maturity; the coordinate is the asset price `S`, the field the option value `V`):

$$
V_t - \tfrac{1}{2} \sigma^2 S^2 V_{SS} - r\, S\, V_S + r\, V = 0
$$

<!-- doctest: skip -->
```python
pde = CustomPDE("V_t - 0.5*sigma^2*S^2*V_SS - r*S*V_S + r*V", domain={"S": (1, 200), "t": (0, 1)},
                fields=["V"], parameters={"sigma": 0.2, "r": 0.05, "K": 100.0},
                initial_condition="max(S - K, 0)",
                boundary_conditions={"dirichlet": {"left": 0.0, "right": 100.0}})
```

## The catalogue

`pinnrl.pdes.catalog.CATALOG` holds forty-three such specifications:

| Family | Entries |
|---|---|
| Parabolic | heat in 1-D to 5-D, with a source, advection–diffusion, Fisher–KPP, Allen–Cahn, Cahn–Hilliard, Burgers, porous medium, Black–Scholes |
| Hyperbolic | wave in 1-D and 2-D, telegraph, convection, Klein–Gordon, sine–Gordon |
| Dispersive | KdV with one or two coefficients, the Euler–Bernoulli beam, Schrödinger as a real system |
| Steady | Poisson in 2-D and 4-D, Helmholtz, a convection–diffusion boundary layer, Bratu |
| ODEs | logistic, damped oscillator, pendulum, Lotka–Volterra, Van der Pol, Lorenz, SIR, a CSTR |
| Systems | a coupled reaction–diffusion pair with a closed form, Gray–Scott, Brusselator, linear shallow water, Stokes |
| Other | a three-state Hamilton–Jacobi–Bellman equation |

```python
from pinnrl.pdes.catalog import CATALOG, from_catalog

pde = from_catalog("kdv_soliton")
pde = from_catalog("burgers", parameters={"nu": 0.1})   # override any entry
print(len(CATALOG), sorted(CATALOG)[:5])
```

Every entry with a closed-form solution is checked by the test suite: the exact solution, used as the network, must zero the residual and match the initial condition. The catalogue is thus also the regression test of the expression framework.

## Limits

- The derivative notation applies to fields, not to expressions: write `(u^2)_xx` as `2*u_x^2 + 2*u*u_xx`, or use a callable residual.
- Where the [compiled kernel](#a-faster-step-the-compiled-kernel) does not apply, derivatives are taken by nested autograd. Order four and above is then slow and numerically noisy, as for the shipped Cahn–Hilliard equation. Cost grows with the number of coordinates and with the order, so a 5-D Laplacian is five second-derivative passes.
- A boundary condition applies to both faces of an axis alike (Dirichlet takes `left`/`right` values); conditions that differ by face in kind, or Robin conditions, need a callable residual or a subclass.
- An unknown function is a smooth network: a jump in the true coefficient is recovered as a steep ramp, and a field the data barely feel (see F09) collapses to the constant that fits best. Check identifiability before trusting the recovered shape.
- Without `normalize=True` the network sees raw coordinates and raw field values; a problem in physical units needs the [non-dimensionalization](#non-dimensionalization) step, or hand scaling.
- Integral terms, delays and non-local operators are outside the expression language.
