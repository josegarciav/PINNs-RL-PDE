# Custom equations

Any PDE, ODE or system can be written as text and trained with the same loop, samplers, optimizers and inverse-problem machinery as the shipped equations. `CustomPDE` parses the residual with [SymPy](https://www.sympy.org), takes every derivative by autograd, and is a regular `PDEBase`, so `PDETrainer`, `pinnrl-train` and the dashboard treat it like any other problem.

## A first equation

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

The residual is the left-hand side of `expression = 0`. `build_trainer` returns a `PDETrainer` with the packaged defaults (`config.yaml`) for the named architecture, and keyword arguments override keys of the `training` block (`num_epochs`, `batch_size`, `num_collocation_points`, `num_boundary_points`, `num_initial_points`, `learning_rate`, `weight_decay`, `optimizer`, `loss_weights`, `mode`, ...); `model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False}` overrides the architecture block. You can equally build a `PINNModel` and `PDETrainer` yourself, as in [Getting started](getting_started.md#python-api).

## Notation

| Written as | Meaning |
|---|---|
| `u`, `v`, `theta` | a field (network output), as listed in `fields` (default `["u"]`) |
| `x`, `y`, `z`, `r`, `t` | a coordinate (network input), as listed in `domain`; `t` is time |
| `u_t`, `u_xx`, `u_xy`, `theta_tt` | partial derivatives: field, underscore, one coordinate name per differentiation, any order |
| `nu`, `alpha`, `D`, `k` | a parameter: any other name, supplied in `parameters` |
| `sin cos tan exp log sqrt tanh sinh cosh abs sign erf min max atan2 sigmoid relu`, `pi` | functions and the constant π; `^` or `**` is a power; `2*x`, not `2 x` |

Coordinates with several letters work (`u_theta`), as long as the subscript decomposes into the coordinate names in one way. A name that reads like a derivative with an unknown coordinate (`u_zz` when `z` is not in the domain) is an error, and so is a parameter that the equations use but `parameters` does not define.

## Domains and kinds of problem

`domain` maps each coordinate to `(lo, hi)`. Spatial coordinates come first, in the order given; `t` is the time axis.

- **Time-dependent PDE:** `{"x": (0, 1), "t": (0, 1)}`, with an `initial_condition`.
- **Second order in time** (wave, oscillators): detected from `_tt` in the equations; `initial_velocity` gives `u_t(x, t0)` (default 0).
- **ODE:** a domain with only `t`, e.g. `{"t": (0, 10)}`. The network receives a dummy spatial input that the solution does not depend on, as the shipped pendulum does.
- **Steady problem** (Poisson, Helmholtz): a domain without `t`. There is no initial term and `t` is fixed at 0.
- **Higher dimension:** `{"x": (0, 1), "y": (0, 1), "t": (0, 2)}`; `u_xx + u_yy` is the Laplacian.

## Initial, boundary and exact solutions

`initial_condition`, `initial_velocity` and `exact_solution` are expressions in the coordinates (the exact solution may use `t`) and the parameters: one string for a scalar problem, a list or `{field: expression}` for a system (fields left out are 0). Each can instead be a callable, `u0(x)` or `u(x, t)` returning an `(N, k)` tensor, when there is no closed form (a numerical reference, interpolated data).

`boundary_conditions` are the generic ones of `PDEBase`, applied to every field on every face: `{"dirichlet": {"value": 0.0}}` or `{"left": 1.0, "right": 0.0}`, `{"neumann": {"value": 0.0}}`, `{"periodic": {}}`, and `{"dirichlet": {"type": "exact"}}` to take values from the exact solution. Per-field conditions are on the [roadmap](roadmap.md).

## Systems

Give one equation per field, in the order of `fields`. A Gray–Scott reaction–diffusion system in 2-D:

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

The residual has shape `(N, 2)`; the residual loss averages over both equations, and residual-based sampling uses the norm over the equations at each point.

Boundary conditions given as `{kind: options}` apply to every field. A system whose fields obey different conditions keys them by field, `boundary_conditions={"u": {"dirichlet": {"value": 1.0}}, "v": {"neumann": {"value": 0.0}}}`; a field left out has none. When one component is small or slow and would otherwise be ignored by the optimizer, `field_weights={"v": 10.0}` multiplies that field's residual, initial and boundary errors in the loss. The catalogue entry `reaction_diffusion_pair`, two heat equations exchanging mass at rate `k`, has a closed form and is the test case for both.

## A residual in Python

When the expression language is not enough (a lookup table, a learned closure, a conditional), pass a callable `residual(f, p)`. `f` is a `Fields` accessor with the same notation as attributes (`f.u`, `f.u_xx`, `f.x`, `f.t`, `f.d("u", "x", "t")`, `f.laplacian("u")`), each derivative computed once by autograd and cached; `p` is the parameter dict, with `nn.Parameter` values for the ones being identified. Return an `(N, k)` tensor or a list of `(N, 1)` tensors.

<!-- doctest: skip -->
```python
import torch

def residual(f, p):
    source = torch.where(f.x < 0.5, 1.0, 0.0)          # a piecewise forcing
    return f.u_t - p["alpha"] * f.u_xx - source

pde = CustomPDE(residual, domain={"x": (0, 1), "t": (0, 1)}, parameters={"alpha": 0.1},
                initial_condition="0", boundary_conditions={"dirichlet": {"value": 0.0}})
```

Pass `time_order=2` with a callable that is second order in time; for text it is detected.

## Inverse problems

Any parameter can be identified: list it in `trainable_parameters` with an initial guess, attach observations (`observation_data={"x": ..., "t": ..., "u": ...}` or `{"path": "obs.npz"}`), and train in `mode="inverse"` (`build_trainer` selects it when there are trainable parameters). Without observations and with an exact solution, `generate_synthetic_observations` creates noisy ones. See [Inverse problems](inverse_problems.md).

The optimizer takes the same absolute step for a PDE parameter as for a weight, so a small positive parameter such as a viscosity of $10^{-3}$ is better identified through its logarithm: write `exp(log_nu)` in the residual and list `log_nu` as trainable. The step becomes relative and the viscosity stays positive. With a closed-form solution, `pinnrl.analysis.closed_form_bound(pde, ["alpha"], x, t, sigma)` gives the Cramér–Rao bound on the parameter from observations at `(x, t)` with noise `sigma`, the floor any unbiased estimate can reach.

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

## Unknown functions

When the unknown physics is a function rather than a number, a coefficient that drifts in time, a heterogeneous field or a closure of the solution, give it as an `unknown_functions` entry and use its name in the equation like a parameter. It becomes an [`UnknownFunction`](api/pdes.md#pdes.unknowns.UnknownFunction), a small network of the inputs you name (coordinates or fields), whose weights train with the data in inverse mode. The subscript notation applies to it too: `alpha_x` is the derivative of the learned function, by autograd, so the conservative form of a heterogeneous diffusion is `u_t - alpha_x*u_x - alpha*u_xx`.

Constraints the physics imposes are built into the output, so training never leaves the admissible set: `positive: true` (a softplus), `bounds: [lo, hi]` (a sigmoid into the interval), `monotone: {u: increasing}` (non-negative weights on every path from that input, for a closure such as a reaction rate that grows with concentration) and `anchor: {at: {X: 0.05}, value: 0.0}`, a value the function is known to take at one point, held exactly by subtracting the network's own value there (a drying rate that vanishes at the equilibrium moisture, a fouling that starts at zero). An anchor with `monotone` fixes the sign everywhere: a closure anchored at zero and increasing is non-negative above its anchor, without a softplus that would keep it from reaching zero. `initial` is the constant the function starts at, the counterpart of a scalar's initial guess; an anchored function starts at its anchor value. A `truth` expression in the inputs records the function the synthetic data came from: `generate_synthetic_observations` and an `exact_solution` that mentions the unknown use it, and `pde.unknown_function_error("alpha")` grades the recovery as a relative L2 error over the input range.

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

A field can also be written in a fixed basis instead of a network, `{"alpha": {"inputs": ["x"], "basis": {"type": "cosine", "modes": 6}}}` (or `legendre`): the unknown is then the coefficient vector `fn.coefficients`, one scalar per mode, so each mode's identifiability can be analysed like a parameter and the coefficients can be sampled by the swarm (K48).

A closure of the field takes the field as input and a range to normalise it: `{"f": {"inputs": ["u"], "input_ranges": {"u": [-1, 1]}, "monotone": "increasing"}}`, used as `u_t - D*u_xx - f`. Problems without a closed form take a numerical reference as `exact_solution`; `pinnrl.benchmarks.heat_variable_reference(alpha)` solves `u_t = (alpha(x, t) u_x)_x` by the method of lines and `.as_exact_solution()` interpolates it. The experiment [F09](benchmarks.md#field-valued-unknowns-experiment-f09) recovers a drifting `alpha(t)` and a two-layer `alpha(x)` this way, and shows which fields the data cannot determine.

The weights of unknown functions are saved next to the model (`unknown_functions.pt`) and restored by `load_run`. The optimizer applies no weight decay to them, nor to scalar unknowns: decay is a prior toward zero, which is a regularizer for network weights and a bias for physics.

## Measured inputs

A plant model is driven by what was recorded: an inlet temperature, a feed rate, the ambient conditions. Give such a record as an `inputs` entry, a [`TimeSeries`](api/pdes.md#pdes.inputs.TimeSeries) built from a CSV or NPZ file (`{"path": "left.csv", "time_column": "t", "value_column": "T_left"}`) or from arrays, and use its name in the equations or in boundary values. It is read at the time of every collocation or boundary point by linear interpolation and held constant outside the record. Boundary values accept expressions in the coordinates, parameters and inputs, so `dirichlet: {left: "q", right: "0.2*t"}` and `neumann: {value: "sin(2*pi*t)"}` work alongside plain numbers.

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

In a YAML specification, relative `path` entries are taken from the file's own folder. The run snapshot records the inputs by path (or inline when they came from arrays), so `load_run` rebuilds the driven problem.

## Sensors

A plant does not deliver the field at random points of space-time. A probe sits at one place and reports a sequence of readings; a flux meter reports `-alpha*u_x` on a wall; a level or a tank temperature is an integral over the domain. The `sensors` argument describes such observation operators ([`PointSensor`](api/pdes.md#pdes.sensors.PointSensor), [`BoundarySensor`](api/pdes.md#pdes.sensors.BoundarySensor), [`IntegralSensor`](api/pdes.md#pdes.sensors.IntegralSensor)), each with the expression it reports, its timestamps, its own noise level, a `delay` (the record stamped `t` reflects the state at `t - delay`) and a `weight` in the loss. Their records enter the data term as one mean-squared misfit per sensor, next to any scattered `observation_data`; in a [normalized](#non-dimensionalization) problem the misfit of a sensor that reports a field is divided by that field's scale (set `scale=` for any other expression).

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

`pde.evaluate_expression("-alpha*u_x", model, x, t)` is the same machinery for your own derived quantities. `sensor.set_values(records)` attaches real measurements; `pinnrl-train --pde-file` records synthetic ones when a spec's sensors carry none. Experiment [K02](benchmarks.md#sensor-layout-vs-conditioning-experiment-k02) uses `sensor_jacobian` to rank layouts.

After training, `pde.sensor_misfit(trainer.model)` is the one test the run can make on itself: the chi-square of the records under the network, in units of each sensor's noise, against its 99.9 % limit ([`SensorMisfit`](api/pdes.md#pdes.sensors.SensorMisfit)). A twin whose fields miss the records has also missed the unknown (every failed run of J06 and J07b fails this test), so [`train_with_restarts`](api/training.md#training.restarts.train_with_restarts) trains, tests, and starts again from the next seed while the test fails, keeping the best of its attempts:

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

One training can cover a whole range of a parameter: list it under `parametric` with its range, `from_catalog("heat", parameters={}, parametric={"alpha": (0.02, 0.5)})`, and `alpha` becomes an extra input of the network, sampled uniformly with the collocation points and used in the expressions like a coordinate. Boundary and initial conditions apply to the physical axes only. The trained surrogate `u(x, t; alpha)` answers any `alpha` in the range at the cost of one forward pass, and `pde.surrogate_misfit(model, x, t, y, thetas)` returns the data misfit for a batch of parameter values, the vectorized forward model a swarm (`pinnrl.uq.sample_equivalence_region`) or any sampler needs. Experiment K45 measures what such a surrogate loses against per-value training, and uses it as the swarm's forward model.

## Non-dimensionalization

A problem written in its own units, metres, seconds and kelvin, is a bad problem for a network that expects inputs and outputs of order one: the raw heat problem below, on a 1,000 mm slab over an hour with temperatures around 300 K, trains to a 250 K error where the same problem on the unit square trains to a few percent. Give each field its magnitude with `scales` (a scale, or an `(offset, scale)` pair) and pass `normalize=True` to `build_trainer` (`--normalize` on the command line). The network then sees the domain mapped to $[-1, 1]$ and predicts `offset + scale * net`, and the loss terms are taken in those units too: initial, boundary, data and sensor errors divided by the field's scale, the residual multiplied by the time span over the scale (the magnitude of $u_t$). The equations stay as written, in physical units, and the trained model answers in them.

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

With 1,000 Adam steps the normalized model's maximum error is 2 K against 248 K for the raw one (`test_normalization.py`). The run snapshot keeps the bounds and scales in its `model` block, so `load_run` rebuilds the scaled model.

## A faster step: the compiled kernel

For a plain `tanh` network the derivatives a residual needs can be carried forward through the layers next to the value, and the weight gradient taken by one reverse pass, instead of differentiating by autograd twice and backpropagating through that graph. [`pinnrl.native`](api/native.md) does this in a small C++ kernel, compiled on first use, and `derivatives="native"` switches a problem to it:

<!-- doctest: skip -->
```python
trainer = pde.build_trainer("feedforward", model={"layer_norm": False}, derivatives="native")
print(pde.native_active(trainer.model))      # True where the kernel applies; autograd otherwise
```

On the heat problem of experiment [I11](benchmarks.md#the-cost-of-a-training-step-pytorch-c-and-rust-experiment-i11) the step goes from 10.6 to 5.4 ms with the same loss and gradients to rounding. The kernel covers: macOS with a C++ compiler; a feed-forward network of `Linear` and `tanh` without layer normalization, float32 on the CPU; equations given as text, with first and pure second derivatives (`u_x`, `u_t`, `u_xx`, a Laplacian; not `u_xt` or `u_xxx`) and no unknown functions. Anywhere else the residual is computed by autograd, without notice, so the option is safe to leave on. Its weight gradient cannot be differentiated again: the Hessian and sensitivity analyses of `pinnrl.analysis` need the default `derivatives="autograd"`. From the command line: `pinnrl-train --pde-file spec.yaml --derivatives native`.

## From the command line

Write the same specification as YAML and pass it with `--pde-file` in place of `--pde`. Every other flag works as usual, including `--mode inverse --identify r`.

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

The run folder's `config.yaml` keeps the specification, so `load_run` and the dashboard rebuild the problem. `CustomPDE.from_yaml(path)` and `pde.to_dict()` convert in both directions.

## Examples of every kind

Each of these is a complete, trainable problem. They are all entries of the shipped catalogue (next section), so `from_catalog(key)` gives the same object.

**A nonlinear scalar PDE** (Burgers; shock formation for small ν):

<!-- doctest: skip -->
```python
pde = CustomPDE("u_t + u*u_x - nu*u_xx", domain={"x": (-1, 1), "t": (0, 1)},
                parameters={"nu": 0.01 / 3.14159}, initial_condition="-sin(pi*x)",
                boundary_conditions={"dirichlet": {"value": 0.0}})
```

**Second order in time, with an initial velocity** (telegraph equation, exact solution known):

<!-- doctest: skip -->
```python
pde = CustomPDE("u_tt + 2*gamma*u_t - c^2*u_xx", domain={"x": (0, 1), "t": (0, 1)},
                parameters={"c": 1.0, "gamma": 0.5},
                initial_condition="sin(pi*x)", initial_velocity="-gamma*sin(pi*x)",
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="exp(-gamma*t)*cos(sqrt(c^2*pi^2 - gamma^2)*t)*sin(pi*x)")
```

**A third-order dispersive equation** (KdV soliton; derivatives of any order are one subscript):

<!-- doctest: skip -->
```python
pde = CustomPDE("u_t + 6*u*u_x + u_xxx", domain={"x": (-15, 15), "t": (0, 2)},
                parameters={"c": 1.0}, initial_condition="c/2/cosh(sqrt(c)/2*x)^2",
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="c/2/cosh(sqrt(c)/2*(x - c*t))^2")
```

**A steady boundary-value problem** (Helmholtz in 2-D; no `t`, no initial condition):

<!-- doctest: skip -->
```python
pde = CustomPDE("u_xx + u_yy + k^2*u - (k^2 - 2*pi^2)*sin(pi*x)*sin(pi*y)",
                domain={"x": (0, 1), "y": (0, 1)}, parameters={"k": 4.0},
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="sin(pi*x)*sin(pi*y)")
```

**An ODE** (logistic growth; the field is `N`, the only coordinate is `t`):

<!-- doctest: skip -->
```python
pde = CustomPDE("N_t - r*N*(1 - N/K)", domain={"t": (0, 10)}, fields=["N"],
                parameters={"r": 1.0, "K": 10.0, "N0": 1.0}, initial_condition="N0",
                exact_solution="K/(1 + (K/N0 - 1)*exp(-r*t))")
```

**A system of ODEs** (SIR epidemic; one equation per field, initial values per field):

<!-- doctest: skip -->
```python
pde = CustomPDE(["S_t + beta*S*I", "I_t - beta*S*I + gamma*I", "R_t - gamma*I"],
                domain={"t": (0, 50)}, fields=["S", "I", "R"],
                parameters={"beta": 0.3, "gamma": 0.1},
                initial_condition={"S": "0.99", "I": "0.01", "R": "0"})
```

**A process model** (a continuous stirred-tank reactor with an Arrhenius rate; parameters named as an engineer would):

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

<!-- doctest: skip -->
```python
pde = CustomPDE(["u_t + p_x - nu*(u_xx + u_yy)", "v_t + p_y - nu*(v_xx + v_yy)", "u_x + v_y"],
                domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)}, fields=["u", "v", "p"],
                parameters={"nu": 0.1},
                initial_condition={"u": "sin(2*pi*x)*cos(2*pi*y)", "v": "-cos(2*pi*x)*sin(2*pi*y)"},
                boundary_conditions={"periodic": {}})
```

**Many dimensions** (heat in five space dimensions, six network inputs; name the coordinates and the subscripts follow):

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

<!-- doctest: skip -->
```python
pde = CustomPDE("V_t - 0.5*sigma^2*S^2*V_SS - r*S*V_S + r*V", domain={"S": (1, 200), "t": (0, 1)},
                fields=["V"], parameters={"sigma": 0.2, "r": 0.05, "K": 100.0},
                initial_condition="max(S - K, 0)",
                boundary_conditions={"dirichlet": {"left": 0.0, "right": 100.0}})
```

## The catalogue

`pinnrl.pdes.catalog.CATALOG` holds forty-three such specifications: parabolic (heat in 1-D to 5-D, with a source, advection–diffusion, Fisher–KPP, Allen–Cahn, Cahn–Hilliard, Burgers, porous medium, Black–Scholes), hyperbolic (wave in 1-D and 2-D, telegraph, convection, Klein–Gordon, sine–Gordon), dispersive (KdV with one or two coefficients, the Euler–Bernoulli beam, Schrödinger as a real system), steady (Poisson in 2-D and 4-D, Helmholtz, a convection–diffusion boundary layer, Bratu), ODEs (logistic, damped oscillator, pendulum, Lotka–Volterra, Van der Pol, Lorenz, SIR, a CSTR), systems (a coupled reaction–diffusion pair with a closed form, Gray–Scott, Brusselator, linear shallow water, Stokes) and a three-state Hamilton–Jacobi–Bellman equation.

```python
from pinnrl.pdes.catalog import CATALOG, from_catalog

pde = from_catalog("kdv_soliton")
pde = from_catalog("burgers", parameters={"nu": 0.1})   # override any entry
print(len(CATALOG), sorted(CATALOG)[:5])
```

Every entry with a closed-form solution is checked by the test suite: the exact solution, used as the network, must zero the residual and match the initial condition. That makes the catalogue the regression test of the expression framework as well as a set of ready problems.

## Limits

- The derivative notation applies to fields, not to expressions: write `(u^2)_xx` as `2*u_x^2 + 2*u*u_xx`, or use a callable residual.

- Derivatives are taken by nested autograd; order four and above is slow and numerically noisy, as for the shipped Cahn–Hilliard equation. Cost grows with the number of coordinates and with the order, so a 5-D Laplacian is five second-derivative passes.
- A boundary condition applies to both faces of an axis alike (Dirichlet takes `left`/`right` values); conditions that differ by face in kind, or Robin conditions, need a callable residual or a subclass.
- An unknown function is a smooth network: a jump in the true coefficient is recovered as a steep ramp, and a field the data barely feel (see F09) collapses to the constant that fits best. Check identifiability before trusting the recovered shape.
- Without `normalize=True` the network sees raw coordinates and raw field values; a problem in physical units needs the [non-dimensionalization](#non-dimensionalization) step, or hand scaling.
- Integral terms, delays and non-local operators are outside the expression language.
