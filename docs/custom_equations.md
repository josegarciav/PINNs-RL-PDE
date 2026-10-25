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

The residual is the left-hand side of `expression = 0`. `build_trainer` returns a `PDETrainer` with the packaged defaults (`config.yaml`) for the named architecture, and keyword arguments override keys of the `training` block (`num_epochs`, `batch_size`, `num_collocation_points`, `num_boundary_points`, `num_initial_points`, `learning_rate`, `weight_decay`, `optimizer`, `loss_weights`, `mode`, ...). You can equally build a `PINNModel` and `PDETrainer` yourself, as in [Getting started](getting_started.md#python-api).

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

`pinnrl.pdes.catalog.CATALOG` holds forty-one such specifications: parabolic (heat in 1-D to 5-D, with a source, advection–diffusion, Fisher–KPP, Allen–Cahn, Cahn–Hilliard, Burgers, porous medium, Black–Scholes), hyperbolic (wave in 1-D and 2-D, telegraph, convection, Klein–Gordon, sine–Gordon), dispersive (KdV, the Euler–Bernoulli beam, Schrödinger as a real system), steady (Poisson in 2-D and 4-D, Helmholtz, a convection–diffusion boundary layer, Bratu), ODEs (logistic, damped oscillator, pendulum, Lotka–Volterra, Van der Pol, Lorenz, SIR, a CSTR), systems (Gray–Scott, Brusselator, linear shallow water, Stokes) and a three-state Hamilton–Jacobi–Bellman equation.

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
- Boundary conditions apply to every field and every face alike; mixed or per-field conditions need a callable residual or a subclass.
- The network sees raw coordinates: scale your problem so that the domain and the fields are of order one (non-dimensionalization helpers are on the roadmap).
- Integral terms, delays and non-local operators are outside the expression language.
