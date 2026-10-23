# Equations

These are the ten shipped configurations (keys for `pinnrl-train --pde`). Each shipped exact solution is verified by `tests/unit_tests/test_physics_consistency.py`: it must satisfy the PDE, the initial condition and the boundary conditions up to float32 round-off (relative residual below 1e-4, initial-condition error below 1e-6). Burgers uses the Cole–Hopf solution, evaluated numerically, and the pendulum a high-accuracy ODE integration. The Allen–Cahn `x2_cos` benchmark has no closed form and is scored against a [numerical reference](benchmarks.md#reference-solutions).

| Key | PDE | Where it arises | What makes it hard for a PINN |
|---|---|---|---|
| `heat`, `heat_2d` | $u_t = \alpha \nabla^2 u$ | Heat conduction, diffusion of species, the pricing kernel behind Black–Scholes | Little; the baseline. High-frequency initial data decays fast and is easy to under-resolve. |
| `wave` | $u_{tt} = c^2 \nabla^2 u$ | Acoustics, vibrating strings and membranes, electromagnetics | Second order in time, so it needs the initial velocity. Energy is conserved, so errors don't damp out. |
| `burgers` | $u_t + u\,u_x = \nu\,u_{xx}$ | A 1-D model of fluid flow and shock formation; traffic flow | For small $\nu$ a near-discontinuous shock forms. The canonical test for adaptive sampling. |
| `convection` | $u_t + \mathbf{v}\cdot\nabla u = 0$ | Transport of heat, mass or pollutants by a flow | It looks trivial, but PINNs fail for large $v$ ([Krishnapriyan et al., 2021](concepts.md#references)). Information only travels along characteristics. |
| `kdv` | $u_t + \alpha\,u\,u_x + \beta\,u_{xxx} = 0$ | Shallow-water waves, plasma waves; solitons | A third derivative, and a localized pulse that must move at exactly the right speed. |
| `allen_cahn` | $u_t = \varepsilon^2 \nabla^2 u + r\,(u - u^3)$ | Phase separation in alloys, image segmentation | Thin interfaces of width $\sim\varepsilon$, with fast dynamics and metastable states. |
| `cahn_hilliard` | $u_t = M\,\nabla^2\big(u^3 - u - \kappa\,\nabla^2 u\big)$ | Spinodal decomposition, polymer blends, battery electrodes | Fourth order, mass-conserving and stiff, with fine-scale structure. |
| `black_scholes` | $V_\tau = \tfrac12 \sigma^2 S^2 V_{SS} + r S V_S - r V$, with $\tau$ the time to maturity | European option pricing | A kink in the payoff at the strike; spatial and temporal scales that differ by orders of magnitude. |
| `pendulum` | $\theta'' + b\,\theta' + (g/L)\sin\theta = 0$ | A nonlinear oscillator: mechanics, control, robotics | An ODE posed as a PINN. Long time horizons accumulate phase error. |

## Shipped defaults

The defaults in `config.yaml` are each PDE's domain, parameters and recommended architecture. Override them with flags or your own config.

| Key | Domain × time | Parameters | BC | IC | Architecture |
|---|---|---|---|---|---|
| `heat` | [0, 2] × [0, 10] | α = 0.01 | periodic | $\sin(2\pi x)$ (`frequency: 2` periods on [0, 2]) | fourier |
| `heat_2d` | [0, 1]² × [0, 5] | α = 0.01 | Dirichlet 0 | $\sin(2\pi x)\sin(2\pi y)$ | fourier |
| `wave` | [0, 1] × [0, 1] | c = 1 | Dirichlet 0 | $\sin(\pi x)$, at rest | siren |
| `burgers` | [−1, 1] × [0, 1] | ν = 0.01 | Dirichlet 0 | $-\sin(\pi x)$ | resnet |
| `convection` | [0, 2] × [0, 1] | v = 1 | periodic | $\sin(\pi x)$ | fourier |
| `kdv` | [−15, 15] × [0, 5] | α = 6, β = 1 | Dirichlet 0 | soliton, c = 1 | siren |
| `allen_cahn` | [−1, 1] × [0, 1] | ε = 0.1, r = 1 | exact | kink $\tanh\big(\sqrt{r}\,x/(\sqrt2\,\varepsilon)\big)$ | fourier |
| `cahn_hilliard` | [−1, 1] × [0, 1] | M = 1, κ = 0.01 | exact + Neumann 0 | kink $\tanh\big(x/\sqrt{2\kappa}\big)$ | resnet |
| `black_scholes` | S ∈ [0, 200] × τ ∈ [0, 1] | σ = 0.2, r = 0.05 | exact | call payoff $\max(S-K, 0)$, K = 100 | feedforward |
| `pendulum` | t ∈ [0, 10] | g = 9.81, L = 1, b = 0.1 | periodic in a dummy $x$ (θ depends on t only) | θ0 = 0.5 rad, at rest | resnet |

The Allen–Cahn and Cahn–Hilliard defaults start from their stationary kinks, which makes them consistency checks rather than hard dynamics problems. The dynamic Allen–Cahn benchmark (as in Wu et al., 2023) is $u_t = 0.001\,u_{xx} + 5(u - u^3)$ on $t \in [0, 1]$, with $u_0 = x^2\cos(\pi x)$ and $u(\pm 1, t) = -1$. `pinnrl-benchmark sampling` runs it in rescaled time $\tau = 5t$, that is $\varepsilon^2 = 0.0002$, $r = 1$ and $\tau \in [0, 5]$. To set it up yourself, use `initial_condition: {type: x2_cos}`, `boundary_conditions: {dirichlet: {value: -1}}` and `exact_solution: {}` (the default exact BCs come from the kink), and score against `pinnrl.benchmarks.reference.allen_cahn_reference`.

## Adding a PDE

Subclass `PDEBase`, implement `compute_residual` (and `_exact_solution` if one exists), then add one `PDESpec` line to `pinnrl/pdes/registry.py` and a block to `config.yaml`. The new PDE then appears in the CLI and the dashboard. Add its exact-solution check to `test_physics_consistency.py`.
