# Equations

This page lists the ten equations that ship with `pinnrl` (the keys of `pinnrl-train --pde`): each equation, its symbols, its default problem and its exact solution. Read it to pick a test problem or to check what a key solves.

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

- **Subscripts**: $u_t = \partial u/\partial t$, $u_{xx} = \partial^2 u/\partial x^2$, and so on.
- **$\nabla^2$**: the Laplacian, the sum of the second derivatives over the spatial axes.
- **Exact solution**: a formula that satisfies the equation and its conditions. Training is scored against it.

Each shipped exact solution is verified by `tests/unit_tests/test_physics_consistency.py`. It must satisfy, up to float32 round-off:

- the PDE: relative residual below 1e-4;
- the initial condition: error below 1e-6;
- the boundary conditions.

Two are evaluated numerically: Burgers (the Cole–Hopf solution) and the pendulum (a high-accuracy ODE integration). The Allen–Cahn `x2_cos` benchmark has no closed form and is scored against a [numerical reference](benchmarks.md#reference-solutions).

## The equations one by one

The domains, parameters and conditions below are the defaults of `config.yaml`.

### Heat (`heat`)

$$
u_t = \alpha\, u_{xx}, \qquad u(x, t) = \sin(2\pi x)\, e^{-4\pi^2 \alpha t}
$$

- **$u$**: temperature. **$\alpha$**: thermal diffusivity, 0.01.
- **Domain**: $x \in [0, 2]$, $t \in [0, 10]$.
- **Conditions**: periodic in $x$; $u(x, 0) = \sin(2\pi x)$.
- **Exact solution** (`sin_exp_decay`): in general $A \prod_d \sin(2\pi k x_d / L_d)\, e^{-\alpha \sum_d (2\pi k / L_d)^2 t}$, with $k$ = `frequency` periods over the axis length $L_d$.

### Heat in 2-D (`heat_2d`)

$$
u_t = \alpha\,(u_{xx} + u_{yy}), \qquad u(x, y, t) = \sin(2\pi x)\sin(2\pi y)\, e^{-8\pi^2 \alpha t}
$$

- **$\alpha$**: 0.01.
- **Domain**: $(x, y) \in [0, 1]^2$, $t \in [0, 5]$.
- **Conditions**: $u = 0$ on the four faces; $u(x, y, 0) = \sin(2\pi x)\sin(2\pi y)$.
- **Exact solution** (`sine_2d`): in general $A \sin(k_x \pi x)\sin(k_y \pi y)\, e^{-\alpha \pi^2 (k_x^2 + k_y^2) t}$.

### Wave (`wave`)

$$
u_{tt} = c^2 u_{xx}, \qquad u(x, t) = \sin(\pi x)\cos(c \pi t)
$$

- **$u$**: displacement. **$c$**: wave speed, 1.
- **Domain**: $x \in [0, 1]$, $t \in [0, 1]$.
- **Conditions**: $u = 0$ at both ends; $u(x, 0) = \sin(\pi x)$, released from rest ($u_t(x, 0) = 0$).
- **Exact solution** (`sine`): a standing wave; in $d$ dimensions $A \prod_d \sin(k\pi x_d)\cos(\omega t)$ with $\omega = c k \pi \sqrt{d}$.

### Burgers (`burgers`)

$$
u_t + u\,u_x = \nu\,u_{xx}, \qquad u(x, t) = \frac{\int u_0(y)\,\varphi_0(y)\, e^{-z^2}\,dz}{\int \varphi_0(y)\, e^{-z^2}\,dz}, \quad y = x - \sqrt{4\nu t}\,z
$$

- **$u$**: velocity. **$\nu$**: viscosity, 0.01.
- **Shock**: a near-discontinuous front; it forms at $x = 0$ for small $\nu$.
- **Domain**: $x \in [-1, 1]$, $t \in [0, 1]$.
- **Conditions**: $u(\pm 1, t) = 0$; $u_0(x) = -\sin(\pi x)$.
- **Exact solution** (`cole_hopf`, 1-D): the Cole–Hopf quotient above with $\varphi_0(y) = \exp\big(-\cos(\pi y)/(2\pi\nu)\big)$, integrated by the trapezoid rule (`pinnrl.benchmarks.reference.burgers_cole_hopf`).

### Convection (`convection`)

$$
u_t + v\,u_x = 0, \qquad u(x, t) = \sin\big(\pi (x - v t)\big)
$$

- **$u$**: a transported quantity. **$v$**: flow velocity (`velocity`), 1.
- **Characteristic**: a line $x - vt = \text{const}$ along which $u$ is carried unchanged.
- **Domain**: $x \in [0, 2]$, $t \in [0, 1]$.
- **Conditions**: periodic in $x$; $u(x, 0) = \sin(\pi x)$.
- **Exact solution** (`sin`): the initial profile moved with the flow, $u_0(x - vt)$.

### Korteweg–de Vries (`kdv`)

$$
u_t + \alpha\,u\,u_x + \beta\,u_{xxx} = 0, \qquad u(x, t) = \frac{3c}{\alpha}\operatorname{sech}^2\!\Big(\tfrac12\sqrt{c/\beta}\,(x - x_0 - c t)\Big)
$$

- **$u$**: wave height. **$\alpha$**: nonlinearity, 6. **$\beta$**: dispersion, 1.
- **Soliton**: a single pulse that keeps its shape and moves at speed $c$ (`speed`, 1) from $x_0$ (`position`, 0).
- **Domain**: $x \in [-15, 15]$, $t \in [0, 5]$. 1-D only.
- **Conditions**: $u = 0$ at both ends; the soliton at $t = 0$.
- **Exact solution** (`soliton`): the formula above.

### Allen–Cahn (`allen_cahn`)

$$
u_t = \varepsilon^2 u_{xx} + r\,(u - u^3), \qquad u(x, t) = \tanh\!\Big(\frac{\sqrt{r}\,x}{\sqrt{2}\,\varepsilon}\Big)
$$

- **$u$**: phase, $\pm 1$ in the two phases. **$\varepsilon$**: interface width (`epsilon`), 0.1. **$r$**: reaction rate (`reaction_rate`), 1.
- **Kink**: a stationary front between the two phases.
- **Domain**: $x \in [-1, 1]$, $t \in [0, 1]$.
- **Conditions**: boundary values taken from the exact solution; the kink at $t = 0$.
- **Exact solution** (`tanh`): the kink above. It does not move.
- **No exact solution**: the `x2_cos` initial condition, $u_0 = x^2 \cos(\pi x)$ (see [below](#shipped-defaults)).

### Cahn–Hilliard (`cahn_hilliard`)

$$
u_t = M\,\partial_{xx}\big(u^3 - u - \kappa\,u_{xx}\big), \qquad u(x, t) = \tanh\!\Big(\frac{x}{\sqrt{2\kappa}}\Big)
$$

- **$u$**: concentration. **$M$**: mobility (`mobility`), 1. **$\kappa$**: gradient energy (`kappa`), 0.01.
- **Domain**: $x \in [-1, 1]$, $t \in [0, 1]$.
- **Conditions**: boundary values from the exact solution and $u_x = 0$ on each face (fourth order needs two); the kink at $t = 0$.
- **Exact solution** (`tanh`): the stationary kink above.
- **No exact solution**: the `random` initial condition for spinodal-decomposition runs.

### Black–Scholes (`black_scholes`)

$$
V_\tau = \tfrac12 \sigma^2 S^2 V_{SS} + r S V_S - r V, \qquad V(S, \tau) = S\,\Phi(d_1) - K e^{-r\tau}\,\Phi(d_2)
$$

$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\,\tau}{\sigma\sqrt{\tau}}, \qquad d_2 = d_1 - \sigma\sqrt{\tau}
$$

- **$V$**: option price. **$S$**: asset price. **$\tau$**: time to maturity (the `t` input).
- **$\sigma$**: volatility, 0.2. **$r$**: risk-free rate, 0.05. **$K$**: strike, 100. **$\Phi$**: the standard normal distribution function.
- **Domain**: $S \in [0, 200]$, $\tau \in [0, 1]$.
- **Conditions**: boundary values from the exact solution; the call payoff $V(S, 0) = \max(S - K, 0)$.
- **Exact solution** (`black_scholes`, 1-D): the European call price above. For `option_type: put`, $V = K e^{-r\tau}\,\Phi(-d_2) - S\,\Phi(-d_1)$.

### Pendulum (`pendulum`)

$$
\theta'' + b\,\theta' + \frac{g}{L}\sin\theta = 0
$$

- **$\theta$**: angle. **$b$**: damping (`damping`), 0.1. **$g$**: gravity, 9.81. **$L$**: length, 1.
- **Domain**: $t \in [0, 10]$, with a dummy $x$ the solution does not depend on.
- **Conditions**: $\theta(0) = 0.5$ rad, $\theta'(0) = 0$; periodic in the dummy $x$.
- **Exact solution** (`ode`): none in closed form. The trajectory is integrated with `scipy` to a relative tolerance of 1e-11.

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

- **BC, IC**: boundary and initial conditions. **Dirichlet**: a fixed value of $u$. **Neumann**: a fixed value of $u_x$. **exact**: values taken from the exact solution.
- The Allen–Cahn and Cahn–Hilliard defaults start from their stationary kinks. That makes them consistency checks, not hard dynamics problems.

The dynamic Allen–Cahn benchmark (as in Wu et al., 2023) is

$$
u_t = 0.001\,u_{xx} + 5\,(u - u^3), \qquad u_0 = x^2\cos(\pi x), \qquad u(\pm 1, t) = -1, \qquad t \in [0, 1]
$$

- `pinnrl-benchmark sampling` runs it in rescaled time $\tau = 5t$, that is $\varepsilon^2 = 0.0002$, $r = 1$ and $\tau \in [0, 5]$.

To set it up yourself:

1. Set `initial_condition: {type: x2_cos}`.
2. Set `boundary_conditions: {dirichlet: {value: -1}}`.
3. Set `exact_solution: {}` (the default exact BCs come from the kink).
4. Score against `pinnrl.benchmarks.reference.allen_cahn_reference`.

## Adding a PDE

Write it: `CustomPDE("u_t + u*u_x - nu*u_xx", domain={"x": (-1, 1), "t": (0, 1)}, parameters={"nu": 0.01})` builds a trainable problem from the residual.

- It covers scalar PDEs, ODEs, steady problems and systems, in Python or as a YAML file for `pinnrl-train --pde-file`. See [Custom equations](custom_equations.md).
- The [catalogue](custom_equations.md#the-catalogue) holds 43 ready problems. An entry that shares a key with this page, such as `heat`, has its own domain and parameters.

To ship one with the package:

1. Subclass `PDEBase` and implement `compute_residual` (and `_exact_solution` if one exists).
2. Add one `PDESpec` line to `pinnrl/pdes/registry.py` and a block to `config.yaml`. The new PDE then appears in the CLI and the dashboard.
3. Add its exact-solution check to `test_physics_consistency.py`.
