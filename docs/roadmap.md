# Roadmap

Package features, in the order the [research program](research.md) needs them. Each item has a done-criterion and names the experiments (`experiments/BACKLOG.md`) or papers (`experiments/PAPERS.md`) it serves. Within a release, items are worked top to bottom.

## v0.4: clean, verified, documented (released 2026-10-24)

- [x] Physics audit: every exact solution satisfies its PDE, IC and BC; real Dirichlet, Neumann and periodic BCs; correct 2-D derivatives. *Done: `test_physics_consistency.py` is green.*
- [x] One training loop (`PDETrainer.train` / `train_steps`), one PDE registry, one entry point per action.
- [x] A working, experimental RL bandit sampler; RAD with `k`/`c`.
- [x] Numerical reference solutions. *Done: Burgers reproduces Basdevant et al. (1986), $u_x(0, 1.6037/\pi) = -152.00516$ for $\nu = 0.01/\pi$; the Allen–Cahn solution is grid-converged.*
- [x] B01/B06 sampling benchmark, 10 seeds at equal steps and equal evaluations. *Done: [Benchmarks](benchmarks.md), `experiments/B01_sampling_equal_compute/`.*
- [x] Docs rewritten and audited against the code; ruff with docstring linting; strict docs build.
- [x] Bug batch from the documentation audit (Well loader coordinates, LRW zero-gradient weighting, network defaults, dashboard plots). *Done: each fix has a test that fails without it (`test_audit_regressions.py`, `test_well_loader.py`).*
- [x] Release. *Done: CI and `scripts/verify_release.sh` are green, and v0.4.0 is on PyPI via trusted publishing.*

## v0.5: the inverse-modelling core (released 2026-10-26)

The goal is to estimate unknown physics with known validity conditions and calibrated uncertainty. Diagnostics come before UQ methods, because an unidentifiable or ill-conditioned problem has no meaningful error bar.

0. **User-defined equations.** `CustomPDE`: any PDE, ODE, steady problem or system from its residual as text (SymPy-parsed, autograd derivatives) or a Python callable, with expression-valued initial and exact solutions, identifiable parameters, and `pinnrl-train --pde-file`. Every later module (diagnostics, UQ, the process models of v0.7) is written against it rather than against one class per equation. *Done: `test_custom_pde.py` checks parity with the shipped Heat, Burgers and pendulum residuals; see [Custom equations](custom_equations.md).*
1. ~~**Theory pages.**~~ Well-posedness and conditional stability; structural vs practical identifiability; the Bayesian view (loss = negative log-posterior, data weight = 1/σ²); sources of uncertainty; calibration; the numerical-analysis bridge (Lax equivalence, stability constants, PINN error estimates). *Done: [Theory](theory/index.md), four pages cited by DOI; the maintainer's reference papers are still to be added. Every later module cites the condition it relies on.*
2. ~~**Identification baseline.**~~ Scalar parameters vs noise, sensors, initial guess and data weight. *Done: F01–F05 in `experiments/` and [Benchmarks](benchmarks.md). The surrogate floor is 2 % (heat), 20 % (KdV soliton) and 150 % (Burgers shock); above it the estimator is near-efficient. The data-weight question is K39.*
3. ~~**`pinnrl.analysis`: conditioning diagnostics.**~~ The sensitivity Jacobian by the implicit-function theorem on the trained surrogate; Fisher information, Cramér–Rao bound, correlations and eigen-directions; the loss-Hessian (Lanczos) and tangent-kernel spectra. *Done: `pinnrl.analysis`; K01 (validated on heat and KdV, wrong on the unresolved Burgers shock), K03 and F07 are in `experiments/`. Profile likelihood is a driver (F07), not yet an API.*
4. ~~**Disturbance and stability analysis.**~~ Perturbation suites for data and for the problem; the linear prediction of the estimate's shift; empirical stability constants. *Done: `pinnrl.analysis.disturbance`; K05 (bias and delay shifts match $F^{-1}J^\top\Sigma^{-1}\delta y$ to 20 %), K06 (the surrogate respects the equation's stability above its own field error) and K07 ($C \approx 1$ for linear evolution equations; the beam's error is set by its boundary terms) are in `experiments/`.*
5. ~~**`pinnrl.uq`: calibrated uncertainty.**~~ A likelihood-based data term (σ sets the weight and the misfit tolerance); deep ensembles with a dispersion diagnostic; a Laplace approximation that reuses the Hessian machinery of `pinnrl.analysis`; the **region of equivalence** as the primary output, sampled by a cloud particle swarm over the parameters on a parametric surrogate $u_w(x, t; \theta)$, with intervals, percentiles and CDFs by counting ([theory](theory/equivalence_regions.md)); coverage, width and CRPS metrics. *Done: `pinnrl.uq`; F06/K10 (the Laplace width is exact for the random error, the 2 % bias costs coverage, ensembles over seeds cover 13 %, the physics region 90 %), K39 (the noise-model weight is worse than the tuned one because the surrogate's error is systematic) and K46 (the swarm confirms the ellipse at these noise levels and under-samples its edges) are in `experiments/`. K47 (the weighted swarm is centred but 0.5–0.7 as wide as the posterior) and K45 (one parametric surrogate covers the prior range to 0.2–2 % and the swarm on its misfit reproduces the closed-form region) complete the item.*
6. ~~**Sampling follow-ups for P02.**~~ B02 (more PDEs), B03 (the RAD k, c grid), B04 (resampling frequency), B08 (point-count scaling). *Done: no sampler beats uniform significantly on smooth problems; the resampling cadence changes the Burgers error threefold and the study default moves to 10 steps; good RAD settings lie on c ≈ k/2 and a zero floor with a sharp exponent fails; the error scales as N^−1/3 on Burgers and not at all on Allen–Cahn. P02 can be drafted.*

## v0.6: richer unknowns and inputs (released 2026-11-04)

- [x] **Field-valued unknowns:** a learnable coefficient field $\alpha(x, t)$ or closure $f(u)$, with positivity and monotonicity constraints. *Done: `CustomPDE(unknown_functions=...)` with `UnknownFunction`; F09 recovers $\alpha(t)$ to 4 % and a two-layer $\alpha(x)$ in shape on heat, and shows which fields the data cannot determine.*
- [x] **Coupled multi-field systems** (`output_dim > 1`) with per-field BCs and loss weights. *Done: `CustomPDE(boundary_conditions={field: ...}, field_weights=...)`; the catalogue's `reaction_diffusion_pair` trains to its closed form in `test_coupled_systems.py`.*
- [x] **Measured inputs:** BCs and forcing from interpolated time series. *Done: `CustomPDE(inputs={"q": {"path": "left.csv"}}, boundary_conditions={"dirichlet": {"left": "q"}})`; `test_measured_inputs.py` trains heat with its boundary following a CSV.*
- [x] **Sensor operators:** point, boundary and integral observations with per-sensor noise and delay. *Done: `CustomPDE(sensors=[PointSensor, BoundarySensor, IntegralSensor])`, `pde.observe`, `sensor_jacobian`; K02 ranks layouts by their Fisher information and trains on sensor records (F05's fixed probes are one `PointSensor` each; its results predate the module).*
- [x] **Non-dimensionalization helpers** (a reference scale per variable). *Done: `CustomPDE(scales=...)` + `build_trainer(normalize=True)` / `--normalize`; a heat problem in millimetres, seconds and kelvin trains like the unit problem (`test_normalization.py`). J01 will be the first process model to use it.*
- [x] **High-dimensional stability:** dimension sweeps of stability constants and quadrature error. *Done: K08, heat and convection in $d = 1 \ldots 5$; the quadrature gap is dimension-independent, the stability constant grows with the condition errors, and high-dimensional convection collapses to zero.*

## v0.7+: process twins (now)

- [x] `pinnrl.process`: the falling-film evaporator with its method-of-lines reference simulator and `custom_pde()` form (J01, J02 done); fouling estimation from outlet sensors (J03: the drift to 9–16 %, at the surrogate's limit) and its identifiability (J04: the data allow 5 % on the end value; U and the constant fouling are one unknown; the steam temperature is the nuisance that matters). The drying droplet (`pinnrl.process.DryingDroplet`, J07) is the second unit: its forward PINN solve reaches 8–14 %; the free-function closure collapses to a constant from a cold start and reaches 1.7–2.1 % from a pointwise estimate of the record (J07b), with a 1-in-3 risk of leaving the basin. *Open: calibrated fouling intervals (J06), a pointwise-start helper and a misfit guard for closures, the dryer-scale model. K09 (linearized stability) is done: the physics is convectively stabilised and not stiff; the discretised operator's spectrum is a pseudospectrum.*
- [x] Classical estimators on the same models: `pinnrl.process.ExtendedKalmanFilter` with an RTS smoother and `FallingFilmEvaporator.fouling_state_space` (J05: on a known model the filter beats the PINN twin by a wide margin at a thousandth of the cost; the PINN's case is the partly unknown model). *Open: moving-horizon estimation (P15, P19).*
- Historian adapter: CSV or Parquet plus a tag map to sensor definitions.
- A dashboard "Process" view: soft-sensor trends with uncertainty bands, and time to the next cleaning.
- Bayesian measurement design for sensors and lab sampling (J09; P24).
- Type checking (mypy) across the package.

## Not planned

A hosted service and monetization (until the research results justify it); a custom PDE-builder UI; operator learning beyond the existing FNO-style layers.
