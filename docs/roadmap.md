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

## v0.5: the inverse-modelling core (now)

The goal is to estimate unknown physics with known validity conditions and calibrated uncertainty. Diagnostics come before UQ methods, because an unidentifiable or ill-conditioned problem has no meaningful error bar.

0. **User-defined equations.** `CustomPDE`: any PDE, ODE, steady problem or system from its residual as text (SymPy-parsed, autograd derivatives) or a Python callable, with expression-valued initial and exact solutions, identifiable parameters, and `pinnrl-train --pde-file`. Every later module (diagnostics, UQ, the process models of v0.7) is written against it rather than against one class per equation. *Done: `test_custom_pde.py` checks parity with the shipped Heat, Burgers and pendulum residuals; see [Custom equations](custom_equations.md).*
1. **Theory pages.** Well-posedness and conditional stability; structural vs practical identifiability; the Bayesian view (loss = negative log-posterior, data weight = 1/σ²); sources of uncertainty; calibration; the numerical-analysis bridge (Lax equivalence, stability constants, PINN error estimates). They're built around the maintainer's reference papers, cited by DOI. *Done when: `docs/theory/` pages are published and every later module cites the condition it relies on.*
2. **Identification baseline.** Scalar parameters vs noise, sensors, initial guess and data weight. *Done when: F01–F05 are written up (paper P07).*
3. **`pinnrl.analysis`: conditioning diagnostics.** The sensitivity Jacobian $J = \partial y / \partial \theta$ by autograd; Fisher information $F = J^\top \Sigma^{-1} J$; singular values, condition number and sloppy directions; profile likelihood; the loss-Hessian / NTK spectrum. *Done when: K01, K03 and F07 run on heat, KdV and Burgers (P08, P09).*
4. **Disturbance and stability analysis.** Perturbation suites for data (noise, bias, dropout, delay) and for the problem (IC, BC, parameters); a comparison with the linear prediction $\delta\theta \approx F^{-1} J^\top \Sigma^{-1} \delta y$; empirical stability constants. *Done when: K05–K07 are written up (P10, P11).*
5. **`pinnrl.uq`: calibrated uncertainty.** A likelihood-based data term (σ sets the weight); deep ensembles; a Laplace approximation that reuses the Fisher matrix; coverage, width and CRPS metrics. *Done when: F06/K10 report calibrated intervals (P13).*
6. **Sampling follow-ups for P02.** B02 (more PDEs), B03 (the RAD k, c grid), B04 (resampling frequency), B08 (point-count scaling). *Done when: P02 can be drafted.*

## v0.6: richer unknowns and inputs

- **Field-valued unknowns:** a learnable coefficient field $\alpha(x, t)$ or closure $f(u)$, with positivity and monotonicity constraints. *Done when: $\alpha(x)$ is recovered on heat (F09, P14).*
- **Coupled multi-field systems** (`output_dim > 1`) with per-field BCs and loss weights. *Done when: a two-field reaction–diffusion test passes.*
- **Measured inputs:** BCs and forcing from interpolated time series. *Done when: the heat BCs can be driven by a CSV.*
- **Sensor operators:** point, boundary and integral observations with per-sensor noise and delay. *Done when: F05 and K02 use them.*
- **Non-dimensionalization helpers** (a reference scale per variable). *Done when: J01 trains without hand-scaling.*
- **High-dimensional stability:** dimension sweeps of stability constants and quadrature error. *Done when: K08 is written up (P12).*

## v0.7+: process twins

- `pinnrl.process`: evaporator and spray-dryer models with method-of-lines reference simulators and linearized-stability checks (J01, J07, K09; P17).
- Classical estimators on the same models: EKF/UKF and moving-horizon estimation (J05; P15, P19).
- Historian adapter: CSV or Parquet plus a tag map to sensor definitions.
- A dashboard "Process" view: soft-sensor trends with uncertainty bands, and time to the next cleaning.
- Bayesian measurement design for sensors and lab sampling (J09; P24).
- Type checking (mypy) across the package.

## Not planned

A hosted service and monetization (until the research results justify it); a custom PDE-builder UI; operator learning beyond the existing FNO-style layers.
