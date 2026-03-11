# Roadmap

This document tracks what has been built, what is in progress, and what is planned for `pinnrl`. Items within each milestone are scored on three dimensions (1–5 scale):

- **Adoption difficulty** — how hard it is to implement and ship
- **Scientific impact** — value to the PINN research community
- **Innovation level** — novelty relative to existing tools

---

## v0.1 — Current (Stabilization)

**Status: Complete**

Core solver with 9 PDEs, 7 architectures, RL adaptive sampling, and a Dash dashboard for experiment management.

| Item | Status | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|---|
| 9 PDEs × 7 architectures trainable end-to-end | Done | 2 | 3 | 2 |
| DQN-based RL collocation agent (epsilon-greedy) | Done | 3 | 4 | 3 |
| Adaptive loss weighting (LRW and RBW strategies) | Done | 2 | 3 | 2 |
| Cosine LR annealing + early stopping | Done | 1 | 2 | 1 |
| Plotly/Matplotlib solution visualization | Done | 1 | 1 | 1 |
| Dash real-time training dashboard | Done | 2 | 2 | 2 |
| Experiment directory with config snapshots | Done | 1 | 1 | 1 |
| Apple Silicon (MPS) + CUDA + CPU support | Done | 2 | 1 | 1 |
| Exact analytical solutions for all 9 PDEs | Done | 1 | 2 | 1 |
| Headless CLI training script | Done | 1 | 1 | 1 |

**Known limitations:**

- RL agent requires per-PDE tuning; not plug-and-play
- 2D PDEs partially configured but not fully validated
- No formal benchmark against FEM/FDM solvers

---

## v0.2 — Documentation and Onboarding

**Status: Mostly complete**

| Item | Status | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|---|
| MkDocs site with Material theme | Done | 1 | 1 | 1 |
| Start Here guide (5-minute onboarding) | Done | 1 | 1 | 1 |
| Installation & Setup guide | Done | 1 | 1 | 1 |
| Architecture overview | Done | 1 | 2 | 1 |
| Sampling strategies guide | Done | 1 | 2 | 1 |
| Dashboard guide | Done | 1 | 1 | 1 |
| CONTRIBUTING.md and issue templates | Done | 1 | 1 | 1 |
| Versioned changelog (CHANGELOG.md) | Done | 1 | 1 | 1 |
| GitHub Actions CI: lint + test on push | Done | 1 | 1 | 1 |
| GitHub Actions: docs deployment | Done | 1 | 1 | 1 |
| Jupyter tutorial: heat equation | Done | 1 | 2 | 1 |
| Jupyter tutorial: architecture comparison | Done | 1 | 2 | 1 |
| `mkdocstrings` API reference from docstrings | Done | 2 | 1 | 1 |
| Jupyter tutorial: RL vs uniform sampling | Done | 2 | 3 | 2 |
| PyPI release (`pip install pinnrl`) | To do | 2 | 2 | 1 |
| Linkedin post | To do | 1 | 2 | 1 |

**Remaining priority:** PyPI release — prerequisite for researchers using `pinnrl` as a dependency.

---

## v0.3 — Science Features

**Target:** Q3 2026

Extends scientific capability with methods actively used in the PINN research community.

| Item | Status | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|---|
| FNO architecture (spectral convolutions) | Done | 3 | 4 | 3 |
| Inverse problems: parameter identification | To do | 3 | 5 | 4 |
| L-BFGS second-order optimizer | To do | 2 | 3 | 2 |
| 2D PDEs: full heat_2d training | To do | 3 | 4 | 2 |
| RAR (Residual-Adaptive Refinement) sampling | To do | 2 | 3 | 2 |
| Formal FDM comparison on heat/wave | To do | 2 | 3 | 2 |
| Stratified + RAR sampling benchmarks vs RL | To do | 2 | 3 | 3 |
| `pinnrl-benchmark` CLI subcommand | To do | 2 | 2 | 2 |
| Configurable loss functions (MSE, MAE, Huber) | To do | 1 | 2 | 1 |
| Linkedin post | To do | 1 | 2 | 1 |

**On inverse problems:** Recovering unknown PDE parameters (e.g., identifying `alpha` in the heat equation from noisy sensor data) is one of the highest-value capabilities PINNs offer over classical solvers. Implementation path: add a `trainable_parameters` field to `PDEConfig`, modify `PDEBase.compute_residual` to use `nn.Parameter` tensors, add a data-fitting loss term, and expose via `--mode inverse` CLI flag.

**On L-BFGS:** A quasi-Newton method that often outperforms Adam in the final convergence phase for smooth PDEs. PyTorch provides `torch.optim.LBFGS` with a closure-based API that requires modest changes to the training loop.

---

## v0.4 — Full RL Integration

**Target:** Q4 2026

Makes the RL adaptive sampler a first-class, benchmarked feature rather than an experimental add-on.

| Item | Status | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|---|
| Formal RL benchmark: RL vs uniform vs RAR vs stratified | To do | 3 | 5 | 4 |
| Policy gradient (PPO) alternative to DQN | To do | 3 | 3 | 3 |
| Curriculum learning: coarse-to-fine collocation | To do | 3 | 4 | 4 |
| RL agent pre-training on synthetic PDE families | To do | 4 | 4 | 5 |
| Per-PDE RL configuration presets (tuned defaults) | To do | 2 | 2 | 2 |
| Dashboard: live RL reward + sampling density | To do | 2 | 2 | 3 |
| Reproducible benchmark artifacts (seeds, configs, logs) | To do | 1 | 3 | 1 |
| Linkedin post | To do | 1 | 2 | 1 |

**On the benchmark paper:** A rigorous comparison of RL-based adaptive sampling against RAR and uniform strategies across all nine PDEs, with fixed compute budgets, would be a publishable contribution. Key metrics: L2 error at convergence, wall-clock time to target loss, and collocation efficiency (accuracy per point). This is `pinnrl`'s strongest scientific differentiator.

---

## v1.0 — Production

**Target:** H1 2027

Stable API, production packaging, community infrastructure, and validation against established solvers.

| Item | Status | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|---|
| Benchmark vs FEniCS (FEM) on heat, wave, Burgers | To do | 3 | 4 | 3 |
| Benchmark vs DeepXDE on matching problem set | To do | 3 | 4 | 2 |
| Plugin system for custom PDEs and architectures | To do | 4 | 3 | 3 |
| Stable public API with semantic versioning | To do | 2 | 2 | 1 |
| Conda-forge package | To do | 2 | 1 | 1 |
| Docker image with GPU support | To do | 2 | 1 | 1 |
| Citation metadata (CITATION.cff, Zenodo DOI) | To do | 1 | 2 | 1 |
| Community forum or GitHub Discussions | To do | 1 | 1 | 1 |
| Linkedin post | To do | 1 | 2 | 1 |

---

## Priority paths

Ranked by expected return — combining scientific demand, differentiation from existing tools, and implementation feasibility.

### 1. Inverse problems (parameter identification)

**Scientific demand:** Extremely high. Recovering physical parameters from noisy observations is a central challenge in materials science, geophysics, biomechanics, and climate modeling. Classical methods (adjoint optimization, ensemble Kalman filter) are computationally expensive and mesh-dependent. PINNs handle irregular domains and sparse data naturally.

**Differentiation:** Most PINN libraries support forward problems only. A clean inverse-problem API (observable data in, unknown parameters out) would immediately attract experimental scientists.

**Implementation path:** Add `trainable_parameters` to `PDEConfig`. Modify `PDEBase.compute_residual` to use `nn.Parameter` tensors. Add data-fitting loss `L_data = mean(|u_θ(x_obs) - u_obs|²)`. Expose via `--mode inverse` CLI flag.

### 2. 2D and 3D PDEs with domain decomposition

**Scientific demand:** High. Real engineering problems are rarely 1D. Domain decomposition (XPINNs / CPINNs) partitions the domain into subdomains handled by separate networks, enabling parallelism.

**Implementation path:** The `input_dim=3` case (x, y, t) is already supported by all 7 architectures. Primary work: (a) fix 2D residual computation in `PDEBase`, (b) implement 2D visualization, (c) domain decomposition via interface continuity conditions.

### 3. RL benchmark paper

**Scientific demand:** Moderate but strategic. `pinnrl` is the only open-source PINN library with an integrated RL agent. Formalizing this with reproducible benchmarks converts an experimental feature into a validated contribution.

**Implementation path:** Primarily experimental design, not software. Standardize seeds, training budgets, metrics. Run all 9 PDEs under 4 sampling strategies. Aggregate and write up.

### 4. Dashboard and visualization improvements

**Status: Partially addressed.** The dashboard now supports experiment launching, monitoring, comparison, and solution visualization. Remaining opportunities: live residual heatmaps, RL reward visualization, and static HTML export for sharing.

### 5. High-dimensional PDEs

**Scientific demand:** Growing. PINNs avoid the curse of dimensionality that limits FEM. PDEs in 10–100 dimensions arise in multi-asset options pricing, quantum many-body problems, and stochastic control. All 7 architectures already support arbitrary `input_dim`. The work is in defining sampling strategies, boundary conditions, and reference solutions for high-dimensional test cases.

---

## Exploratory directions

Long-horizon, high-risk, high-reward ideas. Not on the near-term roadmap, but they represent genuine open problems where `pinnrl`'s architecture could provide a foundation.

### FNO operator learning

**Status: Foundation built.** The `fno` architecture implements spectral convolution layers adapted for point-wise PINN inputs. The next step is extending to true operator learning — mapping initial/boundary conditions to full solution fields, enabling zero-shot generalization to new IC/BC pairs without retraining.

### Multi-agent RL for domain decomposition

Deploy a population of RL agents — one per subdomain — that communicate through shared boundary information. Each agent optimizes local residual reduction while a coordinator manages interface conditions. Maps naturally to the XPINNs framework and could yield massive parallelism on distributed hardware.

### PDE discovery from data (SINDy + PINNs)

Learn the governing PDE from data rather than specifying it. Combine SINDy's sparse candidate library with PINN automatic differentiation to jointly identify the equation structure and its solution. Targets fields where the governing PDE is unknown or partially known (fluid mechanics, systems biology).

### Differentiable physics for robotics and control

Export trained PINN solutions as differentiable physics modules embeddable in RL loops for robotics. A PINN trained on the pendulum equation is a differentiable, zero-shot simulator that generalizes across initial conditions.

### Turbulence modeling with PINNs

Extend to Navier-Stokes (not yet in `pinnrl`) for turbulent flows, using a PINN as a closure model for Reynolds stresses trained on high-fidelity DNS data.

### Option pricing with Greeks via autograd

Extend the Black-Scholes implementation to multi-asset pricing (d >= 10) with real-time Greeks (Delta, Gamma, Vega, Theta) computed via the same autograd graph used for the PDE residual. Monte Carlo cannot provide these analytically; a PINN-based pricer returns them for free.
