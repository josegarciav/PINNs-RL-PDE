# Roadmap

This document tracks what has been built, what is in progress, and what is planned for `pinnrl`. Items within each milestone are ordered by combined score of adoption difficulty, scientific impact, and innovation level.

---

## v0.1 — Stabilization

**Status: Complete**

Core solver with 9 PDEs, 7 architectures, RL adaptive sampling, and a Dash dashboard for experiment management.

| Item | Status | Notes |
|---|---|---|
| 9 PDEs trainable end-to-end | Done | Heat, Wave, Burgers, Convection, KdV, Allen-Cahn, Cahn-Hilliard, Black-Scholes, Pendulum |
| 7 neural architectures | Done | FeedForward, ResNet, SIREN, Fourier Features, FNO, Attention, AutoEncoder |
| DQN-based RL collocation agent | Done | Epsilon-greedy exploration, experience replay |
| Adaptive loss weighting (LRW + RBW) | Done | Residual-balancing and learning-rate-based strategies |
| Cosine LR annealing + early stopping | Done | Configurable via `config.yaml` |
| Dash training dashboard | Done | 3 tabs: Live Training (monitor + new training), Comparison, Collocation & Solution |
| Experiment directory with config snapshots | Done | Timestamped dirs, metadata.json, .running marker |
| Apple Silicon (MPS) + CUDA + CPU | Done | Auto-detected or `--device` override |
| Exact analytical solutions for all 9 PDEs | Done | L2 and max-error validation built in |
| Headless CLI training script | Done | `src/training/train.py` with full hyperparameter overrides |

**Known limitations:**
- RL agent requires per-PDE tuning; not plug-and-play
- 2D PDEs partially configured but not fully validated
- No formal benchmark against FEM/FDM solvers

---

## v0.2 — Documentation and Onboarding

**Status: Mostly complete**

| Item | Status | Notes |
|---|---|---|
| MkDocs site with Material theme | Done | `mkdocs.yml` + `docs/` directory, auto-built via GitHub Actions |
| Start Here guide (5-minute onboarding) | Done | `docs/start_here.md` with dashboard screenshots |
| Installation & Setup guide | Done | `docs/setup.md` — uv, pip, GPU configuration |
| Architecture overview | Done | `docs/ARCHITECTURE.md` — system diagram, PINN training loop |
| Sampling strategies guide | Done | `docs/sampling_strategies.md` |
| Dashboard guide | Done | `docs/dashboard.md` — all tabs, screenshots, workflow |
| CONTRIBUTING.md and issue templates | Done | Bug report + new PDE templates |
| Versioned changelog (CHANGELOG.md) | Done | Full bug-fix record from v0.1 |
| GitHub Actions CI: lint + test on push | Done | `.github/workflows/checks.yml` |
| GitHub Actions: docs deployment | Done | `.github/workflows/docs.yml` |
| Jupyter tutorial: heat equation | Done | `notebooks/01_your_first_pinn.ipynb` |
| Jupyter tutorial: architecture comparison | Done | `notebooks/02_comparing_architectures.ipynb` |
| `mkdocstrings` API reference from docstrings | Not started | Needs docstring coverage pass across src/ |
| Jupyter tutorial: RL vs uniform sampling | Not started | Requires standardized benchmark harness |
| PyPI release (`pip install pinnrl`) | Not started | Blocked on final API stabilization |
| Code coverage badge (>80% target) | Not started | CI runs tests but no coverage reporting yet |

**Remaining priority:** PyPI release — prerequisite for researchers using `pinnrl` as a dependency.

---

## v0.3 — Science Features

**Target:** Q3 2026

Extends scientific capability with methods actively used in the PINN research community.

| Item | Status | Notes |
|---|---|---|
| FNO architecture (spectral convolutions) | Done | `src/neural_networks/fno.py`, tested against all 9 PDEs |
| RAR (Residual-Adaptive Refinement) sampling | Not started | |
| L-BFGS second-order optimizer | Not started | Adam warm-up → L-BFGS refinement pattern |
| 2D PDEs: full heat_2d training | Not started | Config exists, residual computation needs fixing |
| Inverse problems: parameter identification | Not started | Highest scientific value — see details below |
| Formal FDM comparison on heat/wave | Not started | |
| Stratified + RAR sampling benchmarks vs RL | Not started | |
| `pinnrl-benchmark` CLI subcommand | Not started | |
| Configurable loss functions (MSE, MAE, Huber) | Not started | |

**On inverse problems:** Recovering unknown PDE parameters (e.g., identifying `alpha` in the heat equation from noisy sensor data) is one of the highest-value capabilities PINNs offer over classical solvers. Implementation path: add a `trainable_parameters` field to `PDEConfig`, modify `PDEBase.compute_residual` to use `nn.Parameter` tensors, add a data-fitting loss term, and expose via `--mode inverse` CLI flag.

**On L-BFGS:** A quasi-Newton method that often outperforms Adam in the final convergence phase for smooth PDEs. PyTorch provides `torch.optim.LBFGS` with a closure-based API that requires modest changes to the training loop.

---

## v0.4 — Full RL Integration

**Target:** Q4 2026

Makes the RL adaptive sampler a first-class, benchmarked feature rather than an experimental add-on.

| Item | Status | Notes |
|---|---|---|
| Formal RL benchmark: RL vs uniform vs RAR vs stratified | Not started | Publishable contribution — see details below |
| Policy gradient (PPO) alternative to DQN | Not started | |
| Curriculum learning: coarse-to-fine collocation | Not started | |
| RL agent pre-training on synthetic PDE families | Not started | |
| Per-PDE RL configuration presets (tuned defaults) | Not started | |
| Dashboard: live RL reward + sampling density | Not started | |
| Reproducible benchmark artifacts (seeds, configs, logs) | Not started | |

**On the benchmark paper:** A rigorous comparison of RL-based adaptive sampling against RAR and uniform strategies across all nine PDEs, with fixed compute budgets, would be a publishable contribution. Key metrics: L2 error at convergence, wall-clock time to target loss, and collocation efficiency (accuracy per point). This is `pinnrl`'s strongest scientific differentiator.

---

## v1.0 — Production

**Target:** H1 2027

Stable API, production packaging, community infrastructure, and validation against established solvers.

| Item | Status | Notes |
|---|---|---|
| Benchmark vs FEniCS (FEM) on heat, wave, Burgers | Not started | |
| Benchmark vs DeepXDE on matching problem set | Not started | |
| Plugin system for custom PDEs and architectures | Not started | |
| Stable public API with semantic versioning | Not started | |
| Conda-forge package | Not started | |
| Docker image with GPU support | Not started | |
| Citation metadata (CITATION.cff, Zenodo DOI) | Not started | |
| Community forum or GitHub Discussions | Not started | |

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
