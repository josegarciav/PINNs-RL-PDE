# Roadmap

This document describes the planned development trajectory for `pinnrl`, from its current alpha state through a production-ready v1.0 release. Items are ordered within each milestone by combined score of market adoption difficulty, scientific impact, and innovation level.

---

## v0.1 — Current (Stabilization)

**Status:** Complete

All nine PDEs are trainable with all six architectures. The RL adaptive sampler is functional. The interactive trainer (`src/interactive_trainer.py`) allows selecting any PDE with any architecture.

| Item | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|
| 9 PDEs × 6 architectures trainable end-to-end | 2 | 3 | 2 |
| DQN-based RL collocation agent (epsilon-greedy) | 3 | 4 | 3 |
| Adaptive loss weighting (LRW and RBW strategies) | 2 | 3 | 2 |
| Cosine LR annealing + early stopping | 1 | 2 | 1 |
| Plotly/Matplotlib solution visualization | 1 | 1 | 1 |
| Dash real-time training dashboard | 2 | 2 | 2 |
| Experiment directory with config snapshots | 1 | 1 | 1 |
| Apple Silicon (MPS) + CUDA + CPU support | 2 | 1 | 1 |

**Known limitations in v0.1:**
- No PyPI release; installation requires cloning the repo
- Limited documentation (README only)
- No formal benchmark against FEM/FDM solvers
- RL agent requires tuning per-PDE; not plug-and-play

---

## v0.2 — Documentation and Onboarding

**Target:** Q2 2026

The primary goal of v0.2 is to lower the barrier to entry for scientists and PhD students who are new to PINNs or to this library.

| Item | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|
| MkDocs site with Material theme (auto-built from `docs/`) | 1 | 1 | 1 |
| `mkdocstrings` API reference from docstrings | 1 | 1 | 1 |
| CONTRIBUTING.md and issue templates | 1 | 1 | 1 |
| Jupyter tutorial: heat equation from scratch | 1 | 2 | 1 |
| Jupyter tutorial: RL vs uniform sampling comparison | 2 | 3 | 2 |
| Jupyter tutorial: architecture selection guide | 1 | 2 | 1 |
| Versioned changelog (CHANGELOG.md) | 1 | 1 | 1 |
| PyPI release (`pip install pinnrl`) | 2 | 2 | 1 |
| GitHub Actions CI: lint + test on push | 1 | 1 | 1 |
| Code coverage badge (>80% target) | 1 | 1 | 1 |

**Priority within v0.2:** Ship the PyPI release first — it is a prerequisite for researchers wanting to use `pinnrl` as a dependency in their own projects without cloning the repo.

---

## v0.3 — Science Features

**Target:** Q3 2026

Extends the scientific capability of the solver with methods that are actively used in the research community.

| Item | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|
| RAR (Residual-Adaptive Refinement) sampling | 2 | 4 | 2 |
| L-BFGS second-order optimizer integration | 2 | 4 | 2 |
| 2D PDEs: full `heat_2d` training (already in config) | 2 | 4 | 2 |
| Inverse problems: parameter identification from data | 3 | 5 | 3 |
| Formal FDM comparison on heat/wave equations | 2 | 3 | 1 |
| Stratified and RAR sampling benchmarks vs RL | 1 | 3 | 1 |
| `pinnrl-benchmark` CLI subcommand | 2 | 3 | 2 |
| Configurable loss functions (MSE, MAE, Huber) | 1 | 2 | 1 |

**On inverse problems:** The ability to recover unknown PDE parameters (e.g., identify `alpha` in the heat equation from noisy sensor data) is one of the highest-value capabilities PINNs offer over classical solvers. It requires adding a `parameter_estimation` mode where selected parameters are included in the optimizer's variable set. This is achievable with minimal changes to the existing `PDEBase` API.

**On L-BFGS:** L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method that often outperforms Adam on the final convergence phase for smooth PDEs. The standard approach is Adam warm-up for 1000–2000 epochs followed by L-BFGS refinement. PyTorch provides `torch.optim.LBFGS` with a closure-based API that requires modest changes to the training loop.

---

## v0.4 — Full RL Integration

**Target:** Q4 2026

Makes the RL adaptive sampler a first-class, well-benchmarked feature rather than an experimental add-on.

| Item | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|
| Formal RL benchmark: RL vs uniform vs RAR vs stratified | 3 | 4 | 3 |
| Policy gradient (PPO) alternative to DQN agent | 4 | 4 | 4 |
| Curriculum learning: coarse-to-fine collocation | 3 | 4 | 3 |
| RL agent pre-training on synthetic PDE families | 4 | 5 | 4 |
| Per-PDE RL configuration presets (tuned defaults) | 2 | 3 | 2 |
| Training dashboard: live RL reward + sampling density | 2 | 2 | 2 |
| Reproducible benchmark artifacts (seeds, configs, logs) | 1 | 3 | 1 |

**On the benchmark paper:** A rigorous comparison of RL-based adaptive sampling against RAR and uniform strategies across all nine PDEs, with fixed compute budgets, would be a publishable contribution. The key metrics are L2 error at convergence, wall-clock time to reach a target loss, and collocation efficiency (accuracy per point). This is `pinnrl`'s strongest scientific differentiator.

---

## v1.0 — Production

**Target:** H1 2027

Stable API, production packaging, community infrastructure, and validation against established solvers.

| Item | Adoption difficulty | Scientific impact | Innovation level |
|---|---|---|---|
| Benchmark vs FEniCS (FEM) on heat, wave, Burgers | 4 | 5 | 2 |
| Benchmark vs DeepXDE on matching problem set | 3 | 4 | 2 |
| Plugin system for custom PDEs and architectures | 4 | 3 | 3 |
| Stable public API with semantic versioning | 2 | 2 | 1 |
| Conda-forge package | 2 | 2 | 1 |
| Docker image with GPU support | 2 | 2 | 1 |
| Citation metadata (CITATION.cff, Zenodo DOI) | 1 | 2 | 1 |
| Community forum or GitHub Discussions | 1 | 2 | 1 |

---

## Priority paths

The following directions are ranked by their expected return — combining open scientific demand, differentiation from existing tools (DeepXDE, NeuralPDE.jl, SciANN), and implementation feasibility with the current codebase.

### 1. Inverse problems (parameter identification)

**Scientific demand:** Extremely high. Recovering physical parameters from noisy observations is a central challenge in materials science, geophysics, biomechanics, and climate modeling. Classical methods (adjoint optimization, ensemble Kalman filter) are computationally expensive and mesh-dependent. PINNs handle irregular domains and sparse data naturally.

**Differentiation:** Most PINN libraries support forward problems. Offering a clean inverse-problem API (observable data in, unknown parameters out) would immediately attract experimental scientists who cannot mesh their domains.

**Implementation path:** Add a `trainable_parameters` field to `PDEConfig`. Modify `PDEBase.compute_residual` to use these parameters as `nn.Parameter` tensors. Add a data-fitting loss term `L_data = mean(|u_θ(x_obs) - u_obs|²)`. Expose via `--mode inverse` CLI flag.

### 2. 2D and 3D PDEs with domain decomposition

**Scientific demand:** High. Real engineering problems are rarely 1D. The `heat_2d` configuration already exists in `config.yaml` but is not fully tested end-to-end. Domain decomposition (XPINNs / CPINNs) partitions the domain into subdomains, each handled by a separate network, enabling parallelism and better scaling.

**Differentiation:** 2D/3D support with domain decomposition is a meaningful step toward industrial applicability. FEM scales to 3D naturally; PINNs do not yet do so reliably. Demonstrating competitive accuracy on a 2D heat or elasticity problem would be a significant result.

**Implementation path:** The `input_dim=3` case (x, y, t) is already supported by all architectures. The primary work is (a) fixing the 2D residual computation in `PDEBase`, (b) implementing 2D visualization, and (c) domain decomposition via interface continuity conditions.

### 3. RL benchmark paper

**Scientific demand:** Moderate but strategic. The combination of RL and PINNs is a novel area with few rigorous empirical evaluations. A well-controlled benchmark study comparing DQN-based sampling against RAR, stratified, and uniform strategies across all nine PDEs would be a publishable NeurIPS/ICLR workshop paper or JMLST paper.

**Differentiation:** `pinnrl` is currently the only open-source PINN library with an integrated RL agent. This is the clearest scientific differentiator. Formalizing it with reproducible benchmarks converts an experimental feature into a validated contribution.

**Implementation path:** Primarily an experimental design and compute problem, not a software problem. Standardize seeds, training budgets, and metrics. Run all nine PDEs under all four sampling strategies. Aggregate results. Write up.

### 4. Real-time dashboard and GUI improvement

**Scientific demand:** Moderate. The existing Dash dashboard provides real-time loss curves and solution plots but has limited interactivity. Adding parameter sliders, architecture selection, and live residual heatmaps would make the library accessible to scientists without Python expertise.

**Differentiation:** Deep learning tools are notoriously opaque. A high-quality interactive dashboard that shows *why* a PINN is struggling (where the residual is large, how the loss weights evolve) would be genuinely useful and shareable via browser.

**Implementation path:** Extend `src/dashboard.py` with Dash callback-driven architecture/PDE selection, collocation point visualization, and residual error heatmaps. Consider a lightweight web export option (e.g., static Plotly HTML).

### 5. High-dimensional PDEs

**Scientific demand:** Growing. PINNs are theoretically exempt from the curse of dimensionality in a way that FEM is not, because they parameterize the solution function globally rather than on a mesh. PDEs in 10–100 dimensions arise in options pricing (multi-asset), quantum many-body problems, and stochastic control.

**Differentiation:** Supporting d-dimensional PDEs (with `input_dim=d+1` for time) and demonstrating accuracy on a d=10 test case would position `pinnrl` alongside research efforts on high-dimensional PDE solvers.

**Implementation path:** Architectures already support arbitrary `input_dim`. The work is in defining appropriate sampling strategies, boundary conditions, and reference solutions for high-dimensional test cases. Black-Scholes in d dimensions is a natural starting point.

---

## Crazy ideas

These are long-horizon, high-risk, high-reward directions. None are on the near-term roadmap, but they represent genuine open problems where `pinnrl`'s architecture could provide a foundation.

### Multi-agent RL for domain decomposition

Rather than a single RL agent selecting collocation points globally, deploy a population of agents — one per subdomain — that communicate through shared boundary information. Each agent optimizes local residual reduction while a coordinator agent manages interface conditions. This maps naturally to the XPINNs (Extended PINNs) framework and could yield massive parallelism on distributed hardware.

**Why it matters:** Domain decomposition is the primary path to scaling PINNs to 3D engineering problems. RL-based coordination of agents could outperform static decomposition strategies.

### PDE discovery from data (SINDy + PINNs)

Rather than specifying the PDE governing equation, learn it from data. The Sparse Identification of Nonlinear Dynamics (SINDy) framework identifies governing equations by fitting sparse linear combinations of candidate terms. Combining SINDy's candidate library with a PINN's automatic differentiation could allow joint identification of both the equation structure and its solution.

**Why it matters:** In fields like fluid mechanics, materials science, and systems biology, the governing PDE is often unknown or only partially known. Equation discovery from experimental data is a major open problem.

### Differentiable physics for robotics and control

Export trained PINN solutions as differentiable physics modules that can be embedded in reinforcement learning loops for robotics. A PINN trained on the pendulum equation, for example, is a differentiable, zero-shot simulator that generalizes across initial conditions — unlike look-up tables or ODE integrators.

**Why it matters:** Differentiable physics is a central challenge in model-based RL. PINNs trained on the relevant governing equations could replace expensive simulation environments in data-efficient RL.

### Turbulence modeling with PINNs

Extend the Navier-Stokes formulation (not yet in `pinnrl`) to turbulent flows, using a PINN as a closure model for Reynolds stresses. The PINN would be trained on high-fidelity DNS (Direct Numerical Simulation) data and used to generalize to new flow configurations.

**Why it matters:** Turbulence closure is one of the great unsolved problems in applied mathematics. PINN-based closures that respect physical conservation laws (by construction, through the physics loss) could outperform purely data-driven neural closures.

### Option pricing with Greeks computation via autograd

The Black-Scholes model is already implemented. A natural extension is multi-asset options pricing in high dimensions (d ≥ 10), combined with real-time computation of Greeks (Delta, Gamma, Vega, Theta) via the same autograd graph used to compute the PDE residual. This would make `pinnrl` directly useful in quantitative finance.

**Why it matters:** Monte Carlo methods for high-dimensional options pricing are computationally expensive and do not provide Greeks analytically. A PINN-based pricer that returns Greeks for free via autograd would be commercially valuable.

### Neural operator learning (FNO integration)

Integrate the Fourier Neural Operator (FNO) as a seventh architecture option. Unlike standard PINNs (which learn a single solution), FNO learns an operator mapping: initial/boundary conditions → solution. This enables zero-shot generalization to new IC/BC pairs without retraining.

**Why it matters:** FNO (Li et al., 2021) is one of the most impactful developments in scientific machine learning. Combining FNO's operator-learning capability with `pinnrl`'s RL adaptive sampling could yield a system that learns physics operators efficiently. The `mapping_size` and Fourier feature infrastructure already in `pinnrl` provides a natural foundation.
