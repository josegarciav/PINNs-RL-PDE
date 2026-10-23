# Experiment backlog

100 experiments for `pinnrl`, grouped into ten tracks of ten and ranked by
priority. Pick from the top; when you start one, create its folder (see
[README.md](README.md)) and change its status here.

## Priority levels

| Level | Meaning |
|---|---|
| **P0** | Correctness or baseline. Later results are not trustworthy until it is done. |
| **P1** | Core research claim: the sampling benchmark paper, or inverse problems toward the plant twin. |
| **P2** | Breadth: fills in the picture, informs defaults. |
| **P3** | Exploratory: worth a timeboxed look once P0–P1 are in hand. |

Status values: `todo`, `in progress`, `done`, `dropped` (say why).

## Default protocol

Unless an experiment says otherwise:

- **Seeds:** 5 for P0/P1 claims, 3 for P2/P3. Pair seeds across arms.
- **Budget:** fix the number of residual evaluations, *including* the extra
  evaluations a method spends on its own (RAD candidate pools, bandit probes).
  Report wall-clock time alongside.
- **Error metric:** relative L2 against a reference on a fixed held-out grid:
  the analytical `exact_solution`, `pinnrl.benchmarks.reference`, or
  `pinnrl.benchmarks.fdm`. Also report max error and final residual.
- **Reporting:** median and IQR across seeds; Wilcoxon signed-rank test for
  paired comparisons. Keep the config and the seed list with the result.

## Run first

The shortest path to results you can trust and publish:

1. ~~**A01**, **A08**, **A09**, **H08**~~ — done: the physics they audited is fixed and
   pinned by `tests/unit_tests/test_physics_consistency.py`
2. **A05** seed-variance baseline (sets the seed count for everything else)
3. **I01** profiling (tells you what a budget costs)
4. **A02** PDE × architecture baseline matrix
5. **B01** main sampling benchmark (in progress) and **B06** budget fairness
6. **D01** optimizer baseline
7. **H01**, **H03** the known failure modes the sampling paper must address
8. **K32** verified reference solvers, then the inverse-modelling baselines: **F01** with **K01**, **K11**
   (identifiability), **K39** (likelihood weighting) and **K05** (disturbance response)
9. **J01** evaporator reference simulator

---

## A. Correctness and baselines

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| A01 | P0 | Exact-solution audit | Do the configured analytical solutions actually satisfy their PDE, IC and BC? | For each `pde_configs` entry, evaluate the residual, IC error and BC error of `exact_solution` with autograd on a dense grid. Flag anything above 1e-4. Cahn–Hilliard "spinodal", Black–Scholes and pendulum first. | done — found 7 of 10 configs inconsistent; fixed, guarded by `tests/unit_tests/test_physics_consistency.py` |
| A02 | P0 | PDE × architecture baseline matrix | Which of the 10 PDE configs × 7 architectures train at all under default settings? | Fixed budget, 3 seeds. Table of rel. L2 per cell, plus a pass/fail cut at rel. L2 < 0.1. This is the reference table for every later comparison. | todo |
| A03 | P0 | PINN vs FDM on heat and wave | How far is a PINN from a classical solver at matched wall-clock time? | `pinnrl-benchmark fdm` vs the PINN at 3 time budgets. Rel. L2 vs time curve. | todo |
| A04 | P0 | Reference-solver convergence | Are the Burgers (Cole–Hopf) and Allen–Cahn (spectral) references accurate enough to grade PINNs? | Grid/quadrature refinement study for `benchmarks/reference.py`. Compare against the published Raissi et al. Burgers data. Target reference error below 1e-5. | todo |
| A05 | P0 | Seed-variance baseline | How many seeds are needed to detect a 2× error difference? | 20 seeds of the default heat and Burgers configs. Spread of rel. L2, and a power analysis that fixes the seed count for later experiments. | todo |
| A06 | P0 | Determinism and device drift | Does the same seed give the same result, and how much do CPU and MPS differ? | Repeat runs on CPU and MPS with fixed seeds. Max deviation of loss curves and final error. | todo |
| A07 | P1 | Early-stopping validity | Does validation loss (what early stopping watches) track true rel. L2? | Log both every N epochs across A02 runs. Rank correlation, and how often early stopping picks a worse model than the final one. | todo |
| A08 | P0 | Boundary/IC point audit | Do non-heat PDEs need more than the fixed 100-point boundary and IC grids that `PDEBase.compute_loss` uses? | Only `HeatEquation` reads `num_boundary_points`/`num_initial_points`. Sweep 25–2000 boundary/IC points on Burgers, wave and Allen–Cahn. Rel. L2 vs point count. | done — every PDE now honours `num_boundary_points`/`num_initial_points`; the sweep itself is still open |
| A09 | P0 | Periodic BC audit | Is periodicity enforced for non-heat PDEs? | The base `"periodic"` BC fits `sin(2πx)` instead of enforcing `u(x_min) = u(x_max)`. Measure the periodicity gap on convection, then compare against a true periodic penalty or a periodic feature embedding. | done — `periodic` now matches u and u_x on opposite faces (tested) |
| A10 | P1 | Float32 vs float64 floor | Is the achievable error limited by precision? | Rerun the best A02 cells in float64. Lowest rel. L2 reached and the cost ratio. | todo |

## B. Collocation sampling benchmark (paper core)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| B01 | P0 | Main sampling benchmark | Uniform vs stratified vs RAD vs RL bandit at equal budget. | `pinnrl-benchmark sampling` on Burgers and Allen–Cahn (`x2_cos`), 10 seeds, 15k steps each (equal training points), plus a uniform-every-100-steps control. Rel. L2 vs `benchmarks.reference`. See `B01_sampling_equal_compute/`. | in progress |
| B02 | P1 | All 1-D PDEs | Does B01's ranking hold on heat, wave, convection, KdV and Cahn–Hilliard? | Same protocol as B01 on each PDE. Rank table and critical-difference diagram. | todo |
| B03 | P1 | RAD `k`, `c` grid | Which RAD hyperparameters work, and are they PDE-dependent? | k ∈ {0, 0.5, 1, 2, 4} × c ∈ {0, 0.5, 1, 2} (the Wu et al. 2023 grid) on Burgers and Allen–Cahn. Heatmap of rel. L2. | todo |
| B04 | P1 | Resampling frequency | How often should points be resampled? | Every batch vs every 10/100 epochs vs a fixed set, for uniform and RAD. Error vs resampling interval. | todo |
| B05 | P1 | RAD candidate-pool size | Accuracy vs cost of `pool_factor`. | pool_factor ∈ {2, 4, 8, 16}, charged to the budget. Pareto front of error vs evaluations. | todo |
| B06 | P0 | Budget fairness | Do adaptive methods still win when their extra residual evaluations count against the budget? | From the B01 run: compare at equal steps and at equal total residual evaluations (overhead counted, from per-checkpoint accounting). Report both rankings. | in progress (with B01) |
| B07 | P2 | Quasi-random sequences | Do Sobol/Halton beat LHS and jittered grids? | Bring `scipy.stats.qmc` from `sampling_techniques.ipynb` into a sampler. Compare on heat and Burgers. | todo |
| B08 | P2 | Collocation-count scaling | Error vs number of points for each strategy. | N ∈ {500, 1k, 2k, 5k, 10k, 50k}. Log–log efficiency curves; read off accuracy per point. | todo |
| B09 | P2 | Causal weighting vs RAD | Does respecting causality (Wang et al. 2022) matter more than residual focus? | Causal residual weighting vs RAD vs both, on Allen–Cahn and KdV. | todo |
| B10 | P2 | Sampling in 2-D | Does the adaptive-sampling advantage grow with dimension? | heat_2d with uniform/stratified/RAD at matched budget. Compare the gap against the 1-D gap. | todo |

## C. RL bandit sampler (experimental)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| C01 | P1 | Bandit vs RAD, cost-adjusted | Is the bandit worth its probe cost compared with RAD? | Subset of B01/B06 with the bandit's probes charged to the budget. The deciding experiment for whether `--rl` stays. | todo |
| C02 | P1 | Cell resolution | How fine should the action grid be? | n_x × n_t ∈ {4×2, 8×4, 16×8, 32×16} on Burgers. Error and policy entropy over training. | todo |
| C03 | P1 | Policy fraction | How much of each batch should the policy control? | policy_fraction ∈ {0.25, 0.5, 0.75, 1.0}. Error, and coverage of the cells the policy starves. | todo |
| C04 | P2 | Temperature and ε schedule | Exploration settings. | temperature ∈ {0.1, 0.5, 1, 2} × ε decay ∈ {0.99, 0.998, 0.9995}. | todo |
| C05 | P2 | Does context help? | Do the state features (log residual stats, progress) improve on a context-free bandit? | Full state vs progress only vs constant state. | todo |
| C06 | P2 | Reward definition | Which reward drives the most useful sampling? | Relative mean \|r\| (current) vs \|r\|² vs loss decrease after training on the cell. | todo |
| C07 | P2 | Classical bandits | Does the DQN beat UCB1, Thompson sampling or EXP3 over the same cells? | Drop-in replacements for the agent with the same probe budget. | todo |
| C08 | P2 | Probe budget | Trade-off between probe_cells and probe_points. | Grid at a fixed total of probe evaluations. | todo |
| C09 | P3 | Policy-gradient sampler | PPO over continuous point placement (roadmap v0.4). | Gaussian-mixture policy over (x, t), reward = residual. Compare against C01. | todo |
| C10 | P3 | Transfer across PDEs | Does a pretrained bandit help on a nearby problem? | Train on Burgers ν=0.01, reuse on ν=0.005 and on Allen–Cahn. Steps to reach the same error. | todo |

## D. Optimization and loss balancing

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| D01 | P0 | Optimizer baseline | Adam vs L-BFGS vs Adam→L-BFGS. | Heat, Burgers, Allen–Cahn; 5 seeds. Rel. L2 and wall-clock time. | todo |
| D02 | P1 | Adam→L-BFGS switch point | When should the switch happen? | adam_lbfgs_switch_ratio ∈ {0.3, 0.5, 0.7, 0.9}. | todo |
| D03 | P1 | Adaptive loss weights | Does LRW or RBW beat tuned static weights? | Off vs lrw vs rbw on 5 PDEs. Error, plus weight trajectories. | todo |
| D04 | P1 | Static weight sensitivity | How sensitive is accuracy to the boundary and initial weights? | Boundary and initial weights ∈ {1, 10, 100}, 3×3 grid per PDE. | todo |
| D05 | P2 | Loss function under noise | MSE vs MAE vs Huber with clean and noisy observations. | Inverse heat with noise σ ∈ {0, 0.05, 0.2}, plus 5% outliers. Parameter error. | todo |
| D06 | P2 | LR schedule | Cosine vs reduce-on-plateau vs constant, and the LR value. | LR ∈ {1e-4, 1e-3, 5e-3, 1e-2} × 3 schedules. | todo |
| D07 | P2 | Gradient clipping | Does clipping help or hurt? | clip ∈ {0, 1, 10} on Burgers and KdV. | todo |
| D08 | P2 | Mini-batch vs full batch | Is stochasticity a useful regulariser for Adam? | batch_size ∈ {256, 1024, full}. | todo |
| D09 | P2 | Hard vs soft constraints | Exact IC/BC by output transform, `u = g + d(x,t)·N`, vs penalty terms. | Heat, wave, Burgers. Error, and the convergence speed of residual loss. | todo |
| D10 | P3 | Second-order optimisers | Do newer methods beat L-BFGS (e.g. SOAP, NysNewton-CG)? | Best D01 configs, with time to reach the target error. | todo |

## E. Architectures

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| E01 | P1 | Parameter-matched comparison | Which architecture wins at roughly 100k parameters? | All 7 architectures on 5 PDEs, widths tuned to equal parameter count. | todo |
| E02 | P1 | SIREN ω₀ | Sensitivity to the frequency hyperparameter. | ω₀ ∈ {1, 5, 10, 30} on wave and KdV. | todo |
| E03 | P1 | Fourier-feature scale | Spectral bias vs noise as the scale grows. | scale ∈ {1, 2, 4, 10} × mapping_size ∈ {64, 256, 512} on heat and convection. | todo |
| E04 | P2 | Depth vs width | At a fixed parameter count, is deeper or wider better? | FeedForward and ResNet, 3 depth/width splits. | todo |
| E05 | P2 | Activation for high-order PDEs | tanh vs gelu vs silu vs sin for 3rd- and 4th-order derivatives. | KdV and Cahn–Hilliard. | todo |
| E06 | P2 | Layer norm | Does the default `layer_norm: true` hurt higher derivatives? | On/off for feedforward and fourier on KdV and Allen–Cahn. | todo |
| E07 | P2 | FNO in pointwise mode | Is FNO meaningful as a pointwise PINN backbone, or only as an operator? | FNO in PINN mode vs data_only on active_matter (G01). | todo |
| E08 | P2 | Keep-or-cut: attention and autoencoder | Do these beat an MLP anywhere? | Best results from A02 and E01. If they never win, remove them from the package. | todo |
| E09 | P3 | Separable PINNs for 2-D | Per-axis subnetworks with an outer product. | heat_2d, against E01's winner. | todo |
| E10 | P3 | KAN-style layers | Kolmogorov–Arnold layers vs MLP. | Heat and Burgers at equal parameter count. | todo |

## F. Inverse problems and identifiability

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| F01 | P0 | Heat α recovery baseline | Accuracy of α vs noise and number of observations. | noise ∈ {0, 0.01, 0.05, 0.1} × N_obs ∈ {20, 50, 200, 1000}. Relative error of α. | todo |
| F02 | P1 | Joint KdV (α, β) | Are two parameters jointly identifiable? | Recovery error and correlation of the estimates across seeds. | todo |
| F03 | P1 | Burgers ν near the shock | Recovery of ν at ν = 0.01/π. | Same noise grid as F01. | todo |
| F04 | P1 | Initial-guess sensitivity | Does the estimate depend on the starting guess? | 10 initial guesses spanning 0.1–10× the true value. | todo |
| F05 | P1 | Fixed sensors | Plant-like data: a few fixed probe locations, densely sampled in time. | 2–8 fixed x positions vs random scatter at an equal data count. | todo |
| F06 | P1 | Uncertainty and coverage | Are ensemble confidence intervals for α calibrated? | 10-member ensembles over 50 synthetic datasets. Empirical coverage of the 90% CI. | todo |
| F07 | P2 | Profile likelihood | Practical identifiability via profile likelihood. | Fix α on a grid, retrain, plot the loss profile. Flat profile means not identifiable. | todo |
| F08 | P2 | Model-form error | What does a parameter estimate absorb when the model is wrong? | Fit a heat model with unknown α to Burgers data. Bias, and the residual structure. | todo |
| F09 | P2 | Time-varying parameter | Can a drifting α(t), fouling-like, be recovered? | Small network for α(t). Synthetic linear and step drifts. | todo |
| F10 | P2 | Classical baseline | PINN inverse vs least squares around an FDM forward model. | `scipy.optimize` + `benchmarks/fdm.py`. Accuracy and time. | todo |

## G. Data modes and The Well

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| G01 | P1 | FNO on active_matter | Baseline numbers for data_only FNO, comparable with The Well paper. | Notebook 04 setup, scaled up. VRMSE as in The Well paper. | todo |
| G02 | P1 | Three modes on Burgers | data_augmented vs data_only vs forward with sparse reference data. | Burgers with 1% of the reference grid as data. | todo |
| G03 | P1 | How much data does physics save? | Data count needed to reach a target error, with vs without the residual. | n_points ∈ {128 … 16k}, data_only vs data_augmented. | todo |
| G04 | P2 | Loader throughput | Streaming vs local cache. | Time per epoch and first-batch latency. | todo |
| G05 | P2 | Gray–Scott with physics | data_augmented with a new reaction–diffusion residual. | New PDE class; data_only vs data_augmented. | todo |
| G06 | P2 | Multi-channel outputs | Does collapsing to channel 0 hide errors on shear_flow and rayleigh_benard? | Per-channel errors. | todo |
| G07 | P2 | Across-trajectory generalisation | Train on 1–16 trajectories, test on held-out ones. | Error on held-out trajectories vs n_traj. | todo |
| G08 | P3 | Extrapolation in time | Train on t ∈ [0, 0.5], test on [0.5, 1]. | data_only vs data_augmented. | todo |
| G09 | P3 | Noise robustness | Gaussian noise on Well data. | σ ∈ {0, 1%, 5%, 10%} of the signal std. | todo |
| G10 | P3 | Super-resolution | Train on 4× downsampled fields, evaluate at full resolution. | data_augmented vs bicubic interpolation. | todo |

## H. Hard PDEs and known failure modes

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| H01 | P1 | Convection failure (Krishnapriyan et al. 2021) | At what velocity β does the PINN fail, and does RAD or curriculum fix it? | β ∈ {1, 10, 30, 50}, uniform vs RAD vs curriculum over β. | todo |
| H02 | P1 | Burgers viscosity sweep | Error vs shock sharpness. | ν ∈ {0.1, 0.01, 0.01/π, 0.001}. | todo |
| H03 | P1 | Allen–Cahn benchmark | Reproduce the known failure on `x2_cos` and test the fixes. | ε sweep; vanilla vs RAD vs time-marching vs causal weighting. | todo |
| H04 | P1 | Time-marching windows | Sequential time windows vs the whole domain at once. | Allen–Cahn and KdV, 2/4/8 windows. | todo |
| H05 | P2 | Parameter curriculum | Coarse-to-fine schedule on ν or β (roadmap v0.4). | Against H01/H02 at equal total epochs. | todo |
| H06 | P2 | KdV long-time propagation | Error growth with horizon. | T ∈ {1, 2, 5}. Soliton position and amplitude error. | todo |
| H07 | P2 | Cahn–Hilliard mixed form | Auxiliary μ output (two 2nd-order equations) vs direct 4th order. | Error, and time per epoch. | todo |
| H08 | P1 | Wave initial-velocity audit | `u_t(x,0)` is not enforced; is the wave problem well-posed as configured? | Add the u_t IC term; compare error with and without it. | done — second-order PDEs enforce `u_t(x, 0)` (tested); the with/without comparison is still open |
| H09 | P2 | Black–Scholes Greeks | Accuracy of delta and gamma via autograd vs the closed form. | Error in price, delta and gamma near the strike. | todo |
| H10 | P3 | Pendulum long horizon | Energy drift vs RK45. | Horizons of 10–100 periods. | todo |

## I. Scaling, performance and engineering

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| I01 | P0 | Epoch time breakdown | Where does training time go? | Profile residual autograd, BC/IC, sampling, logging and the live snapshot npz. Fraction of time per stage. | todo |
| I02 | P1 | Device throughput | CPU vs MPS vs CUDA per architecture. | Batch sweep. Points/s and the crossover batch size. | todo |
| I03 | P1 | 1-D to 2-D cost | Points and time needed to keep the error constant from heat to heat_2d. | Error vs N for both; the scaling exponent. | todo |
| I04 | P2 | Vectorised derivatives | `torch.func` (vmap/jacrev) vs the autograd loops in `compute_derivatives`. | Speed-up and numerical agreement. | todo |
| I05 | P2 | `torch.compile` on Python 3.14 | `torch.jit.script` in the Fourier network is deprecated; is compile faster and correct? | Time per epoch and output equality. | todo |
| I06 | P2 | Mixed precision | bf16/fp16 effect on higher derivatives. | KdV and Cahn–Hilliard residual error. | todo |
| I07 | P2 | Parallel seeds | Process-pool seeds (as in sampling_study) vs sequential on M-series. | Throughput per core count. | todo |
| I08 | P2 | 3-D heat feasibility | Can the package handle a 3-D domain? | New config; error and memory. | todo |
| I09 | P3 | Domain decomposition | XPINN/FBPINN-style subdomains in 2-D. | Against I03's single-domain result. | todo |
| I10 | P3 | High-dimensional heat | d = 5, 10. How does the error scale? | Monte-Carlo collocation; error vs d. | todo |

## J. Industrial digital-twin track

Each core feature must also be tested on a textbook PDE (see the A–I tracks)
before it counts as done.

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| J01 | P1 | Evaporator reference simulator | A trusted forward model to grade everything else against. | 1-D falling-film evaporator energy/mass balance, steady and dynamic, with a classical solver (FDM/ODE). Grid convergence. | todo |
| J02 | P1 | PINN forward solve of J01 | Can the PINN match the reference simulator? | Rel. L2 on temperature and concentration profiles. | todo |
| J03 | P1 | Fouling estimation | Recover a time-varying fouling resistance R_f(t) from synthetic sensors. | Inverse mode with R_f(t) as a small network (builds on F09). | todo |
| J04 | P1 | Plant-like sensors | Is R_f identifiable from outlet/boundary measurements only? | Sensor operator on the J01 outputs; profile likelihood (F07 method). | todo |
| J05 | P2 | EKF/UKF baseline | PINN estimator vs a Kalman filter on a lumped model. | Estimation error, latency, tuning effort. | todo |
| J06 | P2 | Calibrated fouling intervals | Are the R_f uncertainty intervals calibrated? | Ensembles or Laplace; coverage as in F06. | todo |
| J07 | P2 | Single-droplet drying model | Forward PINN for droplet moisture and temperature. | Against a classical ODE/PDE reference. | todo |
| J08 | P2 | Stickiness estimation | Estimate the glass-transition margin T − T_g from outlet air T and RH. | Synthetic data from J07 plus noise. | todo |
| J09 | P3 | Measurement design | Which sensor placement minimises parameter variance? | Fisher information over candidate sensor sets on J01/J07. | todo |
| J10 | P3 | Line-level twin | Coupled evaporator → dryer with shared states. | Couple J01 and J07; error propagation across units. | todo |

## K. Conditioning, stability and numerical analysis

The theory behind trusting an estimate of unknown physics, as 50 baseline experiments in six themes. Diagnostics (K.1–K.2) come before UQ methods (K.5): an unidentifiable or ill-conditioned problem has no meaningful error bar. Shared notation: sensitivity Jacobian $J = \partial y/\partial\theta$; Fisher information $F = J^\top\Sigma^{-1}J$; condition number $\kappa$.

### K.1 Identifiability and sensor information (paper P08)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| K01 | P0 | Fisher information and identifiability | Which parameter combinations do the sensors determine? | $J$ by autograd, $F$, its singular values, $\kappa(F)$ and sloppy directions, for heat α, KdV (α, β) and Burgers ν. | todo |
| K02 | P1 | Sensor layout vs conditioning | How do sensor count and placement change $\kappa(F)$? | A grid of sensor sets on K01's problems; $\kappa(F)$ and the parameter error against the number of sensors. Feeds J09. | todo |
| K11 | P0 | Structural identifiability | Which parameters can never be separated, whatever the noise? | Symbolic analysis (sympy) of scaling symmetries and modal reductions: pendulum (g, L, b), KdV (α vs amplitude), heat (α from boundary flux). Expected rank deficiencies, checked against K01. | todo |
| K12 | P1 | Profile likelihood vs Fisher regions | When does the Gaussian (Fisher) approximation of confidence regions fail? | Profile likelihood vs $F^{-1}$ ellipses on nonlinear problems (Burgers ν, pendulum); coverage of each. Links F07. | todo |
| K13 | P1 | Observation window | When in time do measurements carry information? | Fisher information against window start and length (heat decay, Allen–Cahn approach to steady state); information per sample. | todo |
| K14 | P1 | Correlated parameters | How strongly coupled are jointly identified parameters? | Joint identification of pendulum (g, L, b), KdV (α, β) and Black–Scholes (σ, r); correlation matrix, eigen-directions, error per direction. | todo |
| K15 | P1 | Functionals only | What can boundary or integral sensors recover? | Rank and spectrum of $J$ when only boundary values, fluxes or spatial averages are observed (heat, convection). The template for evaporator sensors (J04). | todo |
| K16 | P1 | Accuracy of PINN sensitivities | Are autograd sensitivities through a trained PINN trustworthy? | ∂u/∂θ from the PINN vs finite differences of the reference solver; error against training residual. | todo |
| K17 | P2 | Optimal sensor placement baseline | How much do design criteria beat ad-hoc layouts? | Greedy D-, A- and E-optimal selection vs uniform and random layouts at a fixed sensor count; parameter variance. | todo |

### K.2 Training conditioning and optimization geometry (paper P09)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| K03 | P1 | Training conditioning | Is PINN training ill-conditioned by the differential operator? | Loss-Hessian / NTK spectra (Lanczos) across PDEs and architectures; condition number vs final error. | todo |
| K04 | P2 | Preconditioning | Do second-order or preconditioned optimizers fix K03's ill-conditioning? | L-BFGS, Gauss–Newton and NysNewton-CG vs Adam on the worst K03 cases, at equal budget. Links D10. | todo |
| K18 | P1 | Conditioning vs PDE order | How does κ grow with derivative order? | Hessian condition number for 2nd (heat), 3rd (KdV) and 4th (Cahn–Hilliard) order at matched networks; fitted growth law. | todo |
| K19 | P1 | Conditioning vs sharpness | How does κ scale with frequency or interface width? | Heat frequency k, Burgers ν and Allen–Cahn ε sweeps; κ and error against the parameter (log–log slope). | todo |
| K20 | P1 | Loss-term imbalance | How do residual, BC and IC terms shape the Hessian? | Block structure of the Hessian per loss term; effect of static weights and LRW/RBW on κ. Links D03, D04. | todo |
| K21 | P1 | Non-dimensionalization | How much conditioning comes from units? | The same problem in physical vs scaled variables for x, t, u and θ; κ, error and steps to tolerance. Baseline for the v0.6 scaling helpers. | todo |
| K22 | P2 | NTK dynamics | How does the kernel spectrum evolve during training, per architecture? | NTK eigenvalues at checkpoints for MLP, Fourier and SIREN; the spectral-bias timeline. Links E01–E03. | todo |
| K23 | P2 | Inverse loss landscape | Are there multiple minima or flat valleys in θ? | Loss slices over θ (network re-optimized per slice) for KdV (α, β) and pendulum (g, L); count of basins. | todo |
| K24 | P2 | Precision floor | Does float32 corrupt curvature estimates? | $F$ and Hessian spectra in float32 vs float64; smallest resolvable singular value. Links A10. | todo |

### K.3 Disturbance and robustness (paper P10)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| K05 | P0 | Data-disturbance response | How far does the estimate move under sensor noise, bias, dropout and delay? | Perturbation sweeps; empirical δθ vs the linear prediction $F^{-1}J^\top\Sigma^{-1}\delta y$ from K01. | todo |
| K25 | P1 | Sensor bias | How does a systematic offset bias θ, and can the residual detect it? | Constant and drifting offsets on one or all sensors; bias in θ and the change in final residual. | todo |
| K26 | P1 | Missing data | Random vs block (sensor-failure) gaps. | Missing-at-random vs whole-sensor loss vs time gaps at matched data counts; θ error and $\kappa(F)$. | todo |
| K27 | P1 | Timing errors | Sensitivity to timestamp misalignment and delay. | Constant delays and jitter on observation times; θ bias against delay/timescale ratio. | todo |
| K28 | P1 | Outliers and heavy tails | Which loss keeps θ stable under outliers? | Student-t and contaminated noise; MSE vs Huber vs MAE data terms; θ error and breakdown point. Links D05. | todo |
| K29 | P1 | Model-form disturbance | How biased is θ when the physics is incomplete? | Data from heat plus a small source or advection term, fitted with pure heat; bias in α against the size of the missing term. Links F08. | todo |
| K30 | P1 | Basin of attraction | Which initial guesses converge to the truth? | Multi-start over a log-grid of initial θ; convergence map and success rate. Links F04. | todo |
| K31 | P2 | Worst-case disturbance | Is the linear theory's worst direction really the worst? | Perturb data along the top singular vector of $F^{-1}J^\top\Sigma^{-1}$ vs random directions of equal norm; ratio of the θ errors. | todo |

### K.4 Stability and the numerical-analysis bridge (papers P10, P11)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| K06 | P1 | Problem-disturbance stability | Does the learned solution respect the PDE's own stability? | Perturb IC, BC and parameters; compare the PINN's response with the reference solver's (amplification ratios). | todo |
| K07 | P1 | Empirical stability constants | How does the training residual translate into solution error? | Ratio of error to residual across PDEs and runs; compare with conditional-stability bounds (error ≤ C · (residual + quadrature error)). | todo |
| K32 | P0 | Verified reference solvers | Do the classical references converge at their theoretical order? | Grid-refinement studies of the FDM, method-of-lines and Cole–Hopf references; observed order vs theory, von Neumann/CFL limits. Links A04. | todo |
| K33 | P1 | Consistency vs convergence | When does a small residual not mean a small error? | Residual and error as points and steps grow; the PINN analogue of Lax equivalence; counterexamples (Allen–Cahn, convection). | todo |
| K34 | P1 | Structure preservation | Does the PINN respect energy laws? | Heat $L^2$ decay, wave energy conservation, Allen–Cahn free-energy monotonicity, Cahn–Hilliard mass conservation; size of violations vs error. | todo |
| K35 | P1 | Stability rates of inverse problems | How fast does the θ error shrink with noise level δ? | log–log slope of θ error vs δ: Lipschitz (heat α forward-in-time) vs logarithmic (backward heat, unique continuation). The classes of ill-posedness on known examples. | todo |
| K36 | P1 | Long-time stability | Does error grow linearly or exponentially in time? | Pendulum, KdV and wave over long horizons; error growth rate vs the reference solver's. Links H06, H10. | todo |
| K37 | P2 | Stiffness | How does stiffness translate into PINN difficulty? | Allen–Cahn and Cahn–Hilliard with decreasing ε; stiffness ratio of the method-of-lines Jacobian vs PINN error and κ. | todo |
| K38 | P1 | Quadrature error | The gap between training residual and true residual. | Residual on the training points vs a dense independent grid as N grows; the quadrature term of the error bound. Links B08. | todo |

### K.5 Uncertainty quantification baselines (paper P13)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| K10 | P1 | Interval methods compared | Which UQ method gives calibrated intervals for identified physics? | Ensembles vs Laplace (reusing K01's Fisher) vs conformal, on F01/F02; coverage, width, CRPS. With F06. | todo |
| K39 | P0 | Likelihood weighting | Does setting the data weight from the noise model beat tuned weights? | Data weight $1/\sigma^2$ (Gaussian likelihood) vs tuned static weights; θ error and calibration of the resulting intervals. | todo |
| K40 | P1 | Ensemble size | How many members are enough? | Deep ensembles of 3, 5, 10 and 20; coverage, width and cost. | todo |
| K41 | P1 | Laplace variants | How much of the network must the Laplace approximation cover? | θ only vs last layer vs full generalized Gauss–Newton; coverage vs cost. | todo |
| K42 | P2 | Reference posteriors | A gold standard to judge cheap methods. | HMC/NUTS on small PINNs for heat α and pendulum (g, L); posterior vs ensemble and Laplace. | todo |
| K43 | P2 | Cheap approximate methods | Do MC-dropout and mean-field VI give usable intervals? | Coverage and width vs K42's reference posteriors; documented failure modes. | todo |
| K44 | P1 | Conformal intervals | Distribution-free coverage for field predictions. | Split conformal on held-out sensors for u(x, t); coverage vs number of calibration points. | todo |
| K45 | P1 | Coverage under shift | Do intervals stay calibrated off the training regime? | Extrapolation in time and to unseen parameter or operating conditions; coverage degradation per method. | todo |
| K46 | P1 | Posterior predictive checks | Can residual diagnostics flag model-form error? | Predictive checks and residual whiteness tests on K29's misspecified data; detection rate. | todo |
| K47 | P2 | Field-valued UQ | Pointwise intervals for u(x, t) and for a coefficient field α(x). | Ensemble and Laplace bands vs truth; coverage maps. Links F09. | todo |

### K.6 Dimension and coupled systems (papers P12, P23)

| ID | P | Experiment | Question | Setup and metric | Status |
|---|---|---|---|---|---|
| K08 | P2 | Dimension scaling | How do stability constants and quadrature error grow with dimension? | Heat and convection in d = 1…5 at matched points per dimension and matched total points; links I10. | todo |
| K09 | P2 | Linearized stability of coupled systems | Stiffness and stability of multi-field process models. | Jacobian eigenvalues of the method-of-lines system for J01 (and later J10); stiffness ratio; consequences for PINN training. | todo |
| K48 | P1 | Coupled fields | Are cross-coupling coefficients identifiable? | A two-field reaction–diffusion system (e.g. Gray–Scott); $\kappa(F)$ for diffusion vs reaction coefficients; θ error. | todo |
| K49 | P1 | Many parameters | How does conditioning degrade as the parameter count grows? | Layered-medium heat with 2–20 piecewise diffusivities; $\kappa(F)$ and error against the count; when regularization becomes necessary. | todo |
| K50 | P2 | High-dimensional stability | Stability constants and sampling in many dimensions. | Heat in d = 2…10: empirical stability constant (K07) and Monte Carlo vs quasi-Monte Carlo sampling error (B07). | todo |
