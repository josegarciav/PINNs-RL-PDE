# Changelog

All notable changes to `pinnrl` are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Changed
- **PDE unknowns get no weight decay.** The scalar parameters identified in inverse mode and the weights of unknown functions form their own Adam parameter group with `weight_decay=0`. The configured decay still applies to the network. Decay on a physical parameter is a prior toward zero; with Adam's normalization it drove the weights of an unknown function to zero at the learning rate whenever the data gradient was weak, which flattened every recovered field to a constant (found in F09). A full rerun of F01 without the decay reproduces every cell, so the scalar results stand.
- **The inverse recipe is now documented as `weight_decay=0`, `gradient_clipping=0`.** D01 found and F01's full-grid rerun (`--clean`) confirms that the packaged network regularization was the 2 % floor of the identified heat diffusivity: with both off the floor is 0.1–0.2 % and the error follows the Cramér–Rao bound. The packaged defaults are unchanged (they are forward-fit defaults); the inverse-problem docs and the experiments state the clean settings.

### Added
- **L-BFGS safeguard.** An L-BFGS step whose loss is not finite or exceeds the loss before the step tenfold is rolled back and the L-BFGS learning rate halved; after three such steps in a row the trainer returns to Adam. D01 and J02 saw the line search step to an unbounded loss on a network Adam had not yet smoothed.
- **Field-valued unknowns** (roadmap v0.6 item 1). `CustomPDE(unknown_functions={"alpha": {...}})` replaces a coefficient by an `UnknownFunction`, a small network of the coordinates and fields it depends on, used in the expressions like a parameter and differentiable with the subscript notation (`alpha_x`). Constraints are built into the output: `positive` (softplus), `bounds` (sigmoid) and `monotone` (non-negative weights on the paths from the chosen inputs). `initial` sets the constant start, `truth` the function the synthetic data come from, which `generate_synthetic_observations`, an exact solution that mentions the unknown, `pde.evaluate_unknown(..., truth=True)` and `pde.unknown_function_error` use. The weights join the inverse optimizer, the YAML spec (`to_dict`, `--pde-file`), the run folder (`unknown_functions.pt`) and `load_run`. `Fields` exposes `f.alpha`, `f.alpha_x` and `f.unknowns` to callable residuals.
- **Process models** (v0.7, J01): `pinnrl.process.FallingFilmEvaporator`, a falling-film evaporator tube (liquid flow and solids fraction transported along a heated tube, boiling-point elevation, fouling resistance, measured feed and steam signals) with a steady-state ODE solver, a method-of-lines dynamic solver (first-order upwind, BDF) and `custom_pde()` giving the same model as a `CustomPDE` with inputs, one-sided inlet conditions, scales, sensors and an optional unknown fouling function; `mol_rhs(nz)` exposes the method-of-lines right-hand side for stability analyses (K09). Dirichlet conditions given as `{left: ...}` alone constrain that face only (an inlet).
- **Parametric surrogates** (K45): `CustomPDE(parametric={"alpha": (lo, hi)})` makes a parameter an input of the network, sampled over its range with the collocation points and usable in the expressions like a coordinate; conditions apply to the physical faces only. `pde.surrogate_misfit(model, x, t, y, thetas)` is the vectorized data misfit of the trained surrogate, the forward model for the swarm. Experiment K45: one surrogate covers the prior range to 0.2–2 % at a quarter to a half of the cost of a network per value, and the swarm on its misfit reproduces the closed-form region.
- `UnknownFunction(..., basis={"type": "cosine" | "legendre", "modes": n})`: a field as a linear combination of fixed modes of one input, with `fn.coefficients` and `fn.modes(values)` for mode-wise identifiability analyses.
- `pinnrl.benchmarks.heat_variable_reference(alpha, u0, ...)`: the heat equation with a coefficient varying in space and time, `u_t = (alpha(x, t) u_x)_x`, in conservative form by the method of lines (BDF), verified against the closed form for `alpha(t)`. `ReferenceSolution.interpolate(x, t)` and `.as_exact_solution()` make any reference field usable as a `CustomPDE` exact solution.
- **Coupled systems** (roadmap v0.6 item 2): `CustomPDE(boundary_conditions={field: {kind: options}})` gives each field of a system its own boundary conditions, and `field_weights={field: w}` weights a field's residual, initial and boundary errors in the loss. New catalogue entry `reaction_diffusion_pair` (two heat equations exchanging mass, closed form) as the two-field test case.
- **Measured inputs** (roadmap v0.6 item 3): `CustomPDE(inputs={"q": TimeSeries | {"path": "q.csv", ...}})` makes a recorded signal of `t` available by name in the equations (a forcing) and in boundary values, read at each point's time by linear interpolation and held outside the record. Boundary values (`dirichlet` `value`/`left`/`right`, `neumann` `value`) accept expressions in the coordinates, parameters and inputs. `TimeSeries.from_file` reads CSV (header row) and NPZ; relative paths in a `--pde-file` or `from_yaml` spec are taken from the file's folder; the run snapshot keeps the inputs by path.
- **Sensor operators** (roadmap v0.6 item 4). `CustomPDE(sensors=[...])` takes `PointSensor` (a probe at a fixed place read at given times), `BoundarySensor` (an expression such as `-alpha*u_x` on a face) and `IntegralSensor` (a weighted domain integral, or mean), each with the expression it reports, its own `noise`, a transport `delay` (the record stamped `t` reflects the state at `t - delay`) and a `weight`. Records enter the data loss as one misfit term per sensor; `pde.observe(seed)` fills them from the exact solution through the operators (with the true parameters), `sensor.set_values` attaches measurements, and `pinnrl-train --pde-file` records synthetic ones when the spec's sensors carry none. `pde.evaluate_expression(text, model, x, t)` evaluates any expression of the fields and derivatives with a network. In a normalized problem a sensor's misfit is divided by the reported field's scale (or an explicit `scale=`), so a flow sensor in kg/s and a fraction sensor weigh alike (found in J03). `sensor_jacobian(pde, parameters)` differentiates the closed-form records with respect to the parameters, so `fisher_information` gives the Cramér–Rao bound and condition number of a sensor layout before training.
- **Non-dimensionalization** (roadmap v0.6 item 5). `CustomPDE(scales={"T": (300.0, 50.0)})` records each field's offset and magnitude; `build_trainer(normalize=True)` (or `pinnrl-train --normalize`) maps the network's inputs from the domain to `[-1, 1]`, returns `offset + scale * net`, and takes the loss terms in those units (errors divided by the scale, the residual multiplied by the time span over the scale). `ModelConfig` gains `input_bounds`, `output_scale` and `output_offset`, which `PINNModel` applies as buffers, so a checkpoint and `load_run` carry them. A heat problem in millimetres, seconds and kelvin trains to a 2 K error instead of 248 K.
- Experiment F09 (`experiments/F09_field_unknowns/`): recovery of a drifting `alpha(t)` (linear and step), a centred bump `alpha(x)` and a two-layer `alpha(x)` on heat, 3 noise levels, 5 seeds, against the best constant; summarized in [Benchmarks](docs/benchmarks.md).

## [0.5.0] — 2026-10-26

### Changed
- The sampling study's default resampling interval (`StudySettings.resample_every`, `pinnrl-benchmark sampling --resample-every`) is 10 steps instead of 100, following B04.
- Exact Dirichlet targets (`dirichlet: {type: exact}`) are no longer evaluated under `no_grad`, so a trainable parameter that appears in the closed form receives its boundary-term gradient.

### Added
- **User-defined equations.** `pinnrl.CustomPDE` builds a trainable problem from its residual written as text (`"u_t + u*u_x - nu*u_xx"`, parsed by SymPy, every derivative by autograd) or as a Python callable over a `Fields` accessor (`f.u_xx`, `f.laplacian("u")`). It covers scalar PDEs, ODEs (a domain with only `t`), steady problems (no `t`), systems (one equation per field, `(N, k)` residuals), second order in time with `initial_velocity`, expression-valued initial and exact solutions, and identifiable parameters. `pde.build_trainer(arch, **training)` returns a `PDETrainer` with the packaged defaults; `pinnrl-train --pde-file spec.yaml` trains from a YAML specification, and the run snapshot rebuilds it in `load_run`. `sympy` becomes a dependency.
- **A catalogue of 42 equations** (`pinnrl.pdes.catalog`, `from_catalog("kdv_soliton")`): heat in 1-D to 5-D, wave, telegraph, Klein–Gordon, sine–Gordon, KdV, a beam, Schrödinger, Poisson in 2-D and 4-D, Helmholtz, Bratu, Burgers, Fisher–KPP, Allen–Cahn, Cahn–Hilliard, porous medium, Black–Scholes, a 3-state HJB equation, the logistic, oscillator, pendulum, Van der Pol, Lotka–Volterra, Lorenz, SIR and CSTR ODEs, a two-parameter KdV, Gray–Scott, Brusselator, shallow water and Stokes. Every closed-form solution in it is verified to zero its residual by `test_catalog.py`.
- Residual-based sampling uses the norm over the equations of a system, and `validate` compares every component the exact solution provides.
- **Theory pages** (`docs/theory/`): well-posedness and conditional stability, identifiability and conditioning, the Bayesian view and calibration, and the numerical-analysis bridge, with the conditions the inverse-modelling modules cite.
- **`pinnrl.analysis`**: `exact_sensitivities` (symbolic `du/dtheta` of a closed-form solution), `fisher_information` and `closed_form_bound` (the Cramér–Rao bound, correlation and eigen-directions of the Fisher matrix) as the exact reference for the conditioning diagnostics; `surrogate_sensitivities` computes the same Jacobian from a trained network by the implicit-function theorem (matrix-free Hessian blocks, damped conjugate gradient), for problems without a closed form; `loss_hessian_spectrum` (Lanczos with Hessian-vector products) and `tangent_kernel_spectrum` measure the conditioning of training itself.
- **`pinnrl.analysis.disturbance`**: `perturb_observations` (noise, bias, dropout, delay, outliers), `linear_response` (the first-order parameter shift), `stability_ratio` (error over residual on a dense independent sample) and `field_response`.
- **`pinnrl.uq`**: `laplace_posterior` (a Gaussian over the parameters from the surrogate's Fisher information), `sample_equivalence_region` (a cloud particle swarm that samples the set of parameters within the noise tolerance, with a dispersion diagnostic), `ensemble_summary`, `likelihood_weights`, and the metrics `coverage`, `mean_width`, `crps_gaussian`, `crps_samples`.
- `CustomPDE.build_trainer(..., model={...})` overrides keys of the architecture block (`hidden_dims`, `layer_norm`, ...).
- The identification baseline (roadmap v0.5 item 2): experiments F01 (heat α vs noise and count), F02 (joint KdV α, β), F03 (Burgers ν as log ν), F04 (initial guess), F05 (fixed probes) and F07 (profile likelihood), each against the Cramér–Rao bound; and the conditioning diagnostics (item 3): K01 (surrogate Fisher information vs the exact layer) and K03 (Hessian and tangent-kernel spectra). All in `experiments/`, summarized in [Benchmarks](docs/benchmarks.md), reviewed in `notebooks/06_identification_baseline.ipynb` and `notebooks/07_conditioning.ipynb`.
- The disturbance and stability suite (item 4): K05 (data disturbances vs the linear response), K06 (problem perturbations vs the exact response) and K07 (empirical stability constants); the uncertainty experiments (item 5): F06/K10 (coverage of ensemble, Laplace, Fisher and swarm intervals), K39 (the data weight from the noise model) and K46 (swarm sampling vs the Fisher ellipse), reviewed in `notebooks/08_stability_and_uncertainty.ipynb`; and the sampling follow-ups (item 6): B02, B03, B04 and B08. All summarized in [Benchmarks](docs/benchmarks.md).

## [0.4.0] — 2026-10-24

A correctness release. A physics audit, a code review and a clean-install release gate found bugs that changed results in earlier versions. **Results from 0.3.x on 2-D PDEs, the RL sampler, wave, KdV, Allen–Cahn, Cahn–Hilliard, Black–Scholes and pendulum should be recomputed.**

### Fixed: results change
- **2-D PDEs trained without their spatial operator.** n-D gradients and Laplacians were taken with respect to a column slice, so autograd returned zeros, and `heat_2d` and every 2-D PDE trained with `∇²u ≡ 0`. There is now one `PDEBase.compute_derivatives` with `_grad`/`laplacian` helpers.
- **The RL sampler (`--rl`, `strategy="adaptive"`) never learned.** Its DQN was never trained. In the trainer path the agent was never attached, so `--rl` ran uniform sampling. In the benchmark path every point collapsed to one corner of the domain. It is replaced by `ResidualBanditSampler`, an experimental contextual bandit that is verified to learn.
- **The bandit's replay batches ignored the seed.** They came from Python's unseeded `random`, so a bandit run with a fixed seed differed between processes. They now follow `torch.manual_seed`.
- **Exact solutions corrected and verified** against their PDE, IC and BCs: wave (standing, not travelling), convection (`u0(x − vt)`), KdV (soliton speed; `alpha`/`beta` now used), Allen–Cahn (√2 interface width; `reaction_rate` used), Cahn–Hilliard (`mobility`/`kappa` used; stationary kink), Black–Scholes (time to maturity, the real normal CDF, `strike` read), pendulum (nonlinear damped ODE; `damping` used; `θ'(0)` enforced), Burgers (Cole–Hopf for `A·sin(kπx)`; `viscosity` alias).
- **Boundary and initial conditions.**
  - Neumann now constrains `u_x`; it was a Dirichlet condition in disguise.
  - Periodic now matches `u` and `u_x` across faces; it used to fit `sin(2πx)`.
  - Unknown BC types raise instead of silently becoming zero.
  - The IC is no longer imposed on spatial boundaries.
  - `num_boundary_points`/`num_initial_points` are honoured.
  - Second-order-in-time PDEs enforce the initial velocity.
- **Adaptive weighting `lrw`** stepped with a stale extra gradient (`∇total + ∇L_initial`). It also gave a term with zero gradient (a constant loss, e.g. a PDE without boundary conditions) almost all the weight, which switched the residual off. Such terms now get weight 0.
- **The Well data was mapped wrongly.** The loader read keys the installed `the_well` doesn't produce, so every observation had `t = 0` and unit-square coordinates. It now uses the real time and space grids. The registry is rebuilt from each dataset's files, which corrects domains, `dt`, step counts and channel counts (`rayleigh_taylor_instability` has 4 channels, `viscoelastic_instability` 8). `euler_multi_quadrants_periodicBC` (not on the mirror) is removed, which leaves 11 datasets. A Well run's PDE now spans the time range of the data it loads, since trajectory length differs by file in five datasets. The run's `config.yaml` records that window, so `load_run` rebuilds it without reloading the data.
- **Breaking: the `fourier` network builds one hidden layer per `hidden_dims` entry**, like `feedforward`. It used to build one fewer. Fourier checkpoints from 0.3.x won't load.
- **Early-stopping patience** counted validations instead of epochs, so `patience: 100` meant about 1000 epochs.
- **Cosine schedule with warmup** never reached `min_lr`.
- **`generate_collocation_points`** returns exactly `n` points; 1-D used to round down to a perfect square. Tuple domains such as `(-1, 1)` are honoured.

### Fixed: crashes and installs
- `pip install pinnrl && pinnrl-dashboard` failed (`dash` was only in `[viz]`). `dash` is now a core dependency; `[viz]` is kept as an empty extra.
- `pinnrl-train` failed without `--config` (it looked for `src/config/config.yaml`), and crashed on non-Apple machines because the config hard-coded `device: mps`. Unavailable devices now fall back to CPU.
- `pinnrl-train --pde heat` was rejected: the short keys documented in the README now work alongside display names.
- `--collocation-distribution` was silently ignored.
- YAML numbers like `1e-7` were read as strings, which broke L-BFGS.
- Inverse mode crashed on list-valued parameters (`velocity: [1.0]`).
- RAD crashed on all-zero residuals; the bandit crashed when its policy share was 0 points.
- Exact solutions now hold for time domains not starting at 0.
- Networks crashed when an optional key (`omega_0`, `num_heads`, `latent_dim`, `modes`) was missing; the network's own default now applies.
- Dashboard:
  - the port retry never ran;
  - run titles were garbled (one run-name builder and parser now replaces four parsers);
  - the loss breakdown was truncated and shifted, and the learning rate was drawn as a loss curve;
  - 1-D solution surfaces were empty.

### Changed
- **`validate()`**: `l2_error` is now the relative L2 error `‖u − u*‖/‖u*‖` (it was an MSE), with new `mse`, `max_error` and `mean_error` keys.
- **One training loop:** `PDETrainer(model, pde, config, device=None, bandit_sampler=None)`, with `train(num_epochs=None, batch_size=None, num_points=None, experiment_dir=None, progress=True)` taking its defaults from the config. The `optimizer_config`, `viz_frequency`, `early_stopping_config` and `validation_frequency` constructor arguments are removed (use the config). The always-zero smoothness loss is removed.
- **One PDE registry:** `pinnrl.pdes.registry` (`PDE_SPECS`, `get_pde_spec`, `create_pde(name, config)`).
- **Modules:** plotting moves to `pinnrl.plots`; `pinnrl.utils` and `pinnrl.components` are removed (adaptive weights live in `pinnrl.training.adaptive_weights`). Legacy config classes (`EvaluationConfig`, `LoggingConfig`, `PathsConfig`, the duplicate `PDEConfig`) are removed. `rl_agent.py` is trimmed to the DQN used by the bandit.
- **`pinnrl-benchmark sampling`** is now a multi-seed study (uniform, stratified, uniform every 100 steps, RAD, RL bandit) at equal training steps. It records residual evaluations including sampler overhead and reports paired Wilcoxon tests; `--steps`/`--checkpoints` replace the single-seed `--pde/--epochs` harness.
- **Outputs** go to `./runs/` (was `experiments/`, which is now the tracked research folder). The dashboard gains `--no-browser`, and picks the next free port when the requested one is taken.
- **The Well:**
  - `--dataset` sets the mode to the dataset's recommendation unless `--mode` is given;
  - every registered dataset is `data_only` (none is governed by a shipped PDE);
  - the new `--dataset-steps` / `n_steps` argument (and the dashboard's "Time steps" field) sets how many time steps are read per trajectory: 101 by default, 11 for 3-D datasets;
  - the loader gathers only the sampled points, so a 3-D dataset no longer needs gigabytes of RAM.
- **Training:**
  - every `train()` call re-plans: the learning rate restarts from `training.learning_rate`, and the cosine schedule and `adam_lbfgs` switch epoch follow its `num_epochs`;
  - `history["loss_weights"]` holds one entry per epoch (it was per step, which truncated the loss-weight plot);
  - the L-BFGS phase uses the full batch;
  - adaptive-weight logs are DEBUG;
  - `pinnrl` no longer configures logging or the matplotlib backend at import.
- **Dependencies:** `gymnasium`, `pandas`, `seaborn`, `scikit-image` and `dash-bootstrap-components` are removed (nothing imported them).

### Added
- `PDETrainer.train_steps(num_steps, time_budget_s, sample_fn, resample_every, checkpoints, on_checkpoint)`, the same update as `train` for studies.
- RAD parameters `k`, `c`, `pool_factor` and `replacement` (Wu et al., 2023).
- `pinnrl.benchmarks.reference`: Burgers via Cole–Hopf, which reproduces Basdevant et al.'s (1986) `u_x(0, 1.6037/π) = −152.00516` for `ν = 0.01/π`, and Allen–Cahn via method of lines; Allen–Cahn gains the `x2_cos` initial condition.
- `pinnrl.training.train.build_model` and `load_run`; linear LR warmup.
- `tests/unit_tests/test_physics_consistency.py` (about 110 checks) and `scripts/verify_release.sh` (clean-wheel release gate).
- Documentation rewritten: concepts, equations, guides, applications (digital twins), research program, roadmap.

---

## [0.3.0] — 2026-05-03

> Note: the `v0.3.1` tag and GitHub release built this same 0.3.0 package; there is no 0.3.1 on PyPI. The RL-sampling and RAR accuracy claims below were not supported by the code (see 0.4.0).

First post-launch release. Bundles the v0.2 (docs + onboarding) and v0.3 (science features) milestones into a single PyPI cut, since v0.1.0 was the only previous public release.

### Added
- **Inverse problems** — `--mode inverse --identify alpha` jointly trains the network and any PDE parameter marked as trainable; synthetic noisy observations are generated from the analytical solution when no real data is supplied (`pinnrl/training/train.py`, `pinnrl/pdes/pde_base.py`).
- **Fourier Neural Operator architecture** (`fno`) — spectral convolutions adapted for point-wise PINN inputs (`pinnrl/neural_networks/fno.py`).
- **Residual-Adaptive Refinement (RAR) sampling** — `strategy="residual_based"` resamples a candidate pool by current PDE residual every step, dramatically improving convergence on shock problems (`pinnrl/pdes/pde_base.py`).
- **L-BFGS optimizer** and `adam_lbfgs` two-phase optimizer for late-stage convergence (`pinnrl/training/trainer.py`).
- **Configurable loss functions** — `loss_function: mse | mae | huber` per training run, with `huber_delta` knob.
- **Benchmark suite** — `pinnrl-benchmark fdm --pde heat|wave` runs a NumPy FDM baseline; `pinnrl-benchmark sampling --pde ... --strategies uniform stratified residual_based adaptive` trains a small PINN under each collocation strategy and reports L2/max/wall-time. Both subcommands accept `--csv` for downstream comparison (`pinnrl/benchmarks/`).
- **The Well dataset integration** — `pinnrl[well]` extra adds `WellDataset` streaming via Hugging Face plus a 12-entry registry (`active_matter`, `rayleigh_benard`, `MHD_64`, `gray_scott_reaction_diffusion`, …) with auto-filled PDE/architecture defaults. `data_only` and `data_augmented` training modes plus dataset-aware caching live in `pinnrl/datasets/`.
- **2D heat equation** — full `heat_2d` training validated end-to-end with FNO.
- **Dashboard upgrades** — live 3D solution and residual surfaces, dataset selector, parameter-identification panel, experiment comparison view.
- **`pinnrl-dashboard` console script** — replaces `python pinnrl/main.py`. Defaults to port 8050.
- **Notebooks** — `03_rl_vs_uniform_sampling.ipynb`, `04_well_dataset_quickstart.ipynb`, `05_inverse_problem.ipynb` (parameter identification on the heat equation).
- **MkDocs Material site** with API reference auto-generated via `mkdocstrings`, deployed via `.github/workflows/docs.yml`.

### Changed
- **Package name `src/` → `pinnrl/`.** All imports are now `from pinnrl.*`. Wheel built via hatchling with `packages = ["pinnrl"]`; `pinnrl/numerical_solvers/` excluded from the published artifact.
- `pinnrl-train` is the headless training CLI (`pinnrl.training.train:main`).
- `pinnrl-benchmark` is the new benchmark CLI (`pinnrl.benchmarks.cli:main`).
- README PyPI badge now reflects the actual published version.

### Fixed
- **Burgers residual** — `BurgersEquation.compute_residual` was reading derivatives under `dx1`/`d2x1`, but `compute_derivatives` stores 1-D derivatives as `dx` and exposes the Laplacian as `laplacian`. Both `.get()` calls fell through to zeros, so the convection (`u·u_x`) and diffusion (`ν·u_xx`) terms were silently zero — every Burgers PINN was unknowingly minimizing `u_t = 0` rather than the actual equation. Fixed in `pinnrl/pdes/burgers_equation.py`.
- **Active-matter dataset registry** — `output_dim` corrected from 5 to 11 to match the loader's actual field shape; `live_snapshot.npz` now collapses multi-channel outputs to channel 0 instead of crashing the visualization (`pinnrl/datasets/registry.py`, `pinnrl/training/trainer.py`).
- **Well cache key** — `_cache_file` now includes a hash of the resolved `base` path, so HF streaming and local mirrors no longer collide on the same `.npz` (`pinnrl/datasets/well_loader.py`).
- **`--mode` preservation** — `_apply_well_dataset_defaults` no longer overwrites a user-supplied `--mode inverse` with the dataset's `recommended_mode`; explicit user intent wins (`pinnrl/training/train.py`).

### Known issues
- `compute_loss` retains an autograd graph across iterations when used with `trainable_parameters` and most boundary-condition setups, blocking `pinnrl-train --mode inverse` for non-trivial setups. The `05_inverse_problem.ipynb` notebook sidesteps this by writing the loss inline. Tracked for v0.3.1.

---

## [0.2.0]

Skipped — milestone work (docs, MkDocs site, tutorials, PyPI launch post) shipped together with v0.3 in the 0.3.0 release.

---

## Pre-0.3 fixes (0.1.x)

### Fixed
- **Bug #1** — Architecture-specific config (`mapping_size`, `scale`, `omega_0`, `num_heads`, `latent_dim`) now correctly injected into `ModelConfig` from `config.yaml`
- **Bug #2** — Learning rate now correctly read from `training.optimizer_config.learning_rate` (was silently defaulting to 0.001 instead of 0.005)
- **Bug #3** — `InteractiveTrainer.create_pde()` now supports all 9 PDEs (KdV, Convection, Allen-Cahn, Cahn-Hilliard, Black-Scholes were missing)
- **Bug #4** — `update_pde_params()` now uses `pde_name_to_key` for correct config lookup
- **Bug #5** — `create_config()` pde_key extraction now handles multi-word PDE names correctly
- **Bug #8** — Removed dead `create_network()` function referencing non-existent `SirenNetwork`
- **Bug #9** — `FeedForwardNetwork` now creates a new activation module instance per layer (was sharing one instance — invalid in PyTorch)
- **Bug #10** — `requirements.txt`: replaced obsolete `gym>=0.21.0` with `gymnasium>=0.26.0`
- **Bug #12** — `config.yaml` loss weights key renamed `pde` → `residual`; `Config._load_config()` normalizes on load
- **Bug #13** — All architecture dropout defaults set to `0.0` in `config.yaml` (dropout conflicts with PINN residual convergence)
- **Bug #14** — `AutoEncoder` restructured for PINN use: decoder now maps `latent_dim → output_dim` instead of `latent_dim → input_dim`
- `base_network.py` — `torch.load(..., weights_only=False)` for PyTorch ≥ 2.6 compatibility

### Changed
- `.gitignore` — expanded to cover `uv.lock` artifacts, notebook output images, generated docs, and model weight extensions
- `config.yaml` — removed from `.gitignore` (should be tracked as it is the primary config)

---

## [0.0.1] — 2025-04-01

Initial private research prototype.

- PINNs for Heat, Wave, Burgers, Pendulum equations
- FeedForward, ResNet, SIREN, Fourier, Attention, Autoencoder architectures
- Tkinter interactive trainer GUI
- Dash training dashboard
- DQN-based RL collocation agent (experimental)
