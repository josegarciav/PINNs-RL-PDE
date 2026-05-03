# Changelog

All notable changes to `pinnrl` are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.3.0] — 2026-05-03

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
