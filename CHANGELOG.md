# Changelog

All notable changes to `pinnrl` are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `scripts/train.py` — CLI training script with full `config.yaml` override support
- `pyproject.toml` — `uv`-compatible packaging, replaces bare `requirements.txt`
- `uv.lock` — reproducible dependency lock file
- `docs/` — initial documentation: start_here, setup, architecture overview, roadmap
- `notebooks/01_your_first_pinn.ipynb` — beginner tutorial: heat equation
- `notebooks/02_comparing_architectures.ipynb` — architecture comparison on Burgers
- `.github/workflows/tests.yml` — GitHub Actions CI (lint + pytest, Python 3.10–3.12)
- `CONTRIBUTING.md` — guide for adding PDEs, architectures, and submitting PRs
- `mkdocs.yml` — MkDocs Material theme documentation site configuration

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
