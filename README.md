<p align="center">
  <img src="https://raw.githubusercontent.com/josegarciav/PINNs-RL-PDE/main/docs/assets/pinnrl-logo.png" alt="pinnrl" width="420"/>
</p>

# pinnrl

[![PyPI](https://img.shields.io/pypi/v/pinnrl.svg)](https://pypi.org/project/pinnrl/)
[![Checks](https://github.com/josegarciav/PINNs-RL-PDE/actions/workflows/checks.yml/badge.svg)](https://github.com/josegarciav/PINNs-RL-PDE/actions/workflows/checks.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Physics-informed neural networks for forward and inverse PDE problems, evaluated against numerical references.**

A PINN represents the solution of a PDE as a neural network, trained to satisfy the equation, its boundary and initial conditions, and any measured data. `pinnrl` provides:

- **10 PDE configurations:** heat (1-D, 2-D), wave, Burgers, convection, KdV, Allen–Cahn, Cahn–Hilliard, Black–Scholes and a damped pendulum. Each exact solution is checked by the test suite against its PDE, initial condition and boundary conditions.
- **7 architectures:** MLP, ResNet, Fourier features, SIREN, FNO-style, attention and autoencoder.
- **Collocation sampling:** uniform, stratified, residual-based adaptive (RAD), and an experimental reinforcement-learning bandit.
- **Inverse problems:** identify PDE parameters from noisy observations.
- **The Well:** train on 12 high-fidelity simulation datasets.
- **A dashboard** to launch, monitor and compare runs, and **benchmarks** with reference solutions and a multi-seed sampling study.

## Install

```bash
pip install pinnrl              # Python 3.10–3.13; CPU, Apple GPU (mps) or CUDA
pip install 'pinnrl[well]'      # + The Well dataset loaders
```

## Use

```bash
pinnrl-train --pde burgers --arch fourier --epochs 2000                          # results in ./runs/
pinnrl-train --pde heat --arch fourier --mode inverse --identify alpha --initial-guess alpha=0.5
pinnrl-dashboard                                                                 # http://127.0.0.1:8050/
pinnrl-benchmark sampling --seeds 10                                             # uniform, stratified, RAD, RL bandit
```

The Python API is the same pieces: a `PDEConfig` and PDE class, a `PINNModel`, and `PDETrainer`. See [Getting started](https://josegarciav.github.io/PINNs-RL-PDE/getting_started/).

## When a PINN is the right tool

For a well-posed forward problem in low dimension, classical solvers are faster and far more accurate; the docs show the numbers. PINNs pay off for **inverse problems**, for **combining physics with sparse data**, and for **mesh-free or high-dimensional** settings. The project's direction is the first of these applied to industry: **digital twins** that estimate unmeasured states of process equipment (first target: evaporators and spray dryers in food-powder plants). See [Applications](https://josegarciav.github.io/PINNs-RL-PDE/applications/) and the [Research program](https://josegarciav.github.io/PINNs-RL-PDE/research/).

## Documentation

**[josegarciav.github.io/PINNs-RL-PDE](https://josegarciav.github.io/PINNs-RL-PDE/)**: concepts, the equations, guides, benchmarks, the API reference and the roadmap. The tutorials are in [`notebooks/`](notebooks/), and research studies in [`experiments/`](experiments/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `uv sync --all-extras`, `uv run pytest`, conventional commits, and physics changes must keep `tests/unit_tests/test_physics_consistency.py` green.

## Citation

`pinnrl` started as the MSc thesis *On Solving Partial Differential Equations (PDEs) Using Physics-Informed Neural Networks (PINNs)* (University of the Basque Country UPV/EHU, 2025); the thesis is at [docs/assets/thesis.pdf](docs/assets/thesis.pdf).

```bibtex
@software{garcia_pinnrl,
  author = {Garcia Ventura, Jose},
  title  = {pinnrl: physics-informed neural networks with adaptive collocation sampling},
  url    = {https://github.com/josegarciav/PINNs-RL-PDE}
}
```

MIT License © Jose Garcia Ventura
