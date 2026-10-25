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

- **Your own equations:** `CustomPDE("u_t + u*u_x - nu*u_xx", domain={"x": (-1, 1), "t": (0, 1)}, parameters={"nu": 0.01})` turns a residual written as text (or a Python function) into a trainable problem: PDEs, ODEs, steady problems and systems in any number of dimensions, with identifiable parameters. A catalogue of 42 ready equations comes with it.
- **10 shipped PDE configurations:** heat (1-D, 2-D), wave, Burgers, convection, KdV, Allen–Cahn, Cahn–Hilliard, Black–Scholes and a damped pendulum. Each exact solution is checked by the test suite against its PDE, initial condition and boundary conditions.
- **7 architectures:** MLP, ResNet, Fourier features, SIREN, FNO-style, attention and autoencoder.
- **Collocation sampling:** uniform, stratified, residual-based adaptive (RAD), and an experimental reinforcement-learning bandit.
- **Inverse problems:** identify PDE parameters from noisy observations.
- **The Well:** train on 11 high-fidelity simulation datasets.
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
pinnrl-train --pde-file my_equation.yaml --arch fourier                           # your own equation, from YAML
```

```python
from pinnrl import CustomPDE

pde = CustomPDE("u_t - alpha*u_xx", domain={"x": (0, 1), "t": (0, 1)}, parameters={"alpha": 0.1},
                initial_condition="sin(pi*x)", boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)")
trainer = pde.build_trainer("fourier", num_epochs=150, num_collocation_points=1024, batch_size=1024,
                            learning_rate=2e-3, early_stopping={"enabled": False})
trainer.train(progress=False)
print(pde.validate(trainer.model)["l2_error"])
```

The Python API is the same pieces the CLI uses: a PDE (`CustomPDE`, or a `PDEConfig` with a shipped class), a `PINNModel` and `PDETrainer`. See [Custom equations](https://josegarciav.github.io/PINNs-RL-PDE/custom_equations/) and [Getting started](https://josegarciav.github.io/PINNs-RL-PDE/getting_started/).

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
