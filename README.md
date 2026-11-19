<p align="center">
  <img src="https://raw.githubusercontent.com/josegarciav/PINNs-RL-PDE/main/docs/assets/pinnrl-logo.png" alt="pinnrl" width="420"/>
</p>

# pinnrl

[![PyPI](https://img.shields.io/pypi/v/pinnrl.svg)](https://pypi.org/project/pinnrl/)
[![Checks](https://github.com/josegarciav/PINNs-RL-PDE/actions/workflows/checks.yml/badge.svg)](https://github.com/josegarciav/PINNs-RL-PDE/actions/workflows/checks.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/LICENSE)

**Physics-informed neural networks for forward and inverse problems, with a training step compiled per equation, tools that say when an estimate can be trusted, and classical baselines to grade every result against.**

A physics-informed neural network (PINN) represents the solution of a differential equation as a neural network, trained to satisfy the equation, its boundary and initial conditions, and any measured data. `pinnrl` is a package for using that method with open eyes: it makes the training fast, it measures what the data can and cannot determine, and it ships the classical method for each problem so you can see which one wins.

## Goals

1. **A training step cheap enough to study the method properly.** You write the equation as text; `pinnrl` generates and compiles C++ for it. On nine benchmark equations it is the fastest step we measured on a CPU.
2. **Inverse problems with an honest answer.** Next to an estimate you get the bound the data allow, the floor the network adds, a calibrated interval, and a misfit test against a reference solver.
3. **Every result against its classical baseline.** Finite differences, reference solvers, a Kalman filter, moving-horizon estimation and least squares through a solver are in the package and are used as controls.
4. **Digital twins of process equipment** as the application: estimating what a plant does not measure (fouling, a drying-rate curve, the stickiness margin of a powder) from the sensors it has. The first units are evaporators and spray dryers.

## Who it is for

- **Researchers** studying PINN training, sampling or identifiability, who need a fast trainer, reference solutions and multi-seed protocols.
- **Process and control engineers** who want to know what their records can determine before buying a sensor or running a campaign.
- **Teachers and students**: every exact solution in the package is checked by the test suite against its equation, initial condition and boundary conditions.

## When to use a PINN, and when not to

The package's own experiments give a narrower answer than much of the literature. Numbers are from the [benchmarks page](https://josegarciav.github.io/PINNs-RL-PDE/benchmarks/).

**Do not use a PINN when**

- **the forward problem is well posed and a classical solver exists.** Explicit finite differences solve the 1-D heat equation to a relative error of about 7e-4 in about 30 ms; a PINN reaches 2e-4 in 120 s.
- **the model has a cheap, trusted forward solve and you want its states or parameters.** On the evaporator an 11-cell Kalman filter tracks the fouling to 3.5 % in 0.2 s; the PINN twin reaches 9–16 % in five minutes and fails to train on 22 of 36 repeated runs. On the drying droplet, shooting through the reference solver reaches 1.5–1.6 % on every seed in 8–10 s. On the counter-current dryer, least squares recovers three parameters in 10 s and every PINN twin run is rejected by the misfit test.

**Consider a PINN when**

- **there is no trusted forward solver** for the model, or it is too expensive to iterate;
- **an unknown function sits inside the equation** (a coefficient that drifts, a closure), and you can constrain it to stay physical;
- **one network should replace many solves**: a parametric surrogate covers a parameter range to 0.2–2 % at a quarter to a half of the cost of five dedicated networks, and then answers in milliseconds;
- **physics has to be combined with sparse data**, or the problem is mesh-free or high-dimensional.

In every case, run the baseline. The package is built so that you can.

## Why this package

**The fastest PINN step we measured on a CPU.** The same training step (a 4 × 64 `tanh` network, 5,120 points, Adam) in each framework, on nine equations from second to fourth order, one thread on an Apple M4, milliseconds per step ([experiment I17](https://github.com/josegarciav/PINNs-RL-PDE/tree/main/experiments/I17_step_head_to_head)):

| | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`** | **3.7** | **3.7** | **3.7** | **4.4** | **5.5** | **5.7** | **5.7** | **3.8** | **5.1** |
| JAX, `jit`, its best mode | 4.3 | 4.2 | 4.2 | 6.1 | 6.6 | 10.0 | 10.1 | 4.4 | 6.6 |
| PyTorch, textbook loop | 8.7 | 8.8 | 8.8 | 17.7 | 17.7 | 43.6 | 43.6 | 17.7 | 12.0 |
| DeepXDE (PyTorch backend) | 10.1 | 10.0 | 10.0 | 21.5 | 21.9 | 56.0 | 55.7 | 18.8 | 14.5 |
| NVIDIA PhysicsNeMo (`PhysicsInformer`) | 10.9 | 10.7 | 10.7 | n/s | n/s | n/s | n/s | 17.7 | 14.3 |

`pinnrl` is first on each on an Apple M4: 1.1–1.8× JAX's best differentiation mode, 2.4–7.7× eager PyTorch, 2.7–9.9× DeepXDE, and about 3× PhysicsNeMo, whose `PhysicsInformer` stops at second order (n/s: not supported). With layer normalization it is 1.2–1.5× JAX. The step is the package's own engine: Taylor-mode derivatives carried forward through the network, fused per equation into generated C++, compiled once and cached. It is the default, and it covers inverse problems (trainable parameters, unknown functions of the coordinates) and scaled networks. The limits are stated with the result: on a Linux x86 runner JAX's forward mode is 1–11 % ahead on seven of the nine plain-network problems, and `pinnrl` leads there at fourth order and with layer normalization; one thread, one network size; the cost of a step, not the error reached in a given time.

**An inverse toolbox that says when to trust an estimate.** Fisher information and Cramér–Rao bounds computed from a trained network; sensor-layout and campaign design before any measurement is bought; stability constants; Laplace, ensemble and swarm intervals whose coverage has been measured; and a misfit test that runs the fitted model through a reference solver. One of its findings: the 2 % error floor of early inverse runs was the network's weight decay and gradient clipping, not the method (0.1–0.2 % without them).

**Process twins with their classical controls.** A falling-film evaporator, a drying droplet, co-current and counter-current spray dryers and a line that chains them, each with a reference solver, a `CustomPDE` form, and the estimators a PINN has to beat: an extended Kalman filter with a smoother, moving-horizon estimation, least squares through the solver.

**Negative results reported.** The reinforcement-learning collocation sampler does not beat uniform sampling at equal cost. On every unit with a cheap forward solve, the classical estimator beat the PINN twin. Both are in the docs with their numbers.

## How it compares

| | Best at | Choose it over `pinnrl` when |
|---|---|---|
| [DeepXDE](https://github.com/lululxvi/deepxde) | The widest coverage: several backends, complex geometries, operator learning, many worked examples | You need a geometry that is not a box, a backend other than PyTorch, or DeepONets |
| [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (formerly Modulus) | Industrial scale on GPUs: 3-D parameterised geometries, neural operators, multi-GPU training | The problem is large, three-dimensional and you have accelerators |
| JAX-based PINN code | A whole step compiled by XLA, forward and Taylor mode, GPUs, any architecture you can write | You need a custom architecture or a GPU and are willing to write the training loop |
| [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) | Symbolic problem definitions inside Julia's SciML ecosystem, with its stiff solvers | You work in Julia |
| **`pinnrl`** | The fastest step on a CPU for small `tanh` networks; identifiability, calibration and misfit tools for inverse problems; process units with classical estimators | You want to know whether a PINN is the right tool for your inverse problem, and how far to trust its answer |

`pinnrl` is narrower than the first two: box domains, small networks, PyTorch on a CPU (it runs on Apple and CUDA GPUs through autograd, without the compiled step).

## Install

```bash
pip install pinnrl              # Python 3.10–3.13; CPU, Apple GPU (mps) or CUDA
pip install 'pinnrl[well]'      # + The Well dataset loaders
```

The compiled step needs a C++ compiler at first use (the Xcode command line tools on macOS; `g++` and `libopenblas-dev` on Linux). Without one, everything runs on autograd.

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

## What is in the package

| Part | What it gives |
|---|---|
| `pinnrl.CustomPDE` | Equations as text or as a Python function: PDEs, ODEs, steady problems and systems in any number of dimensions, trainable parameters, unknown functions, sensors. A catalogue of 43 ready equations |
| `pinnrl.pdes` | 10 shipped configurations with checked exact solutions: heat (1-D, 2-D), wave, Burgers, convection, KdV, Allen–Cahn, Cahn–Hilliard, Black–Scholes, a damped pendulum |
| `pinnrl.neural_networks` | 7 architectures: MLP, ResNet, Fourier features, SIREN, FNO-style, attention, autoencoder |
| `pinnrl.training` | Adam, L-BFGS and their sequence; fixed and adaptive loss weights; time-marching windows; restarts judged by a misfit; uniform, stratified, residual-based adaptive (RAD) and experimental RL-bandit sampling |
| `pinnrl.native` | The compiled engine: the general Taylor-mode kernel, the step compiled per equation, a compiled evaporator solver |
| `pinnrl.analysis` | Sensitivities, Fisher information, measurement design, Hessian and tangent-kernel spectra, disturbance response, stability ratios |
| `pinnrl.uq` | Laplace posterior, coverage metrics, swarm sampling of equivalence regions |
| `pinnrl.process` | Evaporator, drying droplet, spray dryers, the line twin; Kalman filter, moving-horizon estimator, fouling monitor, historian adapter |
| `pinnrl.benchmarks` | Finite-difference baselines, reference solutions, the multi-seed sampling study |
| `pinnrl.datasets` | Loaders for 11 simulation datasets of The Well |
| `pinnrl-dashboard` | Launch, monitor and compare runs |

## Reproduce the benchmarks

```bash
git clone https://github.com/josegarciav/PINNs-RL-PDE && cd PINNs-RL-PDE && uv sync --all-extras
uv run --with jax --with deepxde python experiments/I17_step_head_to_head/run.py   # the table above, about 45 min
uv run pinnrl-benchmark fdm --pde heat                                             # the classical baseline
uv run pinnrl-benchmark sampling --seeds 10 --steps 15000 --checkpoints 50 --workers 4
```

Every number in the docs comes from a script under [`experiments/`](https://github.com/josegarciav/PINNs-RL-PDE/tree/main/experiments), each with a README stating the question, the protocol and the result. The [benchmarks page](https://josegarciav.github.io/PINNs-RL-PDE/benchmarks/) collects them.

## Documentation

**[josegarciav.github.io/PINNs-RL-PDE](https://josegarciav.github.io/PINNs-RL-PDE/)**: concepts, the equations, the theory behind the inverse tools, guides ([training](https://josegarciav.github.io/PINNs-RL-PDE/training/), [inverse problems](https://josegarciav.github.io/PINNs-RL-PDE/inverse_problems/), [process twins](https://josegarciav.github.io/PINNs-RL-PDE/process_twins/)), benchmarks and the API reference. The tutorials are in [`notebooks/`](https://github.com/josegarciav/PINNs-RL-PDE/tree/main/notebooks). A [software paper](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/paper/paper.md) describes the design and collects the benchmarks.

## Roadmap

v0.9 is in progress: the compiled step now covers the inverse problems and layer normalization; next are research bets on accuracy per second (an optimizer, an architecture and a training controller built for physics-informed losses), tests that protect the architecture, and the dashboard as an experiment tracker. Then process units as reinforcement-learning environments with PID and model-predictive-control baselines (v0.10), decisions that serve the inverse problem (v0.11), discovering unknown physics (v0.12), scale (v0.13) and a stable API (v1.0). Work on a real plant is v7.0, after the platform has passed a rehearsal against a simulated one. The full plan, with what each step is expected to gain, is on the [roadmap page](https://josegarciav.github.io/PINNs-RL-PDE/roadmap/).

## Contributing

See [CONTRIBUTING.md](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/CONTRIBUTING.md). In short: `uv sync --all-extras`, `uv run pytest`, conventional commits, and physics changes must keep `tests/unit_tests/test_physics_consistency.py` green. A claim that something improves accuracy or speed needs an experiment with paired seeds, equal budgets and its baseline. Bug reports, new equations, new units and benchmarks on other hardware are welcome; for anything large, open an issue first.

## Citation

`pinnrl` started as the MSc thesis *On Solving Partial Differential Equations (PDEs) Using Physics-Informed Neural Networks (PINNs)* (University of the Basque Country UPV/EHU, 2025); the thesis is at [docs/assets/thesis.pdf](https://josegarciav.github.io/PINNs-RL-PDE/assets/thesis.pdf).

```bibtex
@software{garcia_pinnrl,
  author = {Garcia Ventura, Jose},
  title  = {pinnrl: physics-informed neural networks with a compiled training step, an inverse-modelling toolbox and classical baselines},
  url    = {https://github.com/josegarciav/PINNs-RL-PDE}
}
```

MIT License © Jose Garcia Ventura
