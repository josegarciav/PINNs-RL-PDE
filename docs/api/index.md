# API Reference

Reference documentation for the `pinnrl` package, generated from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/).

| Module | Description |
|--------|-------------|
| [Configuration](config.md) | Dataclasses built from `config.yaml` (`Config`, `TrainingConfig`, `ModelConfig`, …) |
| [Neural Networks](neural_networks.md) | The `PINNModel` factory and the 7 architecture backends |
| [PDEs](pdes.md) | `CustomPDE`, the catalogue, sensors, inputs and unknown functions; the PDE registry, `PDEBase` and the shipped implementations |
| [Training](training.md) | `PDETrainer` (`train`, `train_steps`), restarts, adaptive loss weights, `build_model`, `load_run`, runs from a configuration |
| [Native engine](native.md) | The compiled kernel, the step compiled per equation, the compiled evaporator solver |
| [RL sampler](rl.md) | The experimental residual-bandit collocation sampler and its DQN |
| [Plots](plots.md) | Solution, training-history and collocation plots |
| [Benchmarks](benchmarks.md) | Numerical reference solutions and the equal-compute sampling study |
| [Analysis](analysis.md) | Sensitivities, Fisher information and the Cramér–Rao bound for identified parameters; Hessian and tangent-kernel spectra; disturbance tools |
| [Uncertainty](uq.md) | Laplace posteriors, swarm sampling of the region of equivalence, ensembles and calibration metrics |
| [Process models](process.md) | The evaporator, the droplet, the spray dryers and the line; the Kalman filter, the moving-horizon estimator, the fouling monitor, the historian adapter |
| [Datasets](datasets.md) | The registry and loaders of The Well |
| [Command-line entry points](cli.md) | The functions behind `pinnrl-train`, `pinnrl-benchmark` and `pinnrl-dashboard` |

A test (`tests/unit_tests/test_docs_api_coverage.py`) fails when a public name of the package has no entry on these pages.
