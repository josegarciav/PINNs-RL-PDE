# API Reference

Reference documentation for the `pinnrl` package, generated from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/).

| Module | Description |
|--------|-------------|
| [Configuration](config.md) | Dataclasses built from `config.yaml` (`Config`, `TrainingConfig`, `ModelConfig`, …) |
| [Neural Networks](neural_networks.md) | The `PINNModel` factory and the 7 architecture backends |
| [PDEs](pdes.md) | The PDE registry, `PDEBase` and the 9 PDE implementations |
| [Training](training.md) | `PDETrainer` (`train`, `train_steps`), adaptive loss weights, `build_model`, `load_run` |
| [RL sampler](rl.md) | The experimental residual-bandit collocation sampler and its DQN |
| [Plots](plots.md) | Solution, training-history and collocation plots |
| [Benchmarks](benchmarks.md) | Numerical reference solutions and the equal-compute sampling study |
| [Analysis](analysis.md) | Sensitivities, Fisher information and the Cramér–Rao bound for identified parameters; Hessian and tangent-kernel spectra; disturbance tools |
| [Uncertainty](uq.md) | Laplace posteriors, swarm sampling of the region of equivalence, ensembles and calibration metrics |
