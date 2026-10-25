# Training

## Command line

```bash
pinnrl-train --pde <key> --arch <architecture> [options]
```

Unset options come from the packaged `config.yaml`: its `training` block, the chosen `architectures` entry, and the PDE's block in `pde_configs`. Pass `--config my.yaml` to use your own.

| Group | Flags |
|---|---|
| Problem | `--pde` (key or display name) or `--pde-file` (your own equation as YAML, see [Custom equations](custom_equations.md)), `--arch` (`feedforward`, `resnet`, `fourier`, `siren`, `fno`, `attention`, `autoencoder`), `--config`, `--device` (`cpu`, `mps`, `cuda`; falls back to CPU) |
| Budget | `--epochs`, `--batch-size`, `--collocation-points`, `--boundary-points`, `--initial-points`, `--lr` |
| Sampling | `--collocation-distribution` (`uniform`, `stratified`, `residual_based`); `--rl` for the experimental bandit sampler (1-D) |
| Optimizer | `--optimizer` (`adam`, `lbfgs`, `adam_lbfgs`), `--lr-scheduler` (`cosine`, `reduce_lr`), `--weight-decay`, `--gradient-clipping`, `--early-stopping` (`on`, `off`) |
| Loss | `--adaptive-weights` (`off`, `lrw`, `rbw`), `--loss-function` (`mse`, `mae`, `huber`), `--huber-delta` |
| Mode | `--mode` (`forward`, `inverse`, `data_only`, `data_augmented`) |
| Inverse | `--identify NAME` (repeatable), `--initial-guess NAME=VALUE`, `--obs-path file.npz`, `--obs-points`, `--obs-noise` |
| Datasets | `--dataset NAME`, `--dataset-split`, `--dataset-traj`, `--dataset-points`, `--dataset-seed`, `--dataset-base` |

Examples:

```bash
# Burgers with residual-based adaptive sampling and Adam followed by L-BFGS
pinnrl-train --pde burgers --arch fourier --epochs 3000 \
  --collocation-distribution residual_based --optimizer adam_lbfgs

# 2-D heat with gradient-balanced loss weights
pinnrl-train --pde heat_2d --arch fourier --adaptive-weights lrw
```

Each run is saved in `./runs/<timestamp>_<PDE display name>_<arch>_<rl|no_rl>/`, with the dataset name in place of the PDE for a Well run (see [Getting started](getting_started.md#first-run-from-the-command-line)). Early-stopping `patience` is counted in epochs; validation runs every `validation_frequency` epochs.

## Python

`PDETrainer` is the only training loop. The CLI, the dashboard, the notebooks and the benchmarks all use it.

<!-- doctest: skip -->
```python
from pinnrl.training import PDETrainer

trainer = PDETrainer(model, pde, config=config, device=device)

# Epoch-based, with validation, early stopping, the LR schedule and run files:
history = trainer.train(num_epochs=2000, experiment_dir="runs/my_run")

# Step-based, for studies: fixed steps or a time budget, no I/O
out = trainer.train_steps(num_steps=5000, resample_every=100)
print(out["steps"], out["train_time_s"], out["loss"][-1])
```

Each `train()` call plans its own schedule: the learning rate restarts from `training.learning_rate`, and the cosine period and `adam_lbfgs` switch epoch follow that call's `num_epochs`. `trainer.history` holds per-epoch entries, including `loss_weights` when adaptive weighting is on.

`train_steps` performs exactly the same update as `train`. It accepts `sample_fn(model, progress) -> (x, t)` to plug in any sampler, and `checkpoints`/`on_checkpoint` to evaluate along the way.

To reload a finished CLI or dashboard run:

<!-- doctest: skip -->
```python
from pinnrl.training.train import load_run

pde, model, config = load_run("runs/20261023_120000_Burgers Equation_fourier_no_rl")
print(pde.validate(model)["l2_error"])
```

## Choosing settings

- Start with `feedforward` or `fourier`, Adam at 1e-3, and uniform sampling. Change one thing at a time.
- Badly scaled loss terms are the most common silent failure: the boundary or initial loss stalls while the residual drops. Try `--adaptive-weights lrw`, or raise `loss_weights.boundary` and `loss_weights.initial`.
- `adam_lbfgs` usually improves the last digit on smooth problems. It rarely rescues a run that Adam got wrong.
- For problems with sharp features, compare `residual_based` against `uniform` at the same budget (see [Benchmarks](benchmarks.md)); don't assume it helps.
