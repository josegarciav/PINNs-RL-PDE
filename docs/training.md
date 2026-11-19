# Training

This page is for anyone who trains a network with `pinnrl`, from the command line or from Python. It states the loss, lists every option with its packaged default, and gives the workflows as steps.

$$
\mathcal{L}(\theta) = w_r\,\mathcal{L}_r + w_b\,\mathcal{L}_b + w_i\,\mathcal{L}_i + w_d\,\mathcal{L}_d,
\qquad
\mathcal{L}_r = \frac{1}{N_r}\sum_{j=1}^{N_r} \ell\big(\mathcal{N}[u_\theta](x_j, t_j)\big)
$$

- **$\mathcal{N}[u_\theta]$**: the residual of the equation $\mathcal{N}[u] = 0$ on the network $u_\theta$.
- **Collocation point**: a point $(x_j, t_j)$ inside the domain where the residual is evaluated. A batch holds $N_r$ of them.
- **$\mathcal{L}_b$, $\mathcal{L}_i$, $\mathcal{L}_d$**: the same mean for the boundary-condition error, the initial-condition error and the misfit $u_\theta - u_\text{obs}$ to the observations. $\mathcal{L}_d$ is zero when none are attached.
- **$\ell$**: the error's square (`mse`, the default), its absolute value (`mae`) or the Huber function (`huber`).
- **$w_r, w_b, w_i, w_d$**: the loss weights `residual`, `boundary`, `initial` and `data`.

The mode selects which terms count. The physics terms are $\mathcal{L}_r$, $\mathcal{L}_b$ and $\mathcal{L}_i$.

| `mode` | Terms in the loss | Trained |
|---|---|---|
| `forward` (default) | physics terms, plus data if observations are attached | $\theta$ |
| `inverse` | physics terms and data | $\theta$ and the selected parameters of the equation (see [Inverse problems](inverse_problems.md)) |
| `data_augmented` | physics terms and data | $\theta$ |
| `data_only` | data only | $\theta$ |

Outside `forward`, a data weight of zero or less is replaced by 1.

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
| Engine | `--derivatives` (`auto`, `autograd`; `native` is an old alias of `auto`), `--normalize` |
| Inverse | `--identify NAME` (repeatable), `--initial-guess NAME=VALUE`, `--obs-path file.npz`, `--obs-points`, `--obs-noise` |
| Datasets | `--dataset NAME`, `--dataset-split`, `--dataset-traj`, `--dataset-points`, `--dataset-steps`, `--dataset-seed`, `--dataset-base` |

Examples:

```bash
# Burgers with residual-based adaptive sampling and Adam followed by L-BFGS
pinnrl-train --pde burgers --arch fourier --epochs 3000 \
  --collocation-distribution residual_based --optimizer adam_lbfgs

# 2-D heat with gradient-balanced loss weights
pinnrl-train --pde heat_2d --arch fourier --adaptive-weights lrw
```

- Each run is saved in `./runs/<timestamp>_<PDE display name>_<arch>_<rl|no_rl>/`.
- A Well run has the dataset name in place of the PDE (see [Getting started](getting_started.md#first-run-from-the-command-line)).
- Early-stopping `patience` is counted in epochs. Validation runs every `validation_frequency` epochs.

### Options and defaults

Defaults are the values of the `training` block of the packaged `config.yaml`. The full key list is in the [configuration reference](reference/config.md).

**Budget**

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `num_epochs` | `--epochs` | 3000 | Epochs |
| `batch_size` | `--batch-size` | 2048 | Collocation points per optimizer step |
| `num_collocation_points` | `--collocation-points` | 5000 | Collocation points per epoch |
| `num_boundary_points` | `--boundary-points` | 5000 | Points per boundary face |
| `num_initial_points` | `--initial-points` | 5000 | Points on the initial condition |

- **Epoch**: `max(1, num_collocation_points // batch_size)` optimizer steps on freshly sampled batches. With `lbfgs` it is one full batch.

**Optimizer**

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `optimizer` | `--optimizer` | `adam` | `adam`, `lbfgs`, or `adam_lbfgs` (Adam first, then L-BFGS) |
| `adam_lbfgs_switch_ratio` | none | 0.7 | Fraction of the epochs spent in the Adam phase |
| `optimizer_config.learning_rate` | `--lr` | 0.005 | Learning rate |
| `optimizer_config.weight_decay` | `--weight-decay` | 0.0005 | L2 penalty on the network weights (Adam only) |
| `gradient_clipping` | `--gradient-clipping` | 1.0 | Largest gradient norm of the network weights; 0 disables it. Not in the file: it is the default of `TrainingConfig` |
| `lbfgs` | none | `history_size` 50, `max_iter` 20, `line_search_fn` `strong_wolfe`, `tolerance_grad` 1e-7, `tolerance_change` 1e-9 | Passed to `torch.optim.LBFGS` |

- L-BFGS runs without gradient clipping, weight decay or adaptive weights.
- Parameters of the equation that are being identified never receive weight decay.
- **L-BFGS safeguard**: a step whose loss is not finite, or more than 10 times the loss before it, is rolled back and the L-BFGS rate is halved. After 3 such steps in a row the trainer returns to Adam.

**Learning-rate schedule**

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `scheduler_type` | `--lr-scheduler` | `cosine` | `cosine`, `reduce_lr` or `constant` (`constant` only through the config) |
| `warmup_epochs` | none | 0 | Epochs of linear ramp before the schedule starts |
| `reduce_lr_params` | none | `factor` 0.5, `patience` 50, `min_lr` 1e-6 | The plateau schedule, and the floor of the cosine one |

- `cosine` runs from `learning_rate` to `min_lr` over `num_epochs - warmup_epochs`.
- `reduce_lr` watches the epoch training loss.
- While L-BFGS runs, any schedule other than `constant` is replaced by `reduce_lr`.

**Sampler**

| `collocation_distribution` | Default | How points are drawn |
|---|---|---|
| `uniform` | yes | Uniformly at random in the domain |
| `stratified` | no | Each axis cut into equal bins with one random sample per bin (Latin-hypercube style) |
| `residual_based` | no | RAD, residual-based adaptive distribution (Wu et al., 2023): from a uniform pool of 4 times the batch, with probability $p \propto \lvert r\rvert^k / \overline{\lvert r\rvert^k} + c$ in the residual $r$, with $k = 1$, $c = 0$ |
| `adaptive` (`--rl`) | no | Experimental bandit on a 16 × 8 grid of $(x, t)$ cells, 1-D equations only; half the points from the policy, half uniform |

- `residual_based` falls back to `uniform` when the residual cannot be computed or the weights are not finite.

**Loss**

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `loss_weights` | none | `residual` 15, `boundary` 20, `initial` 10, `data` 10 | The static weights $w_r, w_b, w_i, w_d$ |
| `adaptive_weights` | `--adaptive-weights` | off (`strategy` `rbw`, `alpha` 0.7, `eps` 1e-6, `initial_weights` [0.3, 0.4, 0.3]) | `lrw` or `rbw`; replaces $w_r, w_b, w_i$, never $w_d$ |
| `loss_function` | `--loss-function` | `mse` | The reduction $\ell$: `mse`, `mae` or `huber` |
| `huber_delta` | `--huber-delta` | 1.0 | Threshold of the Huber function |

- **`lrw`**: each physics weight is proportional to the inverse of the running mean of that term's gradient norm, $w_k \propto 1 / (\bar g_k + \varepsilon)$. A term with zero gradient gets weight 0.
- **`rbw`**: each physics weight is proportional to the running mean of that term's loss.
- **`alpha`**: how much of the running mean is kept at each update. A lower value adapts faster.
- Adaptive weights sum to 1. The first step uses `initial_weights`.

**Stopping and validation**

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `early_stopping` | `--early-stopping` | `enabled` true, `patience` 100, `min_delta` 1e-7 | Stop when the validation loss has not improved by `min_delta` for `patience` epochs |
| `validation_frequency` | none | 10 | Validate, log, check early stopping and checkpoint every N epochs |

**Engine and normalisation**

| Option | Flag | Default | Meaning |
|---|---|---|---|
| `derivatives` | `--derivatives` | `auto` | `auto` takes the residual's derivatives from the compiled engine (the C++ kernel `pinnrl.native`, see [The native engine](engine.md)) where it applies and from autograd elsewhere; `autograd` switches the engine off |
| `normalize` | `--normalize` | off | Network inputs mapped from the domain to $[-1, 1]$, outputs scaled by the `scales` of the equation's spec, loss terms in those units |

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

The two loops apply exactly the same update:

| | `train` | `train_steps` |
|---|---|---|
| Unit | epochs | steps, or a time budget (`time_budget_s`) |
| Validation, early stopping, schedule, warm-up | yes | no |
| Run files | yes, with `experiment_dir` | no |
| Sampler | the configured one | the configured one, or `sample_fn(model, progress) -> (x, t)` |
| Evaluation along the way | every `validation_frequency` epochs | `checkpoints` and `on_checkpoint` |

- Each `train()` call plans its own schedule: the learning rate restarts from `training.learning_rate`, and the cosine period and the `adam_lbfgs` switch epoch follow that call's `num_epochs`.
- `trainer.history` holds per-epoch entries, including `loss_weights` when adaptive weighting is on.

### Time-marching

A network trained on the whole time span at once can settle on a wrong solution with a small residual: a trivial solution close to the initial condition, or a smoothed-out sharp transition. `train_windows` trains on a growing time window $[t_0, c_k]$ instead, where the last cutoff $c_k$ is the end of the time domain.

<!-- doctest: skip -->
```python
cutoffs = [2.5, 5.0, 10.0]                         # the last one is the end of the time domain
out = trainer.train_windows(cutoffs, steps_per_window=4000, resample_every=50, lbfgs_steps=500)
```

1. For each cutoff, it sets `pde.time_cutoff`, which restricts the collocation samplers, the observations and the sensor records to the window.
2. It trains `steps_per_window` steps. The network and the identified unknowns carry over to the next window.
3. After the last window it resets `pde.time_cutoff`.
4. With `lbfgs_steps > 0` it finishes with that many L-BFGS steps on the whole span.

Experiment [H04](benchmarks.md) measures what it buys on the systems where one-shot training fails.

### Restarts

A network that identifies an unknown function from sensor records can fail in ways the loss curve does not show. `pinnrl.training.train_with_restarts` retrains from a new seed until the fit passes a test it can run without knowing the truth:

$$
\chi^2 = \sum_{k} \left(\frac{H_k[u_\theta] - y_k}{\sigma_k}\right)^2 \le \chi^2_{\text{level}}(n)
$$

- **$y_k$, $\sigma_k$**: a sensor record and its noise standard deviation.
- **$H_k[u_\theta]$**: what the sensor would read on the trained network.
- **$\chi^2_{\text{level}}(n)$**: the `level` quantile of the chi-square distribution for $n$ records.

Steps:

1. Seed, build a trainer with `make_trainer(seed)`, and run `train(trainer)`.
2. Compute the sensor misfit with `pde.sensor_misfit(trainer.model, level)`.
3. If it passes, stop. Otherwise start again from `seed + 1`.
4. After `attempts` runs, return the run with the lowest $\chi^2$; `result.accepted` says whether it passed.

| Argument | Default | Meaning |
|---|---|---|
| `attempts` | 3 | The most runs to train |
| `level` | 0.999 | The chi-square quantile a run must stay under |
| `seed` | 0 | The first seed; attempt $k$ uses `seed + k` |

In J06, 22 of 36 runs collapsed to a constant fouling or stuck at a bound. In J07b, one closure in three left the basin its start had put it in. The misfit test caught every one of those failures. See [Process twins](process_twins.md).

### Reloading a run

To reload a finished CLI or dashboard run:

<!-- doctest: skip -->
```python
from pinnrl.training.train import load_run

pde, model, config = load_run("runs/20261023_120000_Burgers Equation_fourier_no_rl")
print(pde.validate(model)["l2_error"])
```

## Choosing settings

1. Start with `feedforward` or `fourier`, Adam at 1e-3, and uniform sampling.
2. Change one thing at a time.
3. Watch each loss term, not only the total. Badly scaled terms are the most common silent failure: the boundary or initial loss stalls while the residual drops.
4. If that happens, try `--adaptive-weights lrw`, or raise `loss_weights.boundary` and `loss_weights.initial`.
5. On smooth problems, try `adam_lbfgs`. It usually improves the last digit. It rarely rescues a run that Adam got wrong.
6. On problems with sharp features, compare `residual_based` against `uniform` at the same budget (see [Benchmarks](benchmarks.md)). Do not assume it helps.
7. For an inverse run, set `weight_decay=0` and `gradient_clipping=0` first (see [Inverse problems](inverse_problems.md#regularisation-defaults-for-inverse-runs)).
