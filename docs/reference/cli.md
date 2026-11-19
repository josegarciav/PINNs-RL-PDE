# Command-line reference

The package installs three console scripts. Every flag is listed here; `--help` on any of them prints the same. The functions behind them are on the [API page](../api/cli.md).

## `pinnrl-train`

Trains one network on one problem and writes the run to `./runs/` in the working directory. A flag that is not given takes its value from the packaged `config.yaml` (or the file passed with `--config`); see the [configuration reference](config.md).

```bash
pinnrl-train --pde burgers --arch fourier --epochs 2000
pinnrl-train --pde heat --arch fourier --mode inverse --identify alpha --initial-guess alpha=0.5
pinnrl-train --pde-file my_equation.yaml --arch feedforward --normalize
```

**The problem and the network**

| Flag | Values | Default | Effect |
|---|---|---|---|
| `--pde` | a config key or display name (`heat`, `"Heat Equation"`) | `pde_type` of the config | The shipped problem to train |
| `--pde-file` | path to a YAML file | none | Your own equations (see [Custom equations](../custom_equations.md#from-the-command-line)); replaces `--pde` |
| `--arch` | `feedforward`, `resnet`, `fourier`, `siren`, `fno`, `attention`, `autoencoder` | required | The architecture |
| `--normalize` | flag | off | Non-dimensionalize the network: inputs mapped from the domain to $[-1, 1]$, outputs scaled by the `scales` of a `--pde-file` problem |
| `--derivatives` | `auto`, `autograd`, `native` | `auto` | `auto`: the [compiled engine](../engine.md) wherever it applies, autograd elsewhere. `autograd`: never the engine. `native`: the earlier name of `auto` |
| `--device` | `cpu`, `mps`, `cuda` | `device` of the config | An unavailable backend falls back to `cpu` |
| `--config` | path | the packaged `config.yaml` | The configuration file |

**The budget**

| Flag | Default | Effect |
|---|---|---|
| `--epochs` | `training.num_epochs` | Number of epochs |
| `--lr` | `training.optimizer_config.learning_rate` | Learning rate |
| `--batch-size` | `training.batch_size` | Collocation points per optimizer step |
| `--collocation-points` | `training.num_collocation_points` | Collocation points per epoch |
| `--boundary-points` | `training.num_boundary_points` | Points per boundary face |
| `--initial-points` | `training.num_initial_points` | Points on the initial condition |

**The optimizer and the loss**

| Flag | Values | Effect |
|---|---|---|
| `--optimizer` | `adam`, `lbfgs`, `adam_lbfgs` | The optimizer, or Adam followed by L-BFGS |
| `--lr-scheduler` | `cosine`, `reduce_lr` | Cosine annealing, or reduce on a plateau of the epoch training loss |
| `--early-stopping` | `on`, `off` | Early stopping on the validation loss |
| `--gradient-clipping` | a number | Maximum gradient norm; `0` disables clipping |
| `--weight-decay` | a number | L2 coefficient on the network's weights |
| `--loss-function` | `mse`, `mae`, `huber` | Reduction of the residual, boundary, initial and data terms |
| `--huber-delta` | a number | Delta of the Huber loss |
| `--adaptive-weights` | `off`, `lrw`, `rbw` | Static weights, gradient-balanced or loss-balanced weights of the loss terms |

**Collocation sampling**

| Flag | Values | Effect |
|---|---|---|
| `--collocation-distribution` | `uniform`, `stratified`, `residual_based` | The sampler (`residual_based` is RAD) |
| `--rl` | flag | The experimental RL bandit sampler (1-D problems); overrides the distribution |

**Inverse problems and data**

| Flag | Default | Effect |
|---|---|---|
| `--mode` | `training.mode` | `forward` (residual and conditions), `inverse` (plus data and trainable parameters), `data_augmented` (residual, conditions and data), `data_only` (regression on the data) |
| `--identify` | none | A parameter to identify in inverse mode; repeatable |
| `--initial-guess` | none | A starting value, `alpha=0.5`; repeatable |
| `--obs-path` | none | An `.npz` file of observations with keys `x`, `t`, `u` |
| `--obs-points` | 200 | Number of synthetic observation points when no file is given |
| `--obs-noise` | 0.01 | Standard deviation of the noise added to synthetic observations |

For inverse runs start from `--weight-decay 0 --gradient-clipping 0`; the packaged regularization biases the estimate ([troubleshooting](../troubleshooting.md#an-identified-parameter-stalls-12-from-the-truth)).

**The Well datasets**

| Flag | Default | Effect |
|---|---|---|
| `--dataset` | none | A dataset name of `pinnrl.datasets.WELL_REGISTRY` |
| `--dataset-split` | `train` | `train`, `valid` or `test` |
| `--dataset-traj` | 1 | Samples to draw (random, without replacement) |
| `--dataset-points` | 4096 | Points sub-sampled from the loaded slice |
| `--dataset-seed` | 0 | Seed of the sub-sampling |
| `--dataset-steps` | the dataset's (101, or 11 in 3-D) | Time steps read per trajectory |
| `--dataset-base` | stream from Hugging Face | A local download directory |

## `pinnrl-benchmark`

Reference benchmarks: classical baselines and the multi-seed sampling study.

### `pinnrl-benchmark fdm`

The bundled finite-difference solver against the analytical solution: relative $L^2$ error at the final time, maximum error and wall time.

| Flag | Default | Effect |
|---|---|---|
| `--pde` | required | `heat` (explicit scheme, periodic) or `wave` (leap-frog, periodic) |
| `--alpha` | 0.1 | Heat diffusivity |
| `--frequency` | 1.0 | Frequency of the initial condition (heat) |
| `--c` | 1.0 | Wave speed |
| `--t-max` | 1.0 | Final time |
| `--nx` | 101 | Grid points |
| `--nt` | 4001 | Time steps |
| `--csv` | none | Write the solution to a CSV file |

### `pinnrl-benchmark sampling`

Collocation samplers at equal training budget, with paired seeds; see [Benchmarks](../benchmarks.md#sampling-study-experiments-b01-and-b06).

| Flag | Default | Effect |
|---|---|---|
| `--pdes` | `burgers allen_cahn` | The problems |
| `--strategies` | `uniform stratified uniform_periodic rad rl_bandit` | The samplers |
| `--seeds` | 10 | Number of seeds (0 to N-1) |
| `--steps` | 15000 | Optimizer steps per run |
| `--checkpoints` | 50 | Evaluations along each run |
| `--n-collocation` | 1000 | Collocation points |
| `--resample-every` | 10 | Steps between redraws of the points |
| `--workers` | 1 | Parallel worker processes |
| `--out` | `runs/sampling_study` | Output directory |

## `pinnrl-dashboard`

Serves the [dashboard](../dashboard.md), from which runs are configured, launched, monitored and compared.

| Flag | Default | Effect |
|---|---|---|
| `--port` | 8050 | Port |
| `--no-browser` | off | Do not open a browser tab |
