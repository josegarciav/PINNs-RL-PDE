# Configuration file

`pinnrl-train` and `Config()` read one YAML file: the packaged `pinnrl/config/config.yaml`, or the one passed with `--config`. A run writes the exact configuration it used next to its results, and `load_run` rebuilds the run from it. The dataclasses the file is turned into are on the [API page](../api/config.md).

The file has these top-level keys.

| Key | Content |
|---|---|
| `device` | `"mps"`, `"cuda"` or `"cpu"`; an unavailable backend falls back to `cpu` |
| `architectures` | One block per architecture, read when that architecture is chosen |
| `pde_type` | The problem `pinnrl-train` trains when `--pde` is not given |
| `pde_configs` | One block per shipped problem |
| `training` | The budget, the optimizer, the loss and the sampler |
| `rl` | The experimental bandit sampler |
| `paths` | `results_dir`, relative to the working directory (`runs`) |

## `architectures`

| Architecture | Keys (packaged values) |
|---|---|
| `feedforward` | `hidden_dims` (7 × 128), `activation` (`tanh`), `dropout` (0), `layer_norm` (`true`) |
| `resnet` | `hidden_dim` (512), `num_blocks` (7), `activation` (`tanh`), `dropout` (0) |
| `siren` | `hidden_dims` (7 × 124), `omega_0` (30) |
| `fourier` | `mapping_size` (512), `hidden_dims` (4 × 512), `scale` (4), `activation` (`tanh`) |
| `attention` | `hidden_dim` (124), `num_layers` (4), `num_heads` (4), `activation` (`gelu`), `dropout` (0) |
| `autoencoder` | `latent_dim` (64), `hidden_dims` (124, 248, 124; the decoder mirrors them), `activation` (`relu`), `layer_norm` (`true`) |
| `fno` | `hidden_dim` (256), `num_blocks` (4), `modes` (16), `activation` (`gelu`) |

The [compiled engine](../engine.md) applies to `feedforward` with `tanh` and no dropout, with or without `layer_norm`. Every other architecture trains by autograd.

## `pde_configs`

Each block describes one problem: `name`, the default `architecture`, `input_dim`, `output_dim`, `parameters`, `domain` (one `[lo, hi]` per space dimension), `time_domain`, `dimension`, `initial_condition`, `boundary_conditions` (`dirichlet`, `neumann` or `periodic`) and `exact_solution`. The shipped keys are `heat`, `heat_2d`, `wave`, `burgers`, `convection`, `kdv`, `allen_cahn`, `cahn_hilliard`, `black_scholes` and `pendulum`; the [equations page](../equations.md) states each problem. Your own equations do not go here: write them as text for [`CustomPDE`](../custom_equations.md) or in a `--pde-file`.

## `training`

| Key | Packaged value | Meaning |
|---|---|---|
| `num_epochs` | 3000 | Epochs |
| `batch_size` | 2048 | Collocation points per optimizer step |
| `num_collocation_points` | 5000 | Collocation points per epoch |
| `num_boundary_points` | 5000 | Points per boundary face |
| `num_initial_points` | 5000 | Points on the initial condition |
| `collocation_distribution` | `uniform` | `uniform`, `stratified`, `residual_based` (RAD) or `adaptive` (the RL bandit, experimental) |
| `optimizer` | `adam` | `adam`, `lbfgs` or `adam_lbfgs` |
| `adam_lbfgs_switch_ratio` | 0.7 | Fraction of the epochs in the Adam phase |
| `optimizer_config.learning_rate` | 0.005 | Learning rate |
| `optimizer_config.weight_decay` | 0.0005 | L2 coefficient on the weights; set it to 0 for inverse problems |
| `gradient_clipping` | 1.0 (not in the file; the default of `TrainingConfig`) | Maximum gradient norm; 0 disables it, as inverse problems need |
| `scheduler_type` | `cosine` | `cosine`, `reduce_lr` or `constant`; `warmup_epochs` (0) ramps the rate linearly first |
| `reduce_lr_params` | `factor` 0.5, `patience` 50, `min_lr` 1e-6 | The plateau scheduler, and the floor of the cosine one |
| `lbfgs` | `history_size` 50, `max_iter` 20, `line_search_fn` `strong_wolfe`, `tolerance_grad` 1e-7, `tolerance_change` 1e-9 | Passed to `torch.optim.LBFGS` |
| `loss_weights` | `residual` 15, `boundary` 20, `initial` 10, `data` 10 | Weights of the loss terms |
| `adaptive_weights` | `enabled` `false`, `strategy` `rbw`, `alpha` 0.7, `eps` 1e-6, `initial_weights` [0.3, 0.4, 0.3] | `lrw` weights by the inverse gradient norm, `rbw` by the loss; replaces `loss_weights` for the physics terms |
| `loss_function` | `mse` | `mse`, `mae` or `huber` (`huber_delta` 1.0) |
| `mode` | `forward` | `forward`, `inverse`, `data_augmented` or `data_only` |
| `validation_frequency` | 10 | Validate, log, check early stopping and checkpoint every N epochs |
| `early_stopping` | `enabled` `true`, `patience` 100, `min_delta` 1e-7 | On the validation loss |

`CustomPDE.build_trainer(**training)` takes the same keys as keyword arguments (`learning_rate` and `weight_decay` at the top level), over the packaged values.

## `rl`

The experimental residual-bandit sampler (`--rl`; 1-D problems): `enabled` (`false`), `n_x_cells` (16), `n_t_cells` (8), `policy_fraction` (0.5: the share of points from the learned policy), `temperature` (0.5), `probe_cells` (32), `probe_points` (8), `probe_every` (10 batches). It does not beat uniform sampling at equal cost ([Benchmarks](../benchmarks.md#sampling-study-experiments-b01-and-b06)).
