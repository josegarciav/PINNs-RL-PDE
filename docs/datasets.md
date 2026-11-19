# Datasets (The Well)

This page shows how to train on the simulation datasets of The Well and use their snapshots as observations. Read it if you want a neural surrogate of reference data, not only of analytical test problems.

$$
\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \bigl\lVert u_\theta(x_i, t_i) - u_i \bigr\rVert^2
$$

- **[The Well](https://github.com/PolymathicAI/the_well)**: a collection of large, high-fidelity PDE simulation datasets (Polymathic AI, BSD-3). `pinnrl` registers 11 of them.
- **$(x_i, t_i, u_i)$**: a sampled point of a simulation snapshot and the field values stored there.
- **$u_\theta$**: the network. Fitted to the snapshots alone, it is a **neural surrogate** of the simulation.
- **$\lVert\cdot\rVert^2$**: the squared error, the default loss function (`mse`).
- **Channel**: one stored field component. The network has one output per channel.

```bash
pip install 'pinnrl[well]'
pinnrl-train --pde heat --dataset active_matter --arch fno --epochs 200 --dataset-steps 9
```

- **`--dataset`**: sets the mode to the dataset's recommendation (`data_only` for every registered dataset) unless you pass `--mode`.
- **`--pde`** (or `--pde-file`): still required. In `data_only` mode it is a placeholder: the dataset's domain, dimension and channel count replace the PDE's, and the physics terms are dropped.

How a run reads the data:

1. Data streams from the Hugging Face mirror (`hf://datasets/polymathic-ai/`).
2. The run reads the first `--dataset-steps` time steps of each of `--dataset-traj` trajectories from `--dataset-split`.
3. It samples `--dataset-points` `(x, t, u)` tuples. `--dataset-seed` fixes the sample.
4. The slice is cached in `~/.cache/pinnrl/well`. Set `PINNRL_WELL_CACHE` to change that.

| Setting | Value |
|---|---|
| Default steps, 2-D datasets | up to 101 (the_well's rollout cap) |
| Default steps, 3-D datasets | 11, which keeps memory and download size in check |
| Dashboard | the launch form has a matching "Time steps" field |
| Fewer steps | a much smaller download: notebook 04 reads 9 steps, about 26 MB per trajectory instead of about 230 MB |

- **Trajectory**: one simulation run, stored as a sequence of time steps.
- **Split**: the `train`, `valid` or `test` part of a dataset.

## Registered datasets

Domains, time steps and channel counts are taken from each dataset's own files.

- **Spatial domain**: the same in every file.
- **`dt × steps`**: the time step and the number of stored steps of the first training file. Five datasets have files that differ:

| Dataset | How its files differ |
|---|---|
| `rayleigh_taylor_instability` | one file per Atwood number, spanning 29.5 to 118 time units |
| `helmholtz_staircase` | one file per frequency, spanning 2.5 to 100 |
| `viscoelastic_instability` | files store 20 or 60 steps |
| `rayleigh_benard`, `planetswe` | the step size varies by up to 1.6 % |

A PDE fed by a Well dataset therefore takes its time window from the observations it loads, not from the registry. The run's `config.yaml` records the window, so `load_run` rebuilds it without the data.

| Name | Physics | Spatial domain | dt × steps |
|---|---|---|---|
| `active_matter` | 2-D active nematic fluid | [0, 10]² | 0.25 × 81 |
| `rayleigh_benard` | 2-D thermal convection | [0, 4] × [0, 1] | 0.25 × 200 |
| `shear_flow` | 2-D shear flow, Kelvin–Helmholtz roll-up | [0, 1]² | 0.1 × 200 |
| `rayleigh_taylor_instability` | 3-D buoyancy-driven mixing | [0, 1]³ | 1.0 × 119 (varies by file) |
| `turbulent_radiative_layer_2D` | 2-D radiatively cooling shear layer | [−0.5, 0.5] × [−1, 2] | 1.6 × 101 |
| `MHD_64` | 3-D magnetohydrodynamic turbulence | [0, 1]³ | 0.01 × 100 |
| `gray_scott_reaction_diffusion` | 2-D reaction–diffusion patterns | [−1, 1]² | 10 × 1001 |
| `viscoelastic_instability` | 2-D elastic turbulence | [0, 2π] × [−1, 1] | 1.0 × 20 (or 60, by file) |
| `acoustic_scattering_maze` | 2-D acoustic scattering, spatially varying density and sound speed | [0, 1]² | 0.02 × 202 |
| `helmholtz_staircase` | 2-D Helmholtz (frequency-domain) scattering | [−8, 8] × [−0.5, 3.5] | 2.0 × 50 (varies by file) |
| `planetswe` | Shallow water on a sphere (colatitude θ, longitude φ in radians; cell centres) | θ ∈ [0.009, 3.132], φ ∈ [0, 6.271] | 1.0 × 1008 |

Limits of the registry:

- None of these datasets is governed by one of `pinnrl`'s PDE classes, so every entry defaults to `data_only`.
- The acoustic and Helmholtz sets involve variable coefficients and a frequency-domain formulation. The constant-speed `wave` equation does not represent them.
- `euler_multi_quadrants` is not registered. Its open-boundary files record negative trajectory counts, which `the_well` 1.2 cannot open, and the periodic variant is not on the mirror.

## Modes

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \underbrace{\mathcal{L}_{\text{residual}} + \mathcal{L}_{\text{boundary}} + \mathcal{L}_{\text{initial}}}_{\text{dropped in data-only mode}}
$$

- **`data_only`**: plain regression on the snapshots, which gives a neural surrogate.
- **`data_augmented`**: the snapshots plus the residual of a PDE.
- **`inverse`**: as `data_augmented`, with trainable parameters in the PDE.

- Use `data_augmented` and `inverse` with your own observations of a system that one of the [shipped equations](equations.md) governs.
- `--mode` overrides the dataset's recommendation. A Well dataset can still be combined with a PDE as an explicit modelling choice, as notebook 04 shows.

From Python, `pinnrl.datasets.load_well_slice(name, split, n_traj, n_points, seed, n_steps=...)` returns the same `{"x", "t", "u"}` tensors, usable as `PDEConfig.observation_data`. Notebook 04 trains an FNO-style network on `active_matter` end to end.
