# Datasets (The Well)

[The Well](https://github.com/PolymathicAI/the_well) (Polymathic AI, BSD-3) is a collection of large, high-fidelity PDE simulation datasets. `pinnrl` registers 11 of them. That lets you train neural surrogates on reference data rather than only on analytical toys, and use simulation snapshots as observations.

```bash
pip install 'pinnrl[well]'
pinnrl-train --pde heat --dataset active_matter --arch fno --epochs 200 --dataset-steps 9
```

`--dataset` sets the mode to the dataset's recommendation (`data_only` for every registered dataset) unless you pass `--mode`. `--pde` is still required, but in `data_only` mode it is a placeholder: the dataset's domain, dimension and channel count replace the PDE's, and the physics terms are dropped.

Data streams from the Hugging Face mirror (`hf://datasets/polymathic-ai/`) and is cached in `~/.cache/pinnrl/well`; set `PINNRL_WELL_CACHE` to change that. A run reads the first `--dataset-steps` time steps of each of `--dataset-traj` trajectories from `--dataset-split`. The default is up to 101 steps (the_well's rollout cap) for 2-D datasets and 11 for 3-D ones, which keeps memory and download size in check. The dashboard's launch form has a matching "Time steps" field. It then samples `--dataset-points` `(x, t, u)` tuples, with `--dataset-seed` fixing the sample. Fewer steps means a much smaller download: notebook 04 reads 9 steps, about 26 MB per trajectory instead of about 230 MB.

## Registered datasets

Domains, time steps and channel counts are taken from each dataset's own files. The spatial domain is the same in every file. `dt × steps` describes the first training file: five datasets have files that differ. `rayleigh_taylor_instability` has one file per Atwood number, spanning 29.5 to 118 time units. `helmholtz_staircase` has one file per frequency, spanning 2.5 to 100. `viscoelastic_instability` files store 20 or 60 steps. The step size of `rayleigh_benard` and `planetswe` varies by up to 1.6 %. A PDE fed by a Well dataset therefore takes its time window from the observations it loads, not from the registry. The run's `config.yaml` records the window, so `load_run` rebuilds it without the data.

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

None of these datasets is governed by one of `pinnrl`'s PDE classes. The acoustic and Helmholtz sets involve variable coefficients and a frequency-domain formulation, which the constant-speed `wave` equation does not represent. Every entry therefore defaults to `data_only`. `euler_multi_quadrants` is not registered: its open-boundary files record negative trajectory counts, which `the_well` 1.2 cannot open, and the periodic variant is not on the mirror.

## Modes

- **`data_only`**: plain regression on the snapshots, which gives a neural surrogate.
- **`data_augmented`** and **`inverse`**: the snapshots plus the residual of a PDE (with trainable parameters in `inverse`). Use these with your own observations of a system that one of the [shipped equations](equations.md) actually governs. Passing `--mode` overrides the dataset's recommendation, so a Well dataset can still be combined with a PDE as an explicit modelling choice, as notebook 04 shows.

From Python, `pinnrl.datasets.load_well_slice(name, split, n_traj, n_points, seed, n_steps=...)` returns the same `{"x", "t", "u"}` tensors, usable as `PDEConfig.observation_data`. Notebook 04 trains an FNO-style network on `active_matter` end to end.
