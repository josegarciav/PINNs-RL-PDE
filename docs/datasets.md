# Datasets (The Well)

[The Well](https://github.com/PolymathicAI/the_well) (Polymathic AI, BSD-3) is a collection of large, high-fidelity PDE simulation datasets. `pinnrl` registers 12 of them. That lets you train on reference data rather than only on analytical toys, and use simulation snapshots as observations in inverse problems.

```bash
pip install 'pinnrl[well]'
pinnrl-train --pde heat --dataset active_matter --arch fno --mode data_only --epochs 200
```

`--pde` is always required. In `data_only` mode it is a placeholder: the dataset's domain, dimensions and channel count replace the PDE's, and the physics terms are dropped. In `data_augmented` and `inverse` modes, pass the matching PDE (e.g. `--pde wave` for `acoustic_scattering_maze`).

Data streams from the Hugging Face mirror (`hf://datasets/polymathic-ai/`) and is cached in `~/.cache/pinnrl/well`; set `PINNRL_WELL_CACHE` to change that. A run samples `--dataset-points` `(x, t, u)` tuples from `--dataset-traj` trajectories of `--dataset-split`, and `--dataset-seed` fixes the sample.

## Registered datasets

| Name | Physics | Default mode |
|---|---|---|
| `active_matter` | 2-D active nematic fluid | `data_only` |
| `rayleigh_benard` | 2-D thermal convection | `data_only` |
| `shear_flow` | 2-D shear flow, Kelvin–Helmholtz roll-up | `data_only` |
| `rayleigh_taylor_instability` | 3-D buoyancy-driven mixing | `data_only` |
| `turbulent_radiative_layer_2D` | 2-D radiatively cooling shear layer | `data_only` |
| `MHD_64` | 3-D magnetohydrodynamic turbulence | `data_only` |
| `gray_scott_reaction_diffusion` | 2-D reaction–diffusion patterns | `data_only` |
| `euler_multi_quadrants_periodicBC` | 2-D compressible Euler, Riemann problems | `data_only` |
| `viscoelastic_instability` | 2-D elastic turbulence | `data_only` |
| `acoustic_scattering_maze` | 2-D acoustic scattering | `data_augmented` |
| `helmholtz_staircase` | 2-D Helmholtz scattering | `data_augmented` |
| `planetswe` | Shallow water on a sphere | `data_only` |

`pinnrl.datasets.WELL_REGISTRY` holds each entry's defaults (architecture, mode, matched PDE). The dashboard's dataset selector fills them in.

## Modes

- **`data_only`**: plain regression on the snapshots, which gives a neural surrogate. Use it when `pinnrl` has no matching PDE.
- **`data_augmented`**: the snapshots plus the residual of a matching PDE. The physics regularizes the fit, and the data anchors the physics.
- **`inverse`**: like `data_augmented`, but some PDE parameters are trainable, which calibrates the model to the data. See [Inverse problems](inverse_problems.md).

From Python, `pinnrl.datasets.load_well_slice(name, split, n_traj, n_points, seed)` returns the same `{"x", "t", "u"}` tensors, usable as `PDEConfig.observation_data`. Notebook 04 trains an FNO-style network on `active_matter` end to end.
