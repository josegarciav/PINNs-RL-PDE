# Benchmark Datasets — The Well

`pinnrl` integrates with [The Well](https://github.com/PolymathicAI/the_well) (Polymathic AI, BSD-3 license), a curated collection of 16 large-scale physics simulation datasets totaling ~15 TB of HDF5 trajectories. Once installed via `pip install 'pinnrl[well]'`, every dataset is selectable from the dashboard "New Training" form and from the `pinnrl-train --dataset` CLI flag.

This page lists every dataset with its physical system, governing equations, scientific application domains, and the loss recipe pinnrl uses to train against it.

---

## How pinnrl uses Well data

The Well stores trajectories as HDF5 arrays of shape `(n_traj, n_steps, *spatial, n_fields)` in float32. `pinnrl.datasets.load_well_slice` flattens a slice into the same `(x, t, u)` tensor triple pinnrl uses everywhere, then trains under one of two recipes:

| Mode | Loss recipe | When to use |
|---|---|---|
| `data_only` | Pure regression on the snapshot field (no PDE residual) | Datasets whose physics is not analytically modelled in pinnrl (most Well datasets). The network learns to reproduce the simulation as a neural surrogate. |
| `data_augmented` | PDE residual + IC/BC + data fit | Datasets whose underlying PDE matches an analytical pinnrl solver (e.g. wave-equation variants for acoustic scattering). Combines the dataset as ground truth with the physics constraint. |

Mode is auto-suggested per dataset; you can override it from the Mode dropdown.

---

## The 16 datasets

Each card lists the canonical name string (passed to `well_dataset_name`), spatial dimensionality, governing equations, scientific application areas, and the recipe pinnrl picks by default. Where a dataset ships in multiple variants (different resolutions, boundary conditions, or geometries), all variant names are noted.

### 1. `active_matter`

- **Variants:** `active_matter`
- **Dimensionality:** 2D + time
- **Physics:** Self-propelled rod-like particles in a Stokes fluid, governed by a coupled set of advection-diffusion equations for concentration, polarisation, and an active stress tensor.
- **Why it matters:** Active matter is the physics of bacterial colonies, swimming microorganisms, motile cell monolayers, and synthetic colloidal microswimmers. Datasets like this support biophysics research on collective motion, tissue mechanics, and the design of micro-robotic swarms.
- **Default recipe:** `data_only` with FNO. Five output channels (concentration, two velocity components, two orientation tensor components).

### 2. `rayleigh_benard`

- **Variants:** `rayleigh_benard`, `rayleigh_benard_uniform`
- **Dimensionality:** 2D + time
- **Physics:** Boussinesq Navier–Stokes equations coupled to a buoyancy field — a fluid heated from below and cooled from above develops convection rolls and turbulent plumes.
- **Why it matters:** A canonical testbed for thermal convection, used in atmospheric science (cloud formation, mantle convection), engineering (cooling systems, heat exchangers), and stellar physics. The `_uniform` variant fixes Rayleigh and Prandtl numbers for clean benchmark comparisons.
- **Default recipe:** `data_only` with FNO. Four channels (buoyancy, pressure, two velocity components).

### 3. `shear_flow`

- **Variants:** `shear_flow`
- **Dimensionality:** 2D + time
- **Physics:** Incompressible Navier–Stokes with a passive tracer — opposing shear bands roll up via Kelvin–Helmholtz instability into vortex sheets and ultimately turbulence.
- **Why it matters:** Shear-driven mixing is the prototype for atmospheric jets, oceanic boundary layers, mixing in chemical reactors, and aircraft wake vortices. The dataset gives ML models a clean entry point into 2D incompressible turbulence.
- **Default recipe:** `data_only` with FNO. Four channels (tracer, pressure, two velocity components).

### 4. `rayleigh_taylor_instability`

- **Variants:** `rayleigh_taylor_instability`
- **Dimensionality:** 3D + time
- **Physics:** Compressible hydrodynamics of two fluid layers with the heavier fluid above the lighter one — gravity drives mushroom-cap mixing fronts that cascade to fully developed turbulence.
- **Why it matters:** The Rayleigh–Taylor instability is the dominant mixing mechanism in inertial confinement fusion targets, supernova remnants, and stellar interiors. Realistic 3D snapshots are expensive to generate from first principles, so a learned surrogate accelerates downstream design loops in fusion ignition and astrophysical modelling.
- **Default recipe:** `data_only` with MLP. Five channels (density, pressure, three velocity components).

### 5. `turbulent_radiative_layer_2D` / `turbulent_radiative_layer_3D`

- **Variants:** `turbulent_radiative_layer_2D`, `turbulent_radiative_layer_3D`
- **Dimensionality:** 2D and 3D + time
- **Physics:** Compressible Euler equations with optically thin radiative cooling — a hot, low-density gas mixes with a cold, dense gas across a shear layer that fragments under cooling instabilities.
- **Why it matters:** Radiative mixing layers control how galaxies feed their halos with cool gas and how supernova ejecta condense into clouds. The 2D/3D pair lets researchers study how dimensional restriction biases the predicted cooling efficiency, a known systematic in galaxy-evolution simulations.
- **Default recipe:** `data_only` with FNO (2D) / MLP (3D). Four channels: density, pressure, velocity components.

### 6. `MHD` (resolutions: `MHD_64`, `MHD_256`)

- **Variants:** `MHD_64`, `MHD_256`
- **Dimensionality:** 3D + time
- **Physics:** Ideal magneto-hydrodynamics — incompressible flow coupled to an evolving magnetic field via the induction equation.
- **Why it matters:** MHD is the backbone of plasma astrophysics (solar wind, accretion disks, galaxy clusters) and laboratory plasma engineering (tokamak confinement, reconnection studies). The 64³ and 256³ resolutions let researchers study spectral cascades and the convergence of trained surrogates with grid refinement.
- **Default recipe:** `data_only` with MLP. Seven channels (density, three velocity components, three magnetic-field components).

### 7. `gray_scott_reaction_diffusion`

- **Variants:** `gray_scott_reaction_diffusion`
- **Dimensionality:** 2D + time
- **Physics:** Two coupled reaction–diffusion PDEs (`A` and `B` species) under different feed/kill rates — the system spontaneously forms spots, stripes, spirals, and self-replicating patterns.
- **Why it matters:** A canonical pattern-formation benchmark with applications across morphogenesis (Turing patterns in animal coats and fish skins), heterogeneous catalysis, and porous-medium chemistry. The dataset spans a range of feed/kill parameters so models can study generalisation across pattern regimes.
- **Default recipe:** `data_only` with FNO. Two channels (species `A` and `B`).

### 8. `euler_multi_quadrants` (BCs: `openBC`, `periodicBC`)

- **Variants:** `euler_multi_quadrants_openBC`, `euler_multi_quadrants_periodicBC`
- **Dimensionality:** 2D + time
- **Physics:** Compressible Euler equations initialised with classic 2D Riemann problems — four piecewise-constant quadrants of density / pressure / velocity that develop shocks, contact discontinuities, and rarefaction waves.
- **Why it matters:** The gold standard for benchmarking shock-capturing schemes. Mastering these flows underpins aerodynamics (transonic and supersonic flight), high-speed propulsion (scramjets), and computational gas dynamics in general. The `openBC` variant probes shock interaction with non-reflecting boundaries; `periodicBC` is for fundamental turbulence work.
- **Default recipe:** `data_only` with FNO. Four channels (density, pressure, two velocity components).

### 9. `acoustic_scattering` (geometries: `maze`, `inclusions`, `discontinuous`)

- **Variants:** `acoustic_scattering_maze`, `acoustic_scattering_inclusions`, `acoustic_scattering_discontinuous`
- **Dimensionality:** 2D + time
- **Physics:** Linear scalar wave equation with a spatially varying wave speed — pressure waves scatter off corridor walls (maze), embedded inclusions, or material discontinuities.
- **Why it matters:** Acoustic / wave scattering is the forward problem behind ultrasonic non-destructive testing, sub-surface seismic imaging, room acoustics design, sonar, and biomedical ultrasound. The three geometric variants stress test how surrogates generalise across qualitatively different scatterer topologies.
- **Default recipe:** `data_augmented` with FNO — pinnrl's wave-equation solver supplies a residual constraint while the dataset anchors the prediction to the reference field. Three channels (pressure, two velocity components).

### 10. `helmholtz_staircase`

- **Variants:** `helmholtz_staircase`
- **Dimensionality:** 2D (frequency-domain)
- **Physics:** Helmholtz equation `(∆ + k²)u = f` on a staircase domain — frequency-domain wave scattering through structured corners.
- **Why it matters:** The Helmholtz equation governs steady-state monochromatic wave propagation: photonic-crystal design, acoustic metamaterials, optical waveguides, and electromagnetic compatibility analysis. The staircase geometry creates singular behaviour at corners that catches naive networks off guard, making it an exacting benchmark.
- **Default recipe:** `data_augmented` with FNO — pinnrl's wave/Helmholtz residual plus the dataset's reference field. Two channels (real and imaginary parts of the complex pressure).

### 11. `viscoelastic_instability`

- **Variants:** `viscoelastic_instability`
- **Dimensionality:** 2D + time
- **Physics:** Oldroyd-B viscoelastic flow — Navier–Stokes coupled to a polymer-stress tensor `c` that evolves with the fluid and resists stretching.
- **Why it matters:** Viscoelastic instabilities (elastic turbulence, drag reduction in pipelines) drive non-Newtonian flow in polymer melts, bio-fluids (blood, mucus), enhanced oil recovery, and microfluidic mixers. ML surrogates here aim to bridge the gap between cheap Newtonian models and expensive constitutive-law-resolved simulations.
- **Default recipe:** `data_only` with FNO. Six channels (three components of the conformation tensor, pressure, two velocity components).

### 12. `planetswe`

- **Variants:** `planetswe`
- **Dimensionality:** 2D on a sphere (longitude × latitude) + time
- **Physics:** Shallow-water equations on a rotating sphere with topographic forcing — height field plus zonal and meridional velocities.
- **Why it matters:** Shallow-water-on-sphere is the workhorse dynamical core for global atmosphere and ocean models. Datasets like this provide a clean, large-scale benchmark for ML weather emulators, climate downscaling, and tide simulation, where matching the spherical geometry and Coriolis dynamics is essential.
- **Default recipe:** `data_only` with FNO. Three channels (height, zonal velocity, meridional velocity).

### 13. `convective_envelope_rsg`

- **Variants:** `convective_envelope_rsg`
- **Dimensionality:** 3D + time
- **Physics:** Radiation-hydrodynamic simulations of the convective envelope of a Red Super-Giant star — buoyancy-driven flow on stellar scales with realistic equation of state and opacity tables.
- **Why it matters:** Red super-giants (Betelgeuse, Antares) are the immediate progenitors of core-collapse supernovae, and their surface variability and mass-loss rate set the stage for those explosions. Direct simulation is enormously expensive; a trained surrogate makes parameter sweeps over stellar mass and metallicity tractable.
- **Default recipe:** `data_only` with MLP. Mass density, pressure, internal energy, three velocity components.

### 14. `post_neutron_star_merger`

- **Variants:** `post_neutron_star_merger`
- **Dimensionality:** 3D + time
- **Physics:** Relativistic magnetohydrodynamics with neutrino transport, capturing the disk and ejecta in the seconds after two neutron stars coalesce.
- **Why it matters:** Neutron-star mergers are the dominant production sites for r-process heavy elements (gold, platinum, rare earths) in the Universe and are also gravitational-wave sources detectable by LIGO. Surrogate models here let astrophysicists rapidly forecast electromagnetic counterparts (kilonovae) for upcoming GW events.
- **Default recipe:** `data_only` with MLP. Density, pressure, velocity, magnetic field, electron fraction.

### 15. `supernova_explosion` (resolutions: `64`, `128`)

- **Variants:** `supernova_explosion_64`, `supernova_explosion_128`
- **Dimensionality:** 3D + time
- **Physics:** Compressible hydrodynamics with self-gravity and a stiffened-gas equation of state — the shock launched by a stellar core collapse propagates through the surrounding stellar envelope.
- **Why it matters:** Modelling supernova explosions is central to nucleosynthesis, cosmic-ray acceleration, feedback in galaxy evolution, and the formation of compact objects. The dual resolutions probe how well surrogates extrapolate from coarse training data to higher-fidelity targets — directly relevant to multi-resolution astrophysical pipelines.
- **Default recipe:** `data_only` with MLP. Density, pressure, three velocity components.

### 16. `turbulence_gravity_cooling`

- **Variants:** `turbulence_gravity_cooling`
- **Dimensionality:** 3D + time
- **Physics:** Driven compressible turbulence with self-gravity and radiative cooling — supersonic flows fragment, gravity collapses dense filaments, and cooling controls the final density distribution.
- **Why it matters:** This is the regime of star formation in molecular clouds: the interplay between supersonic turbulence, gravitational collapse, and cooling sets the stellar initial mass function — one of the most fundamental predictions of theoretical astrophysics. Surrogates accelerate parameter studies of star-formation efficiency in differing galactic environments.
- **Default recipe:** `data_only` with MLP. Density, pressure, three velocity components, gravitational potential.

---

## Quick-reference table

| # | Dataset | Dim | Domain | Recipe | Variants |
|---|---|---|---|---|---|
| 1 | active_matter | 2D | Biophysics / soft matter | `data_only` | 1 |
| 2 | rayleigh_benard | 2D | Thermal convection / climate | `data_only` | 2 |
| 3 | shear_flow | 2D | Mixing / turbulence | `data_only` | 1 |
| 4 | rayleigh_taylor_instability | 3D | Fusion / astrophysics | `data_only` | 1 |
| 5 | turbulent_radiative_layer | 2D + 3D | Galaxy gas dynamics | `data_only` | 2 |
| 6 | MHD | 3D | Plasma physics / astrophysics | `data_only` | 2 (resolutions) |
| 7 | gray_scott_reaction_diffusion | 2D | Pattern formation / chemistry | `data_only` | 1 |
| 8 | euler_multi_quadrants | 2D | Aerodynamics / shock physics | `data_only` | 2 (BCs) |
| 9 | acoustic_scattering | 2D | Imaging / NDT / acoustics | `data_augmented` | 3 (geometries) |
| 10 | helmholtz_staircase | 2D (freq) | Photonics / metamaterials | `data_augmented` | 1 |
| 11 | viscoelastic_instability | 2D | Polymer / non-Newtonian flow | `data_only` | 1 |
| 12 | planetswe | 2D sphere | Climate / weather | `data_only` | 1 |
| 13 | convective_envelope_rsg | 3D | Stellar physics | `data_only` | 1 |
| 14 | post_neutron_star_merger | 3D | GW astrophysics / nucleosynthesis | `data_only` | 1 |
| 15 | supernova_explosion | 3D | Astrophysics | `data_only` | 2 (resolutions) |
| 16 | turbulence_gravity_cooling | 3D | Star formation | `data_only` | 1 |

The currently shipped registry (`pinnrl.datasets.WELL_REGISTRY`) covers a curated subset; extending it to the full 23 named variants is a matter of adding entries to `pinnrl/datasets/registry.py`.

---

## Loading a dataset programmatically

```python
from pinnrl.datasets import load_well_slice

slice_ = load_well_slice(
    name="active_matter",     # any key in WELL_REGISTRY
    split="train",             # train | valid | test
    n_traj=2,                  # how many trajectories to draw from
    n_points=5_000,            # how many (x, t, u) points to keep
    seed=0,                    # RNG seed for reproducibility
    device="cpu",              # torch device
    base=None,                 # None = HuggingFace streaming; or local download dir
)
# slice_["x"]: (5000, 2) — spatial coords
# slice_["t"]: (5000, 1) — time coords
# slice_["u"]: (5000, n_fields) — reference field values
```

Subsequent calls with the same `(name, split, n_traj, n_points, seed)` reuse a cached `.npz` from `~/.cache/pinnrl/well/` (override with `PINNRL_WELL_CACHE`).

---

## Launching a training run on a Well dataset

### From the dashboard

1. Open the **New Training** sub-tab.
2. Tick **Train on a Well benchmark dataset** under the Dataset section.
3. Pick a dataset — the PDE selector, recommended architecture, and Mode dropdown auto-fill from the registry.
4. Choose **Hugging Face streaming** (default) or **Local download dir**.
5. Set trajectories and sampled-point counts, then **Start Training**.

### From the CLI

```bash
pinnrl-train \
  --pde "Heat Equation" --arch fno \
  --epochs 200 --device cpu \
  --dataset active_matter \
  --dataset-traj 2 --dataset-points 5000 \
  --mode data_only
```

When `--dataset` is supplied, registry defaults overlay the PDE block (domain, dimension, output channels) so the placeholder `--pde` value is mostly cosmetic — the data drives the shape of the problem.

---

## When to pick which dataset

- **Quickstart / smoke tests:** `active_matter` (small, 2D, fast convergence on a single trajectory).
- **Convection / thermal physics:** `rayleigh_benard` or `rayleigh_benard_uniform`.
- **Compressible aero / shock benchmarks:** `euler_multi_quadrants_periodicBC`.
- **Wave equation cross-checks (matched mode):** `acoustic_scattering_maze` with `data_augmented`.
- **Climate / atmosphere:** `planetswe`.
- **Plasma / astrophysics:** `MHD_64` (start small, then `MHD_256`).
- **Pattern-formation studies:** `gray_scott_reaction_diffusion`.
- **Stress test for high-dim 3D:** `turbulence_gravity_cooling`, `post_neutron_star_merger`.

The full registry (with metadata) is at `pinnrl/datasets/registry.py`. Add new entries there to make new datasets selectable from the dashboard.

---

## Licensing

The Well datasets are distributed under [BSD-3-Clause](https://github.com/PolymathicAI/the_well/blob/master/LICENSE). When publishing results trained on Well data, cite [the original Polymathic AI paper](https://arxiv.org/abs/2412.00568) alongside `pinnrl`.
