# Experiments

Research studies built on the `pinnrl` package. The package (`pinnrl/`) holds
reusable code that ships to PyPI. This folder holds the questions we are
asking with it, and how each one was answered.

- [PAPERS.md](PAPERS.md) is the paper backlog: 27 papers in four tiers, each mapped to experiments.
- [BACKLOG.md](BACKLOG.md) lists 150 prioritized experiments with IDs `A01`–`K50`.
- Generated outputs from ad-hoc training (`pinnrl-train`, the dashboard) go to
  `runs/`. That folder is gitignored and is not part of this tree.

## Layout

One folder per experiment, named after its backlog ID:

```
experiments/
  BACKLOG.md
  <ID>_<slug>/            e.g. B01_sampling_equal_compute/
    README.md             question, setup, result, conclusion
    run.py | *.ipynb      how to reproduce it (calls into pinnrl)
    config.yaml           optional: the exact config used
    results/              gitignored: CSVs, curves, figures
```

The experiment `README.md` should record:

1. **Question:** the backlog question, verbatim.
2. **Setup:** PDE, architecture, budget, seeds, and the command to rerun it.
3. **Result:** the table or figure, with the median and IQR across seeds.
4. **Conclusion:** what changed in the package or in the backlog because of it.

When an experiment shows that a feature works, move the reusable code into
`pinnrl/` with tests. Keep only the study-specific driver here.

## Current folders

| Folder | Backlog | Contents |
|---|---|---|
| `A03_pinn_vs_numerical/` | A03 | Early standalone notebooks (pure PyTorch, no `pinnrl` imports) comparing PINNs with numerical solvers for heat (1-D/2-D), Allen–Cahn and KdV. Formerly `pinnrl/numerical_solvers/`. |
| `B01_sampling_equal_compute/` | B01, B06 | Uniform vs stratified vs RAD vs RL bandit at equal steps and equal residual evaluations, 10 seeds, Burgers and Allen–Cahn. README with results and conclusions, `plot.py` for the figure. |
| `F01_heat_alpha_recovery/` | F01 | Recovery of the heat diffusivity α from noisy observations: 4 noise levels × 4 observation counts × 10 seeds, against the Cramér–Rao bound. README with results, `run.py`, `plot.py`. |
| `F04_initial_guess/` | F04 | Ten initial guesses from 0.1× to 10× the true heat diffusivity, 10 seeds: the estimate depends on the data, not on the start. README with results, `run.py`, `plot.py`. |
| `F05_fixed_sensors/` | F05 | One to eight fixed probes read densely in time vs random scatter at 200 points, against each layout's Cramér–Rao bound: one mid-domain probe beats the scatter. README with results, `run.py`, `plot.py`. |
| `F02_kdv_joint/` | F02 | Joint identification of the two KdV coefficients against the 2 × 2 Cramér–Rao bound, plus a follow-up on the wrong minimum the fit converges to. `run.py`, `run_followup.py`, `plot.py`. |
| `F03_burgers_nu/` | F03 | Burgers viscosity near the shock, identified as log ν, against a finite-difference bound from the Cole–Hopf reference. `run.py`, `plot.py`. |
| `F07_profile_likelihood/` | F07 | Profile likelihood of the heat diffusivity: the data misfit at fixed α on a grid, against the Fisher parabola. `run.py`. |
| `F09_field_unknowns/` | F09 | Recovery of a drifting α(t) (linear, step) and a coefficient field α(x) (bump, two layers) on heat with `UnknownFunction`, against the best constant. README with results, `run.py`, `plot.py`. |
| `D01_optimizer_baseline/` | D01, D02 | Adam vs L-BFGS vs Adam→L-BFGS at matched wall-clock on heat, Burgers and Allen–Cahn, the switch point, and the F01 floor cell without regularization. README with results, `run.py`. |
| `F11_change_point/` | F11 | The time of a jump in α(t) as a trainable parameter vs F09's network: found to the Cramér–Rao bound with the clean configuration. `run.py`. |
| `K47_swarm_posterior/` | K47 | The swarm's weighted and accepted positions against the exact posterior on heat α and the pendulum. README with results, `run.py`. |
| `K45_parametric_surrogate/` | K45 | A network with the parameters as inputs over the prior range vs per-value training (0.2–2 % over the range at a quarter to a half of the cost), and as the swarm's forward model (the same region as the closed form). `run.py`, `plot.py`. |
| `J01_evaporator_reference/` | J01, J02 | The falling-film evaporator reference simulator (steady, dynamic, grid convergence, scenarios) and its PINN forward solve (0.4–0.6 % on the steam step). `run.py`, `plot.py`. |
| `K09_linearized_stability/` | K09 | Jacobian spectra of the catalogue's ODE systems vs the PINN's error, and the evaporator's method-of-lines spectrum vs its grid. `run.py`, `plot.py`. |
| `J03_fouling_estimation/` | J03 | The evaporator twin recovering a drifting fouling resistance from outlet sensors with the steam and feed records as inputs, function vs scalar. `run.py`, `plot.py`. |
| `H04_time_marching/` | H04 | Time-marching windows (`train_windows`) against one-shot training on the K09 failures, Van der Pol and the drying droplet. `run.py`, `plot.py`. |
| `J07_droplet_drying/` | J07, J07b–e | A drying droplet's PINN forward solve and the identification of its drying-rate closure from a mass-and-temperature record, against parametric least squares; the second pass (pointwise start, two-stage, basis) in `run_closure2.py`, the reference-solver misfit guard in `run_guard.py` and the closure by shooting in `run_shooting.py`. `run.py`, `plot.py`, `plot_closure2.py`, `plot_guard.py`. |
| `J05_ekf_baseline/` | J05, J05b | An extended Kalman filter and smoother on a lumped evaporator model against the PINN twin, with the tuning and the model-mismatch cases; moving-horizon estimation against the filter in `run_mhe.py`. `run.py`, `plot.py`. |
| `J06_fouling_intervals/` | J06 | Coverage, width and bias of the 90 % intervals of the fouling and of the time to clean over 100 data sets: batch fits on the reference, Kalman filters (random walk, trend; 11 and 51 cells), a steam offset ignored or estimated, a saturating curve, and the PINN twin with a misfit guard. README with results, `run.py`, `plot.py`. |
| `J08_stickiness/` | J08 | The stickiness margin of the powder leaving the dryer estimated four ways (water balance, lab assay, the twin, the twin with online records) over the operating window, validated on J11's fitted campaigns. README with results, `run.py`, `plot.py`. |
| `J09_measurement_design/` | J09 | Bayesian measurement design on the evaporator: what each instrument error costs and what each remedy (calibrations, a condensate meter, steam swings, lab samples and their timing) buys in the standard deviation of the fouling, with a validation by 200 simulated fits per design. README with results, `run.py`, `plot.py`. |
| `J11_dryer_closure/` | J11 | The drying-rate closure on a co-current spray-dryer tower (`SprayDryer`): what air thermocouples, outlet humidity and powder moisture at a few operating points determine of it, parametric and free-form, and the prediction at a point outside the campaign. README with results, `run.py`, `plot.py`. |
| `J04_plant_sensors/` | J04 | What outlet sensors determine about the fouling resistance: Fisher bounds on six modes of R_f(t) and four nuisances from the reference solver, and the profile likelihood of the drift. `run.py`, `plot.py`. |
| `K48_reduced_bases/` | K48 | The F09 fields in a cosine basis: Fisher spectrum of the modes from the reference solver, the swarm over the coefficients, and a 6-mode basis vs a network as the unknown. README with results, `run.py`, `plot.py`. |
| `K08_dimension_scaling/` | K08 | Heat and convection in d = 1–5: error, residual, stability constant, quadrature gap and cost vs dimension at matched total and matched per-axis points. README with results, `run.py`, `plot.py`. |
| `K02_sensor_layout/` | K02 | Sensor count, placement and kind vs the Fisher information of the layout (heat α, KdV α–β) with `sensor_jacobian`, plus α identified from 1–8 probes against each layout's bound. README with results, `run.py`, `plot.py`. |
| `K01_fisher_information/` | K01 | Fisher information from the trained surrogate (implicit-function theorem, MINRES) checked against closed forms (heat, KdV) and a numerical reference (Burgers). README with results, `run.py`. |
| `K03_training_conditioning/` | K03 | Loss-Hessian and tangent-kernel spectra across eight operators and two architectures, against the final error. `run.py`. |
| `K05_data_disturbance/` | K05 | Bias, delay, noise, dropout, outliers and single-probe faults on the identified heat diffusivity, paired, against the linear response. `run.py`, `plot.py`. |
| `K06_problem_disturbance/` | K06 | Initial, boundary and parameter perturbations of heat and wave: the trained field's response against the exact one. `run.py`, `plot.py`. |
| `K07_stability_constants/` | K07 | Error vs residual on an independent sample across ten equations and five budgets. `run.py`, `plot.py`. |
| `F06_coverage/` | F06, K10 | Coverage, width and CRPS of ensemble, Laplace, Fisher and swarm intervals over synthetic data sets. `run.py`. |
| `K39_likelihood_weighting/` | K39 | The data weight from the noise model vs fixed weights. `run.py`. |
| `K46_swarm_equivalence/` | K46 | Swarm sampling of the region of equivalence on closed-form misfits vs the Fisher ellipse. `run.py`. |
| `B02_more_pdes/` | B02 | The B01 samplers on heat, convection, wave and the KdV soliton. `run.py`, `plot.py`. |
| `B03_rad_grid/` | B03 | The RAD (k, c) grid on Burgers and Allen–Cahn. `run.py`. |
| `B04_resampling_frequency/` | B04 | Error vs resampling interval for uniform and RAD. `run.py`, `plot.py`. |
| `B08_point_count/` | B08 | Error vs collocation count for uniform and RAD. `run.py`. |
| `B07_quasi_random_sampling/` | B07 | Visual comparison of uniform, LHS and Halton point sets. Formerly `notebooks/sampling_techniques.ipynb`. |
