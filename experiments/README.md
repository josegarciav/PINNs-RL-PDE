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
| `K01_fisher_information/` | K01 | Fisher information from the trained surrogate (implicit-function theorem, MINRES) checked against closed forms (heat, KdV) and a numerical reference (Burgers). README with results, `run.py`. |
| `K03_training_conditioning/` | K03 | Loss-Hessian and tangent-kernel spectra across eight operators and two architectures, against the final error. `run.py`. |
| `B07_quasi_random_sampling/` | B07 | Visual comparison of uniform, LHS and Halton point sets. Formerly `notebooks/sampling_techniques.ipynb`. |
