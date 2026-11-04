# F01: heat diffusivity recovery baseline

## Question

Accuracy of $\alpha$ vs noise and number of observations.

## Setup

- **Problem:** the 1-D heat equation $u_t = \alpha u_{xx}$ on $[0, 1] \times [0, 1]$ with $u_0 = \sin \pi x$, Dirichlet 0, and the exact solution $u = e^{-\alpha \pi^2 t} \sin \pi x$. True $\alpha = 0.1$. Written as the catalogue entry `heat` of `pinnrl.CustomPDE` with `alpha` marked trainable and started at **0.5, five times the truth**.
- **Data:** $N_{\text{obs}} \in \{20, 50, 200, 1000\}$ observations of the exact solution at uniform random points of the domain, with additive Gaussian noise of standard deviation $\sigma \in \{0, 0.01, 0.05, 0.1\}$ (the solution has amplitude 1 at $t = 0$ and $e^{-\pi^2/10} \approx 0.37$ at $t = 1$). The grid is 16 cells.
- **Model and training:** a tanh MLP (4 × 64, no layer norm) with Adam at a constant learning rate of $10^{-3}$ for the weights and for $\alpha$ alike, **5,000 steps**, 1,000 collocation points refreshed every 50 steps, 512 boundary and 512 initial points, loss weights residual / BC / IC / data = 1 / 10 / 10 / 10. The data weight is fixed; setting it from the noise level is K39.
- **Seeds:** 10. A seed fixes the observation points, the noise draw and the initial weights, which are identical across cells for a given seed, so the comparison across cells is paired.
- **Metrics:** the relative error $|\hat\alpha - \alpha| / \alpha$ at the last step, the trajectory of $\hat\alpha$ every 50 steps, the relative L2 error of $u$ on 4,000 random points, and the Cramér–Rao bound on the standard deviation of any unbiased estimator of $\alpha$ from the same observations, $\sigma / \sqrt{\sum_i (\partial u / \partial \alpha)^2}$ with $\partial u / \partial \alpha = -\pi^2 t\, u$, evaluated at the exact solution.
- **Code:** `run.py` (calls `pinnrl.pdes.catalog.from_catalog("heat", ...)` and `CustomPDE.build_trainer`), `plot.py`.

Rerun with:

```bash
python experiments/F01_heat_alpha_recovery/run.py --seeds 10 --steps 5000 --workers 4
python experiments/F01_heat_alpha_recovery/plot.py
```

## Result

![F01](../../docs/assets/benchmarks/f01_heat_alpha.png)

Relative error of $\hat\alpha$ at step 5,000, as the median [IQR] over 10 seeds; the spread (standard deviation of $\hat\alpha$ across seeds, relative to $\alpha$) next to the Cramér–Rao bound for the same data; the mean bias; and the relative L2 error of the field.

| Noise σ | $N_{\text{obs}}$ | Rel. error of α | Spread | CR bound | Bias | Rel. L2 of u |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.017 [0.013, 0.024] | 0.009 | 0 | −0.018 | 0.023 |
| | 50 | 0.019 [0.016, 0.023] | 0.005 | 0 | −0.020 | 0.031 |
| | 200 | 0.020 [0.015, 0.026] | 0.008 | 0 | −0.022 | 0.025 |
| | 1000 | 0.015 [0.014, 0.018] | 0.009 | 0 | −0.019 | 0.028 |
| 0.01 | 20 | 0.027 [0.018, 0.034] | 0.015 | 0.011 | −0.029 | 0.032 |
| | 50 | 0.016 [0.010, 0.021] | 0.007 | 0.007 | −0.015 | 0.026 |
| | 200 | 0.019 [0.015, 0.020] | 0.005 | 0.003 | −0.019 | 0.028 |
| | 1000 | 0.017 [0.016, 0.019] | 0.005 | 0.002 | −0.019 | 0.025 |
| 0.05 | 20 | 0.041 [0.022, 0.088] | 0.042 | 0.053 | −0.050 | 0.032 |
| | 50 | 0.033 [0.020, 0.038] | 0.032 | 0.035 | −0.009 | 0.031 |
| | 200 | 0.021 [0.013, 0.032] | 0.019 | 0.017 | −0.027 | 0.029 |
| | 1000 | 0.020 [0.016, 0.027] | 0.007 | 0.008 | −0.021 | 0.027 |
| 0.1 | 20 | 0.054 [0.044, 0.157] | 0.077 | 0.106 | −0.090 | 0.037 |
| | 50 | 0.055 [0.025, 0.090] | 0.065 | 0.070 | −0.002 | 0.031 |
| | 200 | 0.024 [0.009, 0.046] | 0.037 | 0.035 | −0.038 | 0.024 |
| | 1000 | 0.025 [0.013, 0.029] | 0.012 | 0.016 | −0.020 | 0.026 |

Trajectories: from the initial guess of 0.5, $\hat\alpha$ enters and stays within 5 % of the truth after 550–650 steps on most seeds. With 50 observations and $\sigma \le 0.01$ every seed settles by step 3,100; with $\sigma = 0.1$ only 4 of 10 seeds stay inside 5 % by step 5,000, the others oscillate around a value a few percent off. Each run takes about 40 s on a laptop CPU with four runs in parallel.

**Rerun without weight decay on $\alpha$ (2026-11-04).** F09 found that the optimizer's weight decay ($5 \times 10^{-4}$) had been acting on the PDE unknowns, and the trainer now exempts them. The full grid rerun with that change (`--out results/no_decay`) reproduces every cell: the median error over all 160 runs is 0.022 in both, and no cell moves by more than the seed scatter. For a scalar whose data gradient is strong, Adam's normalisation makes the decay term negligible.

**Rerun without the network's regularization (`--clean`, 2026-11-04).** D01 then found that the 2 % floor was the packaged weight decay ($5 \times 10^{-4}$) and gradient clipping (norm 1) *on the network*. The full grid with both off (`--clean --out results/clean`, everything else identical) is the table that now stands:

![F01 clean](../../docs/assets/benchmarks/f01_heat_alpha_clean.png)

| Noise σ | $N_{\text{obs}}$ | Rel. error of α | Spread | CR bound | Bias | Rel. L2 of u |
|---|---|---|---|---|---|---|
| 0 | 20 | 0.0010 [0.0005, 0.0016] | 0.002 | 0 | −0.0014 | 0.0019 |
| | 50 | 0.0015 [0.0006, 0.0021] | 0.001 | 0 | −0.0014 | 0.0019 |
| | 200 | 0.0017 [0.0013, 0.0019] | 0.0005 | 0 | −0.0016 | 0.0017 |
| | 1000 | 0.0018 [0.0016, 0.0021] | 0.004 | 0 | −0.0032 | 0.0018 |
| 0.01 | 20 | 0.0098 [0.0037, 0.0148] | 0.009 | 0.011 | −0.0076 | 0.0052 |
| | 50 | 0.0073 [0.0045, 0.0080] | 0.008 | 0.007 | +0.0002 | 0.0035 |
| | 200 | 0.0024 [0.0012, 0.0045] | 0.004 | 0.003 | −0.0037 | 0.0023 |
| | 1000 | 0.0022 [0.0021, 0.0023] | 0.002 | 0.002 | −0.0020 | 0.0022 |
| 0.05 | 20 | 0.024 [0.013, 0.069] | 0.040 | 0.053 | −0.035 | 0.014 |
| | 50 | 0.028 [0.011, 0.038] | 0.035 | 0.035 | +0.008 | 0.011 |
| | 200 | 0.011 [0.006, 0.017] | 0.020 | 0.017 | −0.011 | 0.009 |
| | 1000 | 0.0052 [0.0040, 0.0055] | 0.004 | 0.008 | −0.0030 | 0.0028 |
| 0.1 | 20 | 0.044 [0.034, 0.147] | 0.079 | 0.106 | −0.079 | 0.029 |
| | 50 | 0.055 [0.020, 0.078] | 0.073 | 0.070 | +0.017 | 0.025 |
| | 200 | 0.016 [0.008, 0.032] | 0.039 | 0.035 | −0.021 | 0.011 |
| | 1000 | 0.0100 [0.0092, 0.0108] | 0.009 | 0.016 | −0.0048 | 0.0042 |

The median error over the 160 runs falls from 0.022 to 0.0057 and the median field error from 2.8 % to 0.45 %. Every cell improves; the clean-data floor is now 0.1–0.2 % (bias −0.14 to −0.3 %), and the field error that goes with it is 0.2 %. The run time is unchanged (about 85 s per run with four in parallel).

## Conclusion

1. **The 2 % floor was the network's regularization, not the surrogate's capacity.** With the packaged weight decay and gradient clipping, clean or lightly noisy data ($\sigma \le 0.01$) gave 1.5–2 % whatever the number of observations, as a bias (every seed at $\hat\alpha \approx 0.098$) with a 0.5–0.9 % spread. The first reading blamed the PINN's forward error (2–3 % in $u$). D01's ablation attributed it to the two regularizers, and the clean rerun confirms it on the whole grid: with both off, the same Adam run on clean data lands at 0.1–0.2 % with a field error of 0.2 %, and with $\sigma = 0.01$ the error now falls with $N$ along the Cramér–Rao bound (1.0 %, 0.7 %, 0.24 %, 0.22 % against bounds of 1.1 %, 0.7 %, 0.35 %, 0.16 %). The remaining floor is a tenth of the old one and is the forward error that L-BFGS halves again (D01). The lesson for the package: the defaults that help a forward fit (decay, clipping) are priors on the surrogate that the parameter absorbs, and inverse runs should start from `weight_decay=0`, `gradient_clipping=0` (the recipe of J02/J03).
2. **Above the floor the estimator is close to efficient.** For $\sigma \ge 0.05$ the spread of $\hat\alpha$ across seeds matches the Cramér–Rao bound computed from the same observation points to within about 20 % in 7 of 8 cells (clean run: 0.073 vs 0.070 at $\sigma = 0.1$, $N = 50$; 0.009 vs 0.016 at $N = 1,000$, where ten seeds under-resolve the spread), and with the floor gone the same holds at $\sigma = 0.01$. The $\sigma / \sqrt{N}$ law holds, so the Fisher information predicts how many sensors a given accuracy needs. This is the empirical justification for building the conditioning diagnostics of v0.5 on the sensitivity Jacobian (K01).
3. **Few noisy points bias the estimate low, and the clean rerun keeps the bias.** At $N = 20$ the mean bias grows with noise (−3.5 %, −8 % at $\sigma$ = 0.05, 0.1 in the clean run; −3 %, −5 %, −9 % with the regularizers) and the IQR becomes skewed, with a heavy tail toward small $\alpha$ (one seed at 0.078, a 22 % error, for $\sigma = 0.1$, in both runs). This bias is not the floor's: it survives the removal of the regularizers, so it belongs to the data term. A smoother field fits noisy points as well as a sharper one and has a smaller residual, so the physics loss pulls toward slower diffusion. The fixed data weight of 10 does not account for the noise level, which is what K39 (likelihood weighting, $1/\sigma^2$) tests.
4. **The field error does not predict the parameter error.** With the regularizers the relative L2 error of $u$ was 2–4 % in every cell while the median error of $\alpha$ spanned 1.5–5.5 % and single seeds reached 22 %; in the clean run the field error spans 0.2–3 % and the parameter error 0.1–5.5 %, with the 22 % seed still there at a field error of 3 %. Reporting the field error alone, as most PINN inverse-problem papers do, says little about the identified physics.

**Consequences for the backlog:**
- The 2 % floor is gone with the clean configuration (done, above); F02, F03 and F09 were run with the regularizers on and their "surrogate" floors are to be reread with `--clean` (F02's 20 % KdV offset first).
- K01 (Fisher information by autograd) can be validated against the closed-form bound used here before it is applied to problems without one.
- K39 is next in the F track after F02–F05: the low bias at small $N$ and large $\sigma$ is the signature of a misweighted data term.
- F04 (initial-guess sensitivity) can start from this script; the trajectories suggest 5× is benign for the heat equation.
