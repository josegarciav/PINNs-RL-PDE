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

## Conclusion

1. **The error has two regimes.** With clean or lightly noisy data ($\sigma \le 0.01$) the error of $\hat\alpha$ is 1.5–2 % whatever the number of observations, from 20 to 1,000. It is a **bias** (every seed lands at $\hat\alpha \approx 0.098$), not scatter: the spread across seeds is 0.5–0.9 %. The data carry far more information than that (the Cramér–Rao bound reaches 0.2 % at 1,000 points), so the floor is the PINN's own: the surrogate's field error of 2–3 % is absorbed by $\alpha$. More data cannot fix this; a more accurate forward solve can (a longer or two-stage training, L-BFGS, hard boundary constraints).
2. **Above the floor the estimator is close to efficient.** For $\sigma \ge 0.05$ the spread of $\hat\alpha$ across seeds matches the Cramér–Rao bound computed from the same observation points to within about 20 % in 7 of 8 cells (e.g. 0.065 vs 0.070 at $\sigma = 0.1$, $N = 50$; 0.012 vs 0.016 at $N = 1,000$). The $\sigma / \sqrt{N}$ law holds, so the Fisher information predicts how many sensors a given accuracy needs. This is the empirical justification for building the conditioning diagnostics of v0.5 on the sensitivity Jacobian (K01).
3. **Few noisy points bias the estimate low.** At $N = 20$ the mean bias grows with noise (−3 %, −5 %, −9 %) and the IQR becomes skewed, with a heavy tail toward small $\alpha$ (one seed at 0.078, a 22 % error, for $\sigma = 0.1$). A smoother field fits noisy points as well as a sharper one and has a smaller residual, so the physics loss pulls toward slower diffusion. The fixed data weight of 10 does not account for the noise level, which is what K39 (likelihood weighting, $1/\sigma^2$) tests.
4. **The field error does not predict the parameter error.** The relative L2 error of $u$ is 2–4 % in every cell, while the median error of $\alpha$ spans 1.5–5.5 % and single seeds reach 22 %. Reporting the field error alone, as most PINN inverse-problem papers do, says little about the identified physics.

**Consequences for the backlog:**
- The 2 % floor is a forward-accuracy problem: run F01 again after D01/D02 (learning-rate decay, L-BFGS finish) and report the floor against the forward error.
- K01 (Fisher information by autograd) can be validated against the closed-form bound used here before it is applied to problems without one.
- K39 is next in the F track after F02–F05: the low bias at small $N$ and large $\sigma$ is the signature of a misweighted data term.
- F04 (initial-guess sensitivity) can start from this script; the trajectories suggest 5× is benign for the heat equation.
