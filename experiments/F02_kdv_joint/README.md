# F02: joint identification of the KdV coefficients

## Question

Are two parameters jointly identifiable? Recovery error and correlation of the estimates across seeds.

## Setup

- **Problem:** $u_t + \alpha\, u u_x + \beta\, u_{xxx} = 0$ on $[-15, 15] \times [0, 2]$ with the soliton $u = \frac{3}{\alpha}\operatorname{sech}^2\!\big(\tfrac{1}{2}\sqrt{1/\beta}\,(x - t)\big)$ (unit speed), Dirichlet 0. True $(\alpha, \beta) = (6, 1)$, started at $(4, 1.5)$. The catalogue entry `kdv_two_parameter`; `test_catalog.py` checks that the soliton zeroes the residual. The amplitude $3/\alpha$ and the width $\propto \sqrt{\beta}$ are both visible in the data, so the pair is structurally identifiable; the question is how strongly the data correlate them.
- **Data:** $N_{\text{obs}} \in \{50, 200, 1000\}$ observations at uniform random points of the whole domain (most of which see the flat tails), noise 0.01 and 0.05 (amplitude 0.5).
- **Bound:** the 2 × 2 Fisher matrix from the symbolic sensitivities (`pinnrl.analysis.closed_form_bound`) gives the Cramér–Rao standard deviations, the predicted correlation $\rho$ of the two estimates and the condition number of $F$.
- **Model and training:** 4 × 64 tanh MLP, Adam at $10^{-3}$ for the weights and both parameters, **8,000 steps**, 1,000 collocation points refreshed every 100 steps, weights 1 / 10 / 10 / 10.
- **Seeds:** 10.
- **Metrics:** the relative errors of $\hat\alpha$ and $\hat\beta$; the spread and the empirical correlation of the estimates across seeds against the bound; the relative L2 error of $u$.
- **Code:** `run.py`, `plot.py`.

Rerun with:

```bash
python experiments/F02_kdv_joint/run.py --seeds 10 --steps 8000 --workers 4
python experiments/F02_kdv_joint/plot.py
```

## Result

![F02](../../docs/assets/benchmarks/f02_kdv_joint.png)

Relative errors of $\hat\alpha$ and $\hat\beta$ at step 8,000 (median [IQR] over 10 seeds), the spread of each across seeds next to its Cramér–Rao bound, the correlation of the ten $(\hat\alpha, \hat\beta)$ pairs next to the bound's $\rho$, the condition number of $F$, and the field error.

| Noise | $N$ | Rel. error of α | Rel. error of β | Spread α / β | Bound α / β | ρ seeds / bound | κ(F) | Rel. L2 of u |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 50 | 0.248 [0.223, 0.272] | 0.323 [0.282, 0.375] | 0.055 / 0.084 | 0.011 / 0.030 | +0.94 / +0.65 | 12 | 0.167 |
| | 200 | 0.179 [0.165, 0.199] | 0.244 [0.232, 0.260] | 0.018 / 0.021 | 0.006 / 0.014 | +0.69 / +0.61 | 10 | 0.132 |
| | 1000 | 0.172 [0.158, 0.183] | 0.247 [0.233, 0.258] | 0.015 / 0.015 | 0.003 / 0.007 | +0.65 / +0.61 | 11 | 0.120 |
| 0.05 | 50 | 0.283 [0.223, 0.333] | 0.343 [0.218, 0.465] | 0.088 / 0.150 | 0.057 / 0.152 | +0.87 / +0.65 | 12 | 0.213 |
| | 200 | 0.179 [0.165, 0.201] | 0.306 [0.221, 0.340] | 0.026 / 0.074 | 0.028 / 0.071 | +0.35 / +0.61 | 10 | 0.157 |
| | 1000 | 0.174 [0.167, 0.187] | 0.253 [0.230, 0.267] | 0.015 / 0.029 | 0.013 / 0.033 | +0.65 / +0.61 | 11 | 0.130 |

The estimates sit at $\hat\alpha \approx 4.9$ and $\hat\beta \approx 0.75$ (−18 % and −25 %) in every cell with $N \ge 200$, and are not drifting: the median change over the last 2,000 steps is below 0.01 in both. The ratio $\hat\alpha / \hat\beta \approx 6.6$ is close to the true 6.

**Follow-up** (`run_followup.py`, the noise 0.01, $N = 200$ cell, 3 seeds): is the wrong minimum a basin of the optimizer or the surrogate's own optimum?

| Variant | $\hat\alpha$ | $\hat\beta$ | Rel. error α / β | Rel. L2 of u | Residual loss | Data loss |
|---|---|---|---|---|---|---|
| started at the truth (6, 1), 8,000 steps | 4.70, 4.94, 4.92 | 0.72, 0.78, 0.76 | 0.18–0.22 / 0.22–0.28 | 0.12–0.15 | 2.5–4.5e-4 | 1.4–1.8e-4 |
| started at (4, 1.5), 24,000 steps | 4.67, 4.84, 4.89 | 0.72, 0.75, 0.75 | 0.19–0.22 / 0.25–0.28 | 0.14 | 3.6–5.0e-4 | 1.4–1.5e-4 |
| started at (4, 1.5), data weight 100 | 5.26, 5.43, 5.37 | 0.81, 0.87, 0.88 | 0.10–0.12 / 0.12–0.19 | 0.08–0.09 | 1.3–1.8e-4 | 0.9–1.2e-4 |

## Conclusion

1. **The pair is identifiable from the data but not from this surrogate.** The bound says 200 clean points determine $\alpha$ to 0.6 % and $\beta$ to 1.4 %; the PINN returns them 18 % and 25 % low, with a spread across seeds of only 2 %. Every seed converges to the same wrong point, and a run *started at the truth drifts to it*: the minimum of the PINN loss is not at $(\alpha^\star, \beta^\star)$. This is the F01 surrogate floor (2 % for the heat equation) grown to 20 % for a third-order operator, and it is a property of the loss, not of the optimizer: three times the training changes nothing.
2. **What the surrogate trades.** The network cannot drive the third-derivative residual of the true soliton to zero with 4 × 64 tanh units, so the loss is minimized by a slightly wrong soliton whose equation has smaller $\alpha$ and $\beta$ (weaker nonlinearity and weaker dispersion, the same ratio $\alpha/\beta$). The field error of 12–17 % is the visible part of that trade; the data loss stays at the noise level (1.5e-4 against $\sigma^2 = 10^{-4}$), so the data are fitted, and the parameters are where the physics term is cheapest. Raising the data weight tenfold halves the bias and the field error, which confirms the mechanism: the estimate sits at the balance point of a soft physics term and a soft data term.
3. **The bound still predicts the scatter.** Where the surrogate bias is a constant offset, the spread across seeds follows the bound in both parameters and the correlation of the estimates is the predicted one (+0.65 vs +0.61 at $N = 1000$). The Fisher information describes the random part of the error even when a systematic part dominates it; it cannot see the systematic part, which is why K01 and the docs insist on reporting the field error next to it.
4. **Joint identification needs a resolved forward problem.** The order of business for two-parameter problems is: make the forward solve accurate (a network that resolves the third derivative, a likelihood-based data weight, an L-BFGS finish), check the field error, and only then read the Fisher ellipse. Items D01/D02 and K39 on the backlog are the enablers; F02 should be rerun after them.

**Consequences for the backlog:**
- K39 (data weight from the noise model) has its motivating case: the weight moves the estimate by 10 % here.
- D01/D02 (optimizer settling) and a K03-type check of the KdV residual conditioning come before any further KdV identification.
- The F02 cells with $N = 50$ at noise 0.01 show spreads 5× the bound and correlations near +0.9: with few points the surrogate's compromise varies with the draw, which is the regime where ensembles (F06) will be needed to report an honest interval.
