# K39: likelihood weighting

## Question

Does setting the data weight from the noise model beat tuned weights?

## Setup

- **Problem:** the F01 heat identification with 200 observations at noise $\sigma \in \{0.01, 0.05, 0.1\}$, $\alpha$ started at 5× the truth, 10 seeds.
- **Three data weights:** the fixed 10 used throughout F01–F05; a fixed 100; and the likelihood weight $(N_d / \sigma^2) / (N_r / \sigma_r^2)$ with $N_d = 200$ observations, $N_r = 1000$ collocation points and a model-error scale $\sigma_r = 0.1$ for the residual (`pinnrl.uq.likelihood_weights`), which gives 20, 0.8 and 0.2 at the three noise levels: a weight that *falls* with the noise, unlike a tuned constant.
- **Training:** as in F01 (4 × 64 tanh MLP, 5,000 Adam steps, residual / BC / IC weights 1 / 10 / 10).
- **Metrics:** the relative error of $\hat\alpha$ (median, IQR), the bias and spread across seeds, the residual and data losses, the field error.
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/K39_likelihood_weighting/run.py --seeds 10 --steps 5000 --workers 4
```

## Result

Relative error of $\hat\alpha$ (median [IQR] over 10 seeds), spread and bias across seeds, the residual and data losses of the trained network, and the field error.

| Noise σ | Data weight | Rel. error of α | Spread | Bias | Residual loss | Data loss | Rel. L2 of u |
|---|---|---|---|---|---|---|---|
| 0.01 | 10 (fixed) | 0.019 [0.015, 0.020] | 0.005 | −0.019 | 8.1e-4 | 2.1e-4 | 0.028 |
| | 100 (fixed) | 0.036 [0.030, 0.041] | 0.006 | −0.036 | 2.7e-3 | 2.1e-4 | 0.028 |
| | 20 (likelihood) | 0.022 [0.020, 0.025] | 0.005 | −0.022 | 1.2e-3 | 2.2e-4 | 0.028 |
| 0.05 | 10 (fixed) | 0.021 [0.013, 0.032] | 0.019 | −0.027 | 8.6e-4 | 2.7e-3 | 0.029 |
| | 100 (fixed) | 0.037 [0.032, 0.055] | 0.030 | −0.047 | 4.4e-3 | 2.5e-3 | 0.032 |
| | 0.8 (likelihood) | 0.064 [0.052, 0.089] | 0.018 | −0.070 | 6.0e-4 | 2.8e-3 | 0.025 |
| 0.1 | 10 (fixed) | 0.024 [0.009, 0.046] | 0.037 | −0.038 | 1.1e-3 | 1.0e-2 | 0.024 |
| | 100 (fixed) | 0.048 [0.034, 0.092] | 0.066 | −0.072 | 1.3e-2 | 9.7e-3 | 0.048 |
| | 0.2 (likelihood) | 0.202 [0.183, 0.232] | 0.029 | −0.206 | 5.5e-4 | 1.2e-2 | 0.025 |

## Conclusion

1. **The weight from the noise model is the worst of the three.** With the residual treated as Gaussian model error of scale $\sigma_r = 0.1$, the data weight falls to 0.8 at $\sigma = 0.05$ and 0.2 at $\sigma = 0.1$, and the bias of $\hat\alpha$ grows to −7 % and −21 %: a weak data term leaves the estimate to the physics term, whose minimum is the surrogate's floor (F01), not the truth. The Gaussian likelihood is the right accounting for *noise*, but the residual of a trained PINN is not noise. It is a systematic approximation error, correlated across the domain, with a minimum displaced from the true parameters; weighting it as if it were independent Gaussian error overstates how much the physics term should be trusted.
2. **Too much data weight hurts the other way.** At a fixed 100 the data term chases the noise: the residual loss rises 3–12×, the field error doubles at $\sigma = 0.1$, and the bias doubles. The data term then reports the noise draw more than the physics.
3. **The tuned 10 sits at the balance, and the balance is set by the surrogate, not by the noise.** The right weight is the one at which the data pull on $\alpha$ just overcomes the surrogate's systematic preference without fitting noise. In F02, where the floor was 20 %, a weight of 100 was the improvement; here, where the floor is 2 %, 10 is. The weight should therefore be set from the *surrogate's* error model (its floor relative to the noise), which the likelihood recipe does not know.
4. **For calibration the recipe still matters.** K10 reads the Laplace width off the Fisher information with the noise $\sigma$, which is independent of the loss weight; what the weight changes is the bias, and that is what the F06 coverage shortfall measures.

**Consequences for the backlog:**
- The Bayesian weighting of the [theory](../../docs/theory/bayesian.md) needs a model-error term for the surrogate (a residual scale learned from the data, or an iterative reweighting), which is the honest version of K39 and belongs to K41/K43.
- The data weight stays a documented choice in `pinnrl`, with 10 as the default and the F02 rule of thumb: raise it when the field error is large compared with the noise.
