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

_Pending._

## Conclusion

_Pending._
