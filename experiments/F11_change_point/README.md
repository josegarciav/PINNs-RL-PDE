# F11: change-point identification

## Question

Can the time of a jump in a coefficient be identified as a parameter?

## Setup

Heat with $\alpha(t) = a + b\,\sigma((t - t_0)/0.02)$, truth $(a, b, t_0) = (0.05, 0.1, 0.5)$ (F09's step), the three trainable and started at $(0.1, 0.05, 0.3)$ or $(0.1, 0.05, 0.7)$; 200 observations, noise 0, 0.01, 0.05; 5 seeds; 10,000 Adam steps; against F09's network unknown (same training) and the Cramér–Rao bound on $(a, b, t_0)$ from the closed form (`crlb.json`). Two configurations: the F01 defaults (network weight decay $5 \times 10^{-4}$, gradient clipping 1; `results/`), and the clean one (`--clean --lbfgs`: both off, an L-BFGS finish of 100 steps; `results/clean/`). `run.py`.

Rerun with:

```bash
python experiments/F11_change_point/run.py --seeds 5 --steps 10000 --workers 5 --clean --lbfgs --out experiments/F11_change_point/results/clean
```

## Result

Clean configuration (median over 5 seeds; the $t_0$ error with its range):

| Model | Noise | Rel. L2 of $\hat\alpha$ | $t_0$ error, median [min, max] | CR std of $t_0$ | $\hat a$ | $\hat b$ (CR std) | Rel. L2 of $u$ |
|---|---|---|---|---|---|---|---|
| change point, from 0.3 | 0 | 0.078 | 0.0023 [0.0010, 0.0025] | 0 | 0.0497 | 0.102 (0) | 0.0019 |
| | 0.01 | 0.085 | 0.0075 [0.0046, 0.0135] | 0.0062 | 0.0495 | 0.099 (0.002) | 0.0033 |
| | 0.05 | 0.136 | 0.032 [0.024, 0.086] | 0.031 | 0.0496 | 0.092 (0.011) | 0.012 |
| change point, from 0.7 | 0 | 0.078 | 0.0021 [0.0011, 0.0026] | 0 | 0.0496 | 0.102 | 0.0018 |
| | 0.01 | 0.085 | 0.0072 [0.0047, 0.0133] | 0.0062 | 0.0496 | 0.099 | 0.0033 |
| | 0.05 | 0.134 | 0.031 [0.024, 0.087] | 0.031 | 0.0496 | 0.092 | 0.012 |
| network (F09) | 0 | 0.093 | – | | | | 0.0023 |
| | 0.01 | 0.119 | – | | | | 0.0054 |
| | 0.05 | 0.229 | – | | | | 0.018 |

Default configuration (the first pass): $\hat t_0$ 0.13–0.18 early from either start at every noise level (25 times the bound at noise 0.01), $\hat b$ 0.055–0.057 (half the truth), relative L2 of $\hat\alpha$ 0.32–0.36 against the network's 0.23–0.27, field error 3.6 %.

## Conclusion

1. **A change point is identifiable to the Cramér–Rao bound, once the regularization is off.** With noise 0.01 the median error of $\hat t_0$ is 0.0075 against a bound of 0.0062, with noise 0.05 it is 0.032 against 0.031, from either side of the truth; $\hat b$ is at the truth within its bound and $\hat a$ within 1 %. On clean data the error is 0.002, the surrogate's residual floor (field error 0.2 %). The parametrised step beats the network's ramp at every noise level (0.08 against 0.09–0.23 in L2) because its three numbers carry the sharp feature a smooth network spreads (F09).
2. **The first pass was the regularization, not the estimator.** Under the default weight decay and gradient clipping the surrogate's 3.6 % field error was absorbed into a smaller jump placed early, 25 times the bound, and three times the training did not help. D01's floor finding, F01's clean grid and this rerun agree: the inverse recipe is `weight_decay=0`, `gradient_clipping=0`, an L-BFGS finish.
3. **The rel. L2 of the field unknown has a floor of 0.08 at the step** that the parameters do not see: the sigmoid of width 0.02 is compared on a grid of 121 points, and a $t_0$ error of 0.002 moves the step by a tenth of its width, which costs 8 % in L2. The parameter errors are the honest numbers.

**Consequences for the backlog:** F02 (KdV pair) and F03 (Burgers ν) are the remaining first-pass results under the default configuration; F09/K48 field results likewise. A change point in the evaporator's fouling (a cleaning event) is the process instance (J-series).
