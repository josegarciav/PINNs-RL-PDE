# F11: change-point identification (first pass, under the default training configuration)

## Question

Can the time of a jump in a coefficient be identified as a parameter?

## Setup

Heat with $\alpha(t) = a + b\,\sigma((t - t_0)/0.02)$, truth $(a, b, t_0) = (0.05, 0.1, 0.5)$ (F09's step), the three trainable and started at $(0.1, 0.05, 0.3)$ or $(0.1, 0.05, 0.7)$; 200 observations, noise 0, 0.01, 0.05; 5 seeds; 10,000 Adam steps with the F01 defaults (network weight decay $5 \times 10^{-4}$, gradient clipping 1); against F09's network unknown and the Cramér–Rao bound on $(a, b, t_0)$ from the closed form. `run.py`, results in `results/`.

## Result

| Model | Noise | Rel. L2 of $\hat\alpha$ | $t_0$ error, median [min, max] | $\hat a$ | $\hat b$ | CR std of $t_0$ |
|---|---|---|---|---|---|---|
| change point, from 0.3 | 0.01 | 0.356 | 0.179 [0.089, 0.185] | 0.049 | 0.055 | 0.006 |
| change point, from 0.7 | 0.01 | 0.343 | 0.163 [0.060, 0.171] | 0.049 | 0.056 | 0.006 |
| network (F09) | 0.01 | 0.260 | – | – | – | – |

Noise 0 and 0.05 give the same picture. $\hat a$ is right, $\hat b$ is half the truth and $\hat t_0$ settles 0.1–0.2 early from either start, 25 times the Cramér–Rao bound; the parametrised step fits worse than the network's ramp.

## Conclusion (provisional)

The data determine $t_0$ to 0.006 (the bound), the trained estimator lands 0.16 off: the misfit between a smooth surrogate and a step solution is absorbed into a smaller jump ($b$) placed early, the same surrogate bias as F02 and K48. D01's floor ablation (same day) found that the default network weight decay and gradient clipping cost an order of magnitude of accuracy on the scalar heat problem; **this experiment must be rerun with `weight_decay=0`, `gradient_clipping=0` and an L-BFGS finish before its conclusion stands.** Its driver takes those as the next step.
