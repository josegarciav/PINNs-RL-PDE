# D01 / D02: optimizer baseline, the switch point, and the floor

## Question

Adam vs L-BFGS vs Adam→L-BFGS (D01). When should the switch happen (D02)? And, the question these were queued for: is the 2 % floor of the identified heat diffusivity (F01) a property of the forward optimisation?

## Setup

- **Part A, forward, at matched wall-clock time.** Heat (closed form), Burgers at $\nu = 0.01/\pi$ (Cole–Hopf reference) and Allen–Cahn (method-of-lines reference), the sampling study's setups; a tanh MLP 4 × 64, 1,000 collocation points refreshed every 50 optimizer steps, 512 boundary and initial points, weights 1 / 10 / 10, **no weight decay and no gradient clipping**. Arms: Adam at $10^{-3}$; L-BFGS (full batch, strong Wolfe, 20 inner iterations per step, lr 1); Adam then L-BFGS, switching at a fraction $r \in \{0.3, 0.5, 0.7, 0.9\}$ of the budget. Every arm gets 120 s on two threads and is scored at 15, 30, 60 and 120 s. 3 seeds.
- **Part B, the floor.** F01's cell (heat $\alpha$, 200 observations, noise 0.01, $\alpha$ started at 0.5, data weight 10) with the same clean configuration: Adam 5,000 steps; Adam 3,500 steps then L-BFGS 75 steps (about the same number of residual evaluations); L-BFGS alone, 250 steps. 5 seeds.
- **Ablation of the defaults** (`scratch`, 2 seeds, 5,000 Adam steps on the same cell): F01's `build_trainer` defaults (network weight decay $5 \times 10^{-4}$, gradient clipping 1.0) with each switched off.
- **Code:** `run.py`.

Rerun with `python experiments/D01_optimizer_baseline/run.py --seeds 3 --floor-seeds 5 --budget 120 --workers 4`.

## Result

**Part B and the ablation.** Relative error of $\hat\alpha$ (median [min, max] over 5 seeds), spread across seeds, mean bias, field error:

| Arm (clean configuration) | Rel. error of $\hat\alpha$ | Spread | Bias | Rel. L2 of $u$ | Time |
|---|---|---|---|---|---|
| Adam, 5,000 steps | 0.0041 [0.0012, 0.0124] | 0.0052 | −0.0056 | 0.0023 | 90 s |
| Adam 3,500 → L-BFGS 75 | 0.0021 [0.0001, 0.0103] | 0.0049 | −0.0034 | 0.0012 | 78 s |
| L-BFGS, 250 steps | 0.0024 [0.0001, 0.0101] | 0.0050 | −0.0035 | 0.0011 | 37 s |

| Defaults ablation (2 seeds) | Rel. error of $\hat\alpha$ | Rel. L2 of $u$ |
|---|---|---|
| F01 defaults (weight decay 5e-4, clipping 1.0) | 0.026 / 0.016 | 0.024 / 0.019 |
| weight decay 0 | 0.010 / 0.004 | 0.017 / 0.020 |
| clipping 0 | 0.018 / 0.003 | 0.004 / 0.016 |
| both 0 | 0.012 / 0.002 | 0.0025 / 0.0023 |

**Part A.** Median relative L2 error over 3 seeds at 15 / 30 / 60 / 120 s, and the optimizer steps taken in 120 s:

| Problem | Arm | 15 s | 30 s | 60 s | 120 s | Steps |
|---|---|---|---|---|---|---|
| heat | Adam | 0.005 | 0.003 | 0.002 | 0.001 | 7,225 |
| heat | L-BFGS | 0.001 | 0.0004 | 0.0003 | 0.0002 | 4,364 |
| heat | Adam→L-BFGS, r = 0.3 / 0.5 / 0.7 / 0.9 | 0.004 | 0.003 | 0.0004–0.008 | 0.0002–0.0004 | 5,600–6,300 |
| Burgers | Adam | 0.376 | 0.337 | 0.298 | 0.122 | 6,863 |
| Burgers | L-BFGS | 0.091 | 0.060 | 0.087 | 0.068 | 284 |
| Burgers | Adam→L-BFGS, r = 0.9 | 0.370 | 0.343 | 0.136 | 0.061 | 6,561 |
| Burgers | Adam→L-BFGS, r = 0.3 / 0.5 | 0.370 | 0.325 | 0.22 / 0.14 | diverged in 3 of 6 runs | |
| Allen–Cahn | Adam | 1.17 | 1.17 | 1.13 | 1.01 | 7,194 |
| Allen–Cahn | L-BFGS | 0.68 | 0.56 | 0.52 | 0.52 | 466 |
| Allen–Cahn | Adam→L-BFGS, any r | 1.17 | 1.17 | 0.55–1.14 | 0.52–0.57 | |

## Conclusion

1. **The 2 % floor of F01–F09 was the packaged regularisation, not the surrogate.** With the network's weight decay and gradient clipping off, the same Adam run identifies $\alpha$ to 0.4 % (median) with a field error of 0.23 %, ten times better than F01's 2 % / 2.4 %; the ablation attributes about half of the floor to each setting. Weight decay pulls the network's weights toward zero and clipping at norm 1 slows a 4 × 64 network whose gradients are larger than that; both leave a systematic field error that the parameter absorbs. The remaining bias is −0.3 to −0.6 %, and the spread across seeds (0.5 %) now matches the Cramér–Rao bound of this cell (0.3 %) within what five seeds resolve.
2. **L-BFGS is the forward-accuracy lever.** On heat it reaches $2 \times 10^{-4}$ in 120 s against Adam's $10^{-3}$, and in the floor cell it halves the error and the bias again (0.2 %, field error 0.1 %) in a third of the time. On Burgers it is 3 times better than Adam at 15 s and twice at 120 s; on Allen–Cahn it halves the error that Adam cannot reduce at all in 120 s (the known failure case, H03).
3. **The switch point hardly matters when both phases work, and an early switch can diverge.** On heat every $r$ ends at L-BFGS's accuracy. On Burgers $r = 0.9$ is the best arm (0.061) but $r = 0.3$ and $0.5$ produced NaN in 3 of 6 runs: the L-BFGS line search on a network that Adam has not yet smoothed can step to an unbounded loss. A safeguard (reject a step that increases the loss and fall back to Adam) belongs in the trainer before `adam_lbfgs` becomes a default.

**Consequences for the backlog:** `build_trainer` keeps the packaged defaults, but every inverse experiment should run with `weight_decay=0`, `gradient_clipping=0` (F01's `--clean`, rerun in progress) and preferably an L-BFGS finish; F02's 20 % off optimum, F09's bump, K48's biased coefficients and F11's early change point are all to be re-read after that rerun. D03 (adaptive weights) and D10 (preconditioning) stay open; the L-BFGS safeguard is a new engineering item.
