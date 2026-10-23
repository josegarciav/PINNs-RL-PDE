# B01 / B06: collocation sampling at equal budget

## Question

Do adaptive collocation samplers (RAD, the RL bandit) beat uniform sampling, at equal training steps (B01) and once their own residual evaluations count against the budget (B06)?

## Setup

- **PDEs:**
  - Burgers, $u_t + u u_x = \nu u_{xx}$ with $\nu = 0.01/\pi$ and $u_0 = -\sin(\pi x)$ on $[-1, 1] \times [0, 1]$, Dirichlet 0. Scored against the Cole–Hopf reference (`pinnrl.benchmarks.reference.burgers_reference`).
  - Allen–Cahn, $u_t = 0.001\,u_{xx} + 5(u - u^3)$ with $u_0 = x^2\cos(\pi x)$ and $u(\pm 1) = -1$, solved in rescaled time $\tau = 5t \in [0, 5]$. Scored against the method-of-lines reference (`allen_cahn_reference`).
- **Model and training:** a tanh MLP (4 × 64) with Adam at a constant learning rate of 1e-3, gradient clipping at 1, and loss weights residual / BC / IC = 1 / 10 / 10. Trained for **15,000 steps on 1,000 points** through `PDETrainer.train_steps`.
- **Samplers:**
  - `uniform` and `stratified` draw fresh points every step.
  - `uniform_periodic`, `rad` (k = 1, c = 1, pool of 20 × the batch) and `rl_bandit` refresh every 100 steps.
  - `uniform_periodic` is the control that separates resampling frequency from adaptivity.
- **Seeds:** 10. A seed fixes the initial weights, which are identical across samplers, so comparisons are paired. 50 checkpoints per run record the relative L2 error, the training wall-clock, and the residual evaluations including sampler overhead. RAD spends 20 % more evaluations than uniform; the bandit spends 0.26 % more.
- **Code:** `pinnrl/benchmarks/sampling_study.py` at the commit that adds this folder. The bandit arm was re-run twice. The first re-run followed the dropout fix (`results/rl_bandit_pre_fix_curves.json` keeps the original runs). The second followed seeding its replay batches from torch, so a bandit run now repeats for a given seed (`results/rl_bandit_unseeded_curves.json` keeps the unseeded runs, which reach the same conclusions).

Rerun with:

```bash
pinnrl-benchmark sampling --seeds 10 --steps 15000 --checkpoints 50 --resample-every 100 \
  --workers 4 --out experiments/B01_sampling_equal_compute/results
python experiments/B01_sampling_equal_compute/plot.py
```

## Result

![B01](../../docs/assets/benchmarks/b01_sampling.png)

Relative L2 error, as the median [IQR] over 10 seeds. "Beats uniform" counts paired seeds, with p from a two-sided Wilcoxon signed-rank test.

| PDE | Sampler | Equal steps (B01) | Beats uniform | p | Equal evaluations (B06) | Beats uniform | p |
|---|---|---|---|---|---|---|---|
| Burgers | uniform | 0.055 [0.038, 0.069] | – | – | 0.055 | – | – |
| | stratified | 0.038 [0.030, 0.072] | 6/10 | 0.63 | 0.038 | 6/10 | 0.63 |
| | uniform, every 100 | 0.109 [0.028, 0.136] | 4/10 | 0.16 | 0.109 | 4/10 | 0.16 |
| | RAD | 0.136 [0.050, 0.159] | 3/10 | 0.11 | 0.090 | 2/10 | 0.064 |
| | RL bandit | 0.117 [0.080, 0.132] | 1/10 | 0.064 | 0.105 | 2/10 | **0.014** (uniform better) |
| Allen–Cahn | uniform | 0.568 [0.561, 0.582] | – | – | 0.568 | – | – |
| | stratified | 0.564 [0.562, 0.576] | 5/10 | 0.77 | 0.564 | 5/10 | 0.77 |
| | uniform, every 100 | 0.570 [0.566, 0.572] | 4/10 | 0.92 | 0.570 | 4/10 | 0.92 |
| | RAD | 0.533 [0.529, 0.543] | **10/10** | **0.002** | 0.547 | **10/10** | **0.002** |
| | RL bandit | 0.520 [0.516, 0.523] | **10/10** | **0.002** | 0.518 | **10/10** | **0.002** |

Against the same-cadence control (`uniform_periodic`) on Burgers, RAD is not significantly different (p ≥ 0.13, under both the final error and the median of the last 5 checkpoints). Neither is the bandit by final error (p = 0.77). By the last-5 median it is borderline worse (better on 3/10 seeds, p = 0.049), which does not survive correction for the several comparisons made here. On Allen–Cahn both win on 10/10 seeds (p = 0.002).

## Conclusion

1. **Burgers: no sampler is significantly better than uniform at this budget.** Optimizer noise dominates. Final errors vary 6× across seeds, and the error moves by a factor of about 1.7 between consecutive checkpoints of a single run, because Adam at a constant learning rate never settles. Once its probe cost is charged, the bandit is significantly *worse* than fresh uniform points (p = 0.014).
2. **Allen–Cahn: adaptive sampling is consistently better, but not useful yet.** RAD and the bandit beat uniform on every seed, and the bandit reaches a relative error of 0.7 with about 40 % fewer evaluations than uniform (4.8 M vs 8.4 M). But every method stays in the failure regime (relative L2 ≈ 0.52–0.57). The PINN converges to the wrong, metastable solution whichever sampler is used.
3. **Resampling frequency is a confound.** On Burgers, uniform refreshed every 100 steps is worse than fresh points every step (median 0.109 vs 0.055, n.s.). Part of any "adaptive vs uniform" difference reported in the literature may come from resampling cadence rather than adaptivity.

**Consequences for the backlog:**
- Compare samplers only after the optimizer settles: a decaying learning rate or an L-BFGS finish (D01, D02).
- Measure resampling frequency on its own (B04).
- Fix the Allen–Cahn failure mode first, with time-marching or causal weighting (H03, H04, B09), and then re-ask whether sampling matters.
- RAD stays the recommended adaptive sampler. The bandit stays experimental: its only advantage is early in training on Allen–Cahn (C01).
