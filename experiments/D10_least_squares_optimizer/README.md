# D10: a least-squares optimizer for physics-informed losses

**Question.** How far below Adam and L-BFGS does an optimizer get, per second on one CPU core, when it uses the fact that a PINN loss is a sum of squares? And which of the first ideas of [PLAN.md](PLAN.md) add to that?

$$
L(\theta) = \tfrac12 \lVert r(\theta) \rVert^2, \qquad J = \frac{\partial r}{\partial \theta}, \qquad \delta = -(J^\top J + \lambda I)^{-1} J^\top r
$$

- **$r$**: the stacked residuals of the equation, the initial condition and the boundary condition, one row per point.
- **$\lambda$**: the damping. Large $\lambda$ gives a short gradient step, small $\lambda$ a Gauss–Newton step.
- **$\varepsilon(T)$**: the relative $L_2$ error against the exact or reference solution after $T$ seconds of optimization.

## Setup

- **Equations**: heat ($u_t = 0.1\,u_{xx}$), Burgers ($\nu = 0.01/\pi$), the KdV soliton.
- **Network**: 3 hidden layers of 32, tanh, 2,241 weights, inputs scaled to $[-1, 1]$.
- **Rows**: 2,000 collocation points, 256 initial, 256 boundary, redrawn every few steps. Weights 10 on the initial and boundary terms.
- **Machine**: Apple M4, one thread, double precision, plain PyTorch. 60 s per run, 3 seeds, nothing else running.
- **Clock**: optimization time only. Reading the error is not charged.

Methods:

1. **Levenberg–Marquardt**: dense $J$ by one gradient per point, Cholesky, Nielsen's damping rule.
2. **M1**: the same step with $\lambda$ set from the linearization error, plus one correction that reuses the factorization.
3. **M4**: block Kaczmarz. Blocks of 32 or 128 rows drawn in proportion to their squared residual, each solved exactly.
4. **M5**: variable projection. The last layer is solved by linear least squares at every step; the hidden weights step on the projected Jacobian. Two starts: from the first step, and after 60 plain steps.
5. **M6**: continuation. Levenberg–Marquardt while the viscosity moves from 0.1 to the target in 4, 8 or 16 geometric stages. Burgers only.
6. **M3**: training as filtering. An extended Kalman filter over a stream of fresh batches of 64 or 256 rows, never revisited, carrying the full covariance $P$ of the weights:
   $K = P J^\top (J P J^\top + I)^{-1}$, $\theta \leftarrow \theta - K r$, $P \leftarrow (P - K J P)/0.995$.
   - **Hybrid**: Levenberg–Marquardt for 20 s, then the filter from those weights with $P$ at its prior.
   - **Covariance per layer**: $P$ kept only within each layer (4 blocks, the largest 1,056 weights), one shared $J P J^\top + I$.
   - **Covariance per neuron**: $P$ kept only among the weights that enter one neuron (97 blocks of at most 33).
7. **M8**: multipliers. Initial and boundary rows as constraints with an augmented Lagrangian at penalty 1, inside Levenberg–Marquardt. Its control is Levenberg–Marquardt at loss weights 1.
8. **Baselines**: Adam at $10^{-3}$, L-BFGS (strong Wolfe), Adam for 30 % of the budget then L-BFGS.

```bash
D=experiments/D10_least_squares_optimizer
python $D/run.py --budget 60 --seeds 3 --methods lm flow block32 block128 varpro varpro60 \
    continue4 continue8 continue16 filter64 filter256 al lm_w1 adam lbfgs adam_lbfgs
python $D/run.py --budget 180 --seeds 3 --out $D/results/long \
    --methods lm filter64 lmfilter20 dfilter64layer dfilter64node
python $D/plot.py && python $D/plot.py $D/results/long/runs_3x32.json
```

## Results

Median $\varepsilon(T)$ over 3 seeds, with the range.

- **Last 10 s, mean**: the geometric mean of the readings in the last sixth of the run. On Burgers one reading can differ from the next by a factor of 5 as the points are redrawn, so this column is the steadier summary.

### 60 seconds

| Equation | Method | 10 s | 30 s | 60 s | Last 10 s, mean | Steps |
|---|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 2.7e-05 (2e-05–4e-05) | 2.8e-06 (1e-06–5e-06) | 6.2e-07 (5e-07–2e-06) | 8.6e-07 | 421 |
| heat | M1 damping set by the linearization error | 4.5e-05 (4e-05–8e-05) | 3.5e-06 (3e-06–4e-06) | 1.6e-06 (7e-07–2e-06) | 1.8e-06 | 358 |
| heat | M5 last layer solved, after 60 plain steps | 7.6e-05 (7e-05–2e-04) | 2.9e-08 (2e-08–3e-08) | 4.9e-09 (3e-09–6e-09) | 5.4e-09 | 249 |
| heat | M5 last layer solved, from the start | 3.8e-07 (1e-07–5e-07) | 3.2e-08 (2e-08–5e-08) | 1.1e-08 (9e-09–1e-08) | 1.2e-08 | 346 |
| heat | M4 blocks of 128 rows | 3.2e-04 (2e-04–5e-04) | 3.1e-04 (2e-04–5e-04) | 2.3e-04 (9e-05–3e-04) | 3.0e-04 | 7491 |
| heat | M4 blocks of 32 rows | 1.5e-03 (1e-03–2e-03) | 1.7e-03 (1e-03–3e-03) | 1.5e-03 (1e-03–2e-03) | 1.4e-03 | 14189 |
| heat | M3 Kalman filter, batches of 64 | 2.3e-04 (2e-04–3e-04) | 1.5e-05 (1e-05–2e-05) | 4.4e-07 (4e-07–5e-07) | 5.8e-07 | 3713 |
| heat | M3 Kalman filter, batches of 256 | 2.3e-04 (2e-04–3e-04) | 3.7e-05 (4e-05–4e-05) | 4.1e-06 (3e-06–4e-06) | 4.8e-06 | 2032 |
| heat | M8 multipliers, penalty 1 | 4.3e-05 (4e-05–1e-04) | 9.9e-06 (8e-06–1e-05) | 1.7e-06 (2e-06–4e-06) | 2.3e-06 | 508 |
| heat | Levenberg–Marquardt, loss weights 1 | 1.8e-05 (1e-05–3e-05) | 1.8e-06 (1e-06–2e-06) | 4.3e-07 (4e-07–6e-07) | 4.7e-07 | 518 |
| heat | Adam, then L-BFGS | 2.5e-03 (2e-03–3e-03) | 1.0e-04 (1e-04–1e-04) | 7.6e-05 (6e-05–9e-05) | 7.5e-05 | 4283 |
| heat | L-BFGS | 1.4e-04 (1e-04–2e-04) | 8.6e-05 (8e-05–1e-04) | 7.8e-05 (7e-05–1e-04) | 7.8e-05 | 7517 |
| heat | Adam | 2.5e-03 (2e-03–3e-03) | 6.8e-04 (5e-04–2e-03) | 5.7e-04 (4e-04–6e-03) | 8.9e-04 | 14418 |
| burgers | Levenberg–Marquardt | 9.2e-02 (4e-02–2e-01) | 3.0e-03 (3e-03–4e-03) | 4.5e-03 (3e-03–8e-03) | 3.6e-03 | 403 |
| burgers | M1 damping set by the linearization error | 9.8e-02 (4e-02–1e-01) | 1.0e-02 (6e-03–2e-02) | 3.3e-03 (3e-03–7e-03) | 2.7e-03 | 399 |
| burgers | M5 last layer solved, after 60 plain steps | 1.3e-01 (1e-01–2e-01) | 1.9e-02 (1e-02–2e-02) | 5.3e-03 (1e-03–7e-03) | 4.1e-03 | 194 |
| burgers | M5 last layer solved, from the start | 4.9e-01 (5e-01–5e-01) | 4.8e-01 (5e-01–5e-01) | 4.7e-01 (4e-01–5e-01) | 4.7e-01 | 137 |
| burgers | M4 blocks of 128 rows | 1.9e-01 (2e-01–4e-01) | 1.7e-01 (1e-01–3e-01) | 1.4e-01 (1e-01–2e-01) | 1.4e-01 | 7124 |
| burgers | M4 blocks of 32 rows | 4.0e-01 (4e-01–4e-01) | 3.5e-01 (3e-01–4e-01) | 3.6e-01 (3e-01–4e-01) | 3.5e-01 | 13916 |
| burgers | M6 viscosity in 4 stages | 2.2e-01 (2e-01–2e-01) | 1.7e-03 (1e-03–4e-03) | 3.2e-03 (2e-04–7e-03) | 1.2e-03 | 453 |
| burgers | M6 viscosity in 8 stages | 1.4e-01 (1e-01–1e-01) | 2.4e-03 (1e-03–4e-03) | 2.7e-03 (1e-03–3e-03) | 1.7e-03 | 456 |
| burgers | M6 viscosity in 16 stages | 1.8e-01 (2e-01–2e-01) | 2.5e-03 (1e-03–5e-03) | 1.1e-03 (1e-03–3e-03) | 1.3e-03 | 460 |
| burgers | M3 Kalman filter, batches of 64 | 2.3e-01 (2e-01–3e-01) | 4.0e-02 (6e-03–1e-01) | 4.2e-04 (3e-04–1e-03) | 1.7e-03 | 3663 |
| burgers | M3 Kalman filter, batches of 256 | 1.2e-01 (8e-02–2e-01) | 2.5e-02 (1e-02–1e-01) | 1.7e-03 (1e-03–4e-03) | 1.9e-03 | 1976 |
| burgers | M8 multipliers, penalty 1 | 1.5e-01 (1e-01–2e-01) | 1.4e-02 (1e-02–1e-02) | 1.6e-03 (1e-03–2e-03) | 1.9e-03 | 452 |
| burgers | Levenberg–Marquardt, loss weights 1 | 9.8e-02 (5e-02–2e-01) | 9.1e-03 (5e-03–1e-02) | 1.9e-03 (1e-03–2e-03) | 2.0e-03 | 453 |
| burgers | Adam, then L-BFGS | 3.3e-01 (3e-01–4e-01) | 2.7e-02 (2e-02–9e-02) | 1.4e-02 (8e-03–2e-02) | 1.3e-02 | 4108 |
| burgers | L-BFGS | 2.7e-01 (8e-02–4e-01) | 1.4e-01 (6e-02–5e-01) | 2.2e-02 (5e-03–4e-01) | 3.9e-02 | 564 |
| burgers | Adam | 3.3e-01 (3e-01–4e-01) | 1.3e-01 (5e-02–2e-01) | 4.5e-02 (3e-02–1e-01) | 4.2e-02 | 13937 |
| kdv | Levenberg–Marquardt | 2.0e-04 (1e-04–5e-04) | 9.9e-06 (6e-06–1e-05) | 6.0e-06 (3e-06–6e-06) | 6.4e-06 | 365 |
| kdv | M1 damping set by the linearization error | 7.5e-04 (2e-04–2e-03) | 2.9e-05 (3e-05–6e-05) | 1.1e-05 (7e-06–1e-05) | 1.2e-05 | 325 |
| kdv | M5 last layer solved, after 60 plain steps | 4.5e-04 (3e-04–2e-03) | 7.4e-05 (1e-05–9e-05) | 6.6e-06 (4e-06–1e-05) | 7.3e-06 | 140 |
| kdv | M5 last layer solved, from the start | 5.1e-01 (4e-01–6e-01) | 5.0e-01 (4e-01–6e-01) | 4.8e-01 (4e-01–6e-01) | 4.8e-01 | 184 |
| kdv | M4 blocks of 128 rows | 2.7e-03 (2e-03–4e-03) | 2.4e-03 (2e-03–4e-03) | 2.3e-03 (2e-03–4e-03) | 2.3e-03 | 5163 |
| kdv | M4 blocks of 32 rows | 7.9e-03 (5e-03–1e-02) | 5.0e-03 (4e-03–8e-03) | 4.7e-03 (4e-03–6e-03) | 4.8e-03 | 9549 |
| kdv | M3 Kalman filter, batches of 64 | 9.8e-03 (7e-03–1e-02) | 5.0e-04 (3e-04–6e-04) | 1.3e-05 (1e-05–2e-05) | 1.8e-05 | 3047 |
| kdv | M3 Kalman filter, batches of 256 | 7.7e-03 (6e-03–1e-02) | 1.0e-03 (1e-03–1e-03) | 1.1e-04 (8e-05–1e-04) | 1.4e-04 | 1619 |
| kdv | M8 multipliers, penalty 1 | 1.8e-03 (2e-03–4e-03) | 3.9e-04 (2e-04–4e-04) | 9.7e-05 (7e-05–2e-04) | 1.2e-04 | 367 |
| kdv | Levenberg–Marquardt, loss weights 1 | 4.7e-05 (3e-05–1e-04) | 7.1e-06 (3e-06–1e-05) | 2.4e-06 (6e-07–3e-06) | 2.6e-06 | 369 |
| kdv | Adam, then L-BFGS | 1.2e-01 (1e-01–3e-01) | 2.7e-03 (3e-03–3e-03) | 1.3e-03 (1e-03–1e-03) | 1.3e-03 | 1155 |
| kdv | L-BFGS | 4.2e-03 (4e-03–5e-03) | 2.1e-03 (2e-03–3e-03) | 2.0e-03 (2e-03–2e-03) | 2.0e-03 | 1961 |
| kdv | Adam | 7.3e-02 (3e-02–8e-02) | 1.7e-02 (2e-02–2e-02) | 3.8e-03 (3e-03–5e-03) | 4.2e-03 | 8925 |

### 180 seconds

| Equation | Method | 10 s | 30 s | 60 s | 120 s | Last 30 s, mean | Steps |
|---|---|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 2.2e-05 (1e-05–3e-05) | 2.2e-06 (2e-06–4e-06) | 5.7e-07 (5e-07–1e-06) | 1.6e-07 (1e-07–4e-07) | 1.1e-07 | 1428 |
| heat | M3 Kalman filter, batches of 64 | 2.5e-04 (2e-04–3e-04) | 1.4e-05 (1e-05–2e-05) | 4.4e-07 (4e-07–5e-07) | 1.4e-09 (1e-09–5e-09) | 1.5e-09 | 11071 |
| heat | Levenberg–Marquardt for 20 s, then the filter | 2.4e-05 (1e-05–4e-05) | 3.3e-06 (2e-06–3e-06) | 2.0e-06 (2e-06–3e-06) | 5.8e-09 (4e-09–7e-09) | 1.8e-09 | 9841 |
| heat | M3 filter, covariance per layer | 1.1e-04 (1e-04–2e-04) | 2.1e-06 (2e-06–3e-06) | 1.5e-08 (1e-08–2e-08) | 3.8e-09 (3e-09–5e-09) | 1.6e-09 | 17017 |
| heat | M3 filter, covariance per neuron | 5.1e-04 (5e-04–6e-04) | 5.1e-05 (3e-05–7e-05) | 2.6e-05 (7e-06–3e-05) | 7.6e-06 (5e-06–2e-05) | 4.8e-06 | 20780 |
| burgers | Levenberg–Marquardt | 4.3e-02 (3e-02–1e-01) | 4.8e-03 (3e-03–8e-03) | 2.8e-03 (3e-03–5e-03) | 1.3e-03 (9e-04–3e-03) | 1.3e-03 | 1333 |
| burgers | M3 Kalman filter, batches of 64 | 2.3e-01 (2e-01–3e-01) | 3.8e-02 (6e-03–1e-01) | 4.8e-04 (3e-04–1e-03) | 3.8e-05 (3e-05–2e-04) | 2.0e-05 | 11020 |
| burgers | Levenberg–Marquardt for 20 s, then the filter | 5.7e-02 (2e-02–1e-01) | 2.2e-03 (1e-03–7e-03) | 2.1e-04 (2e-04–4e-04) | 3.1e-05 (2e-05–1e-04) | 2.8e-05 | 9729 |
| burgers | M3 filter, covariance per layer | 7.0e-02 (5e-02–1e-01) | 7.7e-03 (5e-03–9e-03) | 1.7e-03 (1e-03–3e-03) | 4.3e-04 (3e-04–8e-04) | 4.0e-04 | 16554 |
| burgers | M3 filter, covariance per neuron | 2.1e-01 (2e-01–3e-01) | 1.2e-01 (3e-02–2e-01) | 1.1e-01 (3e-02–3e-01) | 9.1e-01 (2e-02–9e-01) | 9.8e-01 | 20949 |
| kdv | Levenberg–Marquardt | 1.5e-04 (1e-04–5e-04) | 9.9e-06 (6e-06–1e-05) | 4.8e-06 (2e-06–6e-06) | 1.5e-06 (6e-07–2e-06) | 9.6e-07 | 1071 |
| kdv | M3 Kalman filter, batches of 64 | 9.8e-03 (7e-03–1e-02) | 5.0e-04 (3e-04–6e-04) | 1.3e-05 (1e-05–2e-05) | 2.3e-07 (9e-08–2e-07) | 1.0e-07 | 9029 |
| kdv | Levenberg–Marquardt for 20 s, then the filter | 1.4e-04 (1e-04–5e-04) | 1.5e-05 (1e-05–2e-05) | 1.5e-05 (1e-05–2e-05) | 2.6e-07 (2e-07–5e-07) | 1.1e-07 | 8008 |
| kdv | M3 filter, covariance per layer | 4.4e-03 (3e-03–6e-03) | 1.1e-04 (7e-05–2e-04) | 2.2e-06 (2e-06–4e-06) | 2.9e-08 (2e-08–6e-08) | 4.2e-09 | 12608 |
| kdv | M3 filter, covariance per neuron | 1.1e-02 (7e-03–2e-02) | 1.1e-03 (1e-03–2e-03) | 5.4e-04 (3e-04–1e-03) | 1.1e-04 (1e-04–1e-03) | 9.5e-05 | 14694 |

The figures are `results/runs_3x32.png` and `results/long/runs_3x32.png` after `plot.py`.

## What it shows

1. **The squares are worth two orders of magnitude on smooth problems.** At 60 s Levenberg–Marquardt is 120 times below Adam then L-BFGS on heat and 220 times below on KdV. It passes the plan's stage 0 gate: under $10^{-6}$ on heat on one core.
2. **Burgers is the exception.** The gain is 3 times, and the error moves up and down between 1e-3 and 1e-2 as the points are redrawn. The sharp front, not the optimizer's order, sets the error. This is where continuation (M6) and residual-chosen points have to earn their place.
3. **Solving the last layer is the largest single gain, on the linear equation only.** Heat reaches 4.9e-09, 127 times below Levenberg–Marquardt and four orders below the baselines. On Burgers and KdV it ties Levenberg–Marquardt within the spread.
4. **A cold last-layer solve fails on nonlinear equations.** Solved on random initial features, the last layer lands in a basin with error 0.47 that the hidden weights do not leave. Sixty plain steps first remove this. The linear equation needs no warm-up.
5. **Small blocks are fast and then stall.** Blocks of 128 rows reach 3e-3 on KdV within 2 s, ahead of every other method at that moment, then stop improving. Blocks of 32 are worse everywhere. Satisfying a few rows exactly disturbs the others.
6. **Setting the damping like an ODE step size changes nothing.** M1 follows Levenberg–Marquardt on all three equations and ends level or slightly behind.
7. **The undamped flow does not move at all.** At the start $J$ has numerical rank 93 of 2,241 (singular values from 7.8 down to 6e-20). With a fixed cut-off of the pseudo-inverse, the step has length 20 to 100 and raises the residual at every step length above 0.01. The damping is the method, not a detail.

### The second group of moves

8. **The filter is the first method to clear the bar on Burgers.** At 180 s it holds 2.0e-05 against 1.3e-03 for Levenberg–Marquardt, 65 times lower, on all three seeds. It is 73 times lower on heat (1.5e-09) and 10 times lower on KdV (1.0e-07).
9. **Its lead comes late.** In the first 30 s Levenberg–Marquardt is 10 to 50 times ahead. The two cross near 60 s; after that Levenberg–Marquardt flattens and the filter keeps falling.
10. **Small batches win.** Batches of 64 beat batches of 256 on every equation: the memory in $P$ does the work a large batch would.
11. **The plain filter diverges, and the fix has two parts.** With forgetting alone, 5 of 6 runs on heat and KdV blew up between 110 and 150 s: $P$ grows without bound in directions the stream does not excite, until its range exceeds double precision.
    - **A cap on the variances** at $10^{12}$ times the prior removed 8 of 9 failures. A cap at the prior itself stalls the method at 1e-04: the growth of $P$ is needed.
    - **A check on each update** removed the last: an update that more than doubles the residual of its own batch is refused and $P$ starts again. It fired at most once per run.
12. **Continuation does not pay at this viscosity.** With 4, 8 or 16 stages it ends within the spread of a cold start.
13. **Multipliers are worse than their own control** on all three equations.
14. **Loss weights of 10 were not a good default.** Weights of 1 are level or better everywhere, and 2 times better on Burgers.
15. **M5b needs no arm.** The last-layer solve on the equation linearized about the current solution is the Gauss–Newton sweep on the last layer that M5 already runs on nonlinear equations.

### Stage 2b: the filter's two weaknesses

16. **The hybrid removes the slow start and keeps the end.** With 20 s of Levenberg–Marquardt first, the error at 10 s and 30 s is Levenberg–Marquardt's, and the last 30 s are level with the plain filter: 1.8e-09 on heat, 2.8e-05 on Burgers, 1.1e-07 on KdV.
    - **The switch costs a pause.** $P$ restarts at its prior, so the error stays flat for about 40 s after the switch. At 60 s the hybrid is behind the plain filter on heat (2.0e-06 against 4.4e-07) and level on KdV. On Burgers it is ahead at every reading up to 120 s.
17. **A covariance per layer survives on two equations of three.**
    - **Heat**: 1.6e-09, level with the full filter, and 30 times ahead of it at 60 s.
    - **KdV**: 4.2e-09, 24 times below the full filter (seeds 1.4e-09, 4.6e-09, 4.6e-08).
    - **Burgers**: 4.0e-04, 20 times above the full filter and 3 times below Levenberg–Marquardt. The coupling between layers matters where the solution has a front.
    - **Cost**: 1.4 to 1.5 times the steps of the full filter in the same time. One layer holds 1,056 of the 2,241 weights, so the saving is small at this size.
18. **A covariance per neuron fails.** 4.8e-06 on heat and 9.5e-05 on KdV, behind Levenberg–Marquardt on both. On Burgers two seeds of three diverge, with over 3,000 refused updates each.

## Verdict against the plan

| Move | Gate | Result |
|---|---|---|
| Stage 0, Levenberg–Marquardt | $10^{-6}$ on heat on one core | Passed |
| M1, residual flow | Beats stage 0 on two of three | Dropped: no gain on any |
| M4, block Kaczmarz | Beats stage 0 on two of three | Dropped as the main optimizer. Kept as a candidate opening move for the agent (M9) |
| M5, variable projection | Beats stage 0 on two of three; a gain on the linear problem | Kept, with a warm start: a large gain on heat, level on the other two |
| M3, filtering | Beats stage 0 on two of three; works without a full-rank covariance | **Kept, and the lead candidate**: ahead on all three at 180 s. Without the full $P$: a block per layer holds on heat and KdV and loses a factor of 20 on Burgers; a block per neuron fails |
| M6, continuation | A cold run is not as good | Dropped at this viscosity |
| M8, multipliers | Beats stage 0 on two of three | Dropped |

Roadmap bet R1 asks for 100 times below Adam then L-BFGS. At 60 s Levenberg–Marquardt meets it on heat and KdV and not on Burgers. The baselines were not run to 180 s, so the filter's margin over them at that time is not measured.

## What it does not show

- **Three seeds**, a network of 2,241 weights, three equations in one space dimension. The plan asks for ten seeds and the nine equations of I17.
- **Dense linear algebra.** The cost per step grows with the cube of the smaller of rows and weights. Larger networks need the matrix-free solve.
- **Double precision in PyTorch.** The engine runs in single precision; errors near $10^{-8}$ will not survive it unchanged.
- **Baselines at their defaults.** Adam's rate and the L-BFGS history were not tuned per equation, and neither was anything in the new methods.
- **The hybrid's 20 s** was set once and not scanned, and $P$ is not initialized from the Levenberg–Marquardt phase.
- **The warm-up length of 60** was chosen once, on a 30 s test, and not scanned.
- **The filter's constants** (forgetting 0.995, prior 1, cap $10^{12}$, the factor 2 in the check) were set once. The cap was chosen from one seed at three values.
- **The filter's cost grows with the square of the number of weights** per step and its memory likewise (40 MB here). A block per layer still costs the square of a layer's weights. A form that scales (Kronecker-factored, or low rank plus diagonal) is untested.
- **No baseline at 180 s** other than Levenberg–Marquardt.
- **The Burgers reference** is the package's Cole–Hopf solution on a 256 by 101 grid; its own accuracy bounds what can be read below about 1e-4.
