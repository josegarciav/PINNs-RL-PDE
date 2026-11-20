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
5. **Baselines**: Adam at $10^{-3}$, L-BFGS (strong Wolfe), Adam for 30 % of the budget then L-BFGS.

```bash
python experiments/D10_least_squares_optimizer/run.py --budget 60 --seeds 3 \
    --methods lm flow block32 block128 varpro varpro60 adam lbfgs adam_lbfgs
python experiments/D10_least_squares_optimizer/plot.py
```

## Results

Median $\varepsilon(T)$ over 3 seeds, with the range.

| Equation | Method | 10 s | 30 s | 60 s | Steps |
|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 2.7e-05 (2e-05–4e-05) | 2.8e-06 (1e-06–5e-06) | 6.2e-07 (5e-07–2e-06) | 421 |
| heat | M1 damping from the linearization error | 4.5e-05 (4e-05–8e-05) | 3.5e-06 (3e-06–4e-06) | 1.6e-06 (7e-07–2e-06) | 358 |
| heat | M5 last layer solved, after 60 plain steps | 7.6e-05 (7e-05–2e-04) | 2.9e-08 (2e-08–3e-08) | **4.9e-09** (3e-09–6e-09) | 249 |
| heat | M5 last layer solved, from the start | 3.8e-07 (1e-07–5e-07) | 3.2e-08 (2e-08–5e-08) | 1.1e-08 (9e-09–1e-08) | 346 |
| heat | M4 blocks of 128 rows | 3.2e-04 (2e-04–5e-04) | 3.1e-04 (2e-04–5e-04) | 2.3e-04 (9e-05–3e-04) | 7491 |
| heat | M4 blocks of 32 rows | 1.5e-03 (1e-03–2e-03) | 1.7e-03 (1e-03–3e-03) | 1.5e-03 (1e-03–2e-03) | 14189 |
| heat | Adam, then L-BFGS | 2.5e-03 (2e-03–3e-03) | 1.0e-04 (1e-04–1e-04) | 7.6e-05 (6e-05–9e-05) | 4283 |
| heat | L-BFGS | 1.4e-04 (1e-04–2e-04) | 8.6e-05 (8e-05–1e-04) | 7.8e-05 (7e-05–1e-04) | 7517 |
| heat | Adam | 2.5e-03 (2e-03–3e-03) | 6.8e-04 (5e-04–2e-03) | 5.7e-04 (4e-04–6e-03) | 14418 |
| burgers | Levenberg–Marquardt | 9.2e-02 (4e-02–2e-01) | 3.0e-03 (3e-03–4e-03) | 4.5e-03 (3e-03–8e-03) | 403 |
| burgers | M1 damping from the linearization error | 9.8e-02 (4e-02–1e-01) | 1.0e-02 (6e-03–2e-02) | 3.3e-03 (3e-03–7e-03) | 399 |
| burgers | M5 last layer solved, after 60 plain steps | 1.3e-01 (1e-01–2e-01) | 1.9e-02 (1e-02–2e-02) | 5.3e-03 (1e-03–7e-03) | 194 |
| burgers | M5 last layer solved, from the start | 4.9e-01 (5e-01–5e-01) | 4.8e-01 (5e-01–5e-01) | 4.7e-01 (4e-01–5e-01) | 137 |
| burgers | M4 blocks of 128 rows | 1.9e-01 (2e-01–4e-01) | 1.7e-01 (1e-01–3e-01) | 1.4e-01 (1e-01–2e-01) | 7124 |
| burgers | M4 blocks of 32 rows | 4.0e-01 (4e-01–4e-01) | 3.5e-01 (3e-01–4e-01) | 3.6e-01 (3e-01–4e-01) | 13916 |
| burgers | Adam, then L-BFGS | 3.3e-01 (3e-01–4e-01) | 2.7e-02 (2e-02–9e-02) | 1.4e-02 (8e-03–2e-02) | 4108 |
| burgers | L-BFGS | 2.7e-01 (8e-02–4e-01) | 1.4e-01 (6e-02–5e-01) | 2.2e-02 (5e-03–4e-01) | 564 |
| burgers | Adam | 3.3e-01 (3e-01–4e-01) | 1.3e-01 (5e-02–2e-01) | 4.5e-02 (3e-02–1e-01) | 13937 |
| kdv | Levenberg–Marquardt | 2.0e-04 (1e-04–5e-04) | 9.9e-06 (6e-06–1e-05) | **6.0e-06** (3e-06–6e-06) | 365 |
| kdv | M1 damping from the linearization error | 7.5e-04 (2e-04–2e-03) | 2.9e-05 (3e-05–6e-05) | 1.1e-05 (7e-06–1e-05) | 325 |
| kdv | M5 last layer solved, after 60 plain steps | 4.5e-04 (3e-04–2e-03) | 7.4e-05 (1e-05–9e-05) | 6.6e-06 (4e-06–1e-05) | 140 |
| kdv | M5 last layer solved, from the start | 5.1e-01 (4e-01–6e-01) | 5.0e-01 (4e-01–6e-01) | 4.8e-01 (4e-01–6e-01) | 184 |
| kdv | M4 blocks of 128 rows | 2.7e-03 (2e-03–4e-03) | 2.4e-03 (2e-03–4e-03) | 2.3e-03 (2e-03–4e-03) | 5163 |
| kdv | M4 blocks of 32 rows | 7.9e-03 (5e-03–1e-02) | 5.0e-03 (4e-03–8e-03) | 4.7e-03 (4e-03–6e-03) | 9549 |
| kdv | Adam, then L-BFGS | 1.2e-01 (1e-01–3e-01) | 2.7e-03 (3e-03–3e-03) | 1.3e-03 (1e-03–1e-03) | 1155 |
| kdv | L-BFGS | 4.2e-03 (4e-03–5e-03) | 2.1e-03 (2e-03–3e-03) | 2.0e-03 (2e-03–2e-03) | 1961 |
| kdv | Adam | 7.3e-02 (3e-02–8e-02) | 1.7e-02 (2e-02–2e-02) | 3.8e-03 (3e-03–5e-03) | 8925 |

The figure is `results/runs_3x32.png` after `plot.py`.

## What it shows

1. **The squares are worth two orders of magnitude on smooth problems.** At 60 s Levenberg–Marquardt is 120 times below Adam then L-BFGS on heat and 220 times below on KdV. It passes the plan's stage 0 gate: under $10^{-6}$ on heat on one core.
2. **Burgers is the exception.** The gain is 3 times, and the error moves up and down between 1e-3 and 1e-2 as the points are redrawn. The sharp front, not the optimizer's order, sets the error. This is where continuation (M6) and residual-chosen points have to earn their place.
3. **Solving the last layer is the largest single gain, on the linear equation only.** Heat reaches 4.9e-09, 127 times below Levenberg–Marquardt and four orders below the baselines. On Burgers and KdV it ties Levenberg–Marquardt within the spread.
4. **A cold last-layer solve fails on nonlinear equations.** Solved on random initial features, the last layer lands in a basin with error 0.47 that the hidden weights do not leave. Sixty plain steps first remove this. The linear equation needs no warm-up.
5. **Small blocks are fast and then stall.** Blocks of 128 rows reach 3e-3 on KdV within 2 s, ahead of every other method at that moment, then stop improving. Blocks of 32 are worse everywhere. Satisfying a few rows exactly disturbs the others.
6. **Setting the damping like an ODE step size changes nothing.** M1 follows Levenberg–Marquardt on all three equations and ends level or slightly behind.
7. **The undamped flow does not move at all.** At the start $J$ has numerical rank 93 of 2,241 (singular values from 7.8 down to 6e-20). With a fixed cut-off of the pseudo-inverse, the step has length 20 to 100 and raises the residual at every step length above 0.01. The damping is the method, not a detail.

## Verdict against the plan

| Move | Gate | Result |
|---|---|---|
| Stage 0, Levenberg–Marquardt | $10^{-6}$ on heat on one core | Passed |
| M1, residual flow | Beats stage 0 on two of three | Dropped: no gain on any |
| M4, block Kaczmarz | Beats stage 0 on two of three | Dropped as the main optimizer. Kept as a candidate opening move for the agent (M9) |
| M5, variable projection | Beats stage 0 on two of three; a gain on the linear problem | Kept, with a warm start: a large gain on heat, level on the other two |

Roadmap bet R1 asks for 100 times below Adam then L-BFGS. It holds on two of these three equations, not on Burgers.

## What it does not show

- **Three seeds**, a network of 2,241 weights, three equations in one space dimension. The plan asks for ten seeds and the nine equations of I17.
- **Dense linear algebra.** The cost per step grows with the cube of the smaller of rows and weights. Larger networks need the matrix-free solve.
- **Double precision in PyTorch.** The engine runs in single precision; errors near $10^{-8}$ will not survive it unchanged.
- **Baselines at their defaults.** Adam's rate and the L-BFGS history were not tuned per equation, and neither was anything in the new methods.
- **The warm-up length of 60** was chosen once, on a 30 s test, and not scanned.
- **The Burgers reference** is the package's Cole–Hopf solution on a 256 by 101 grid; its own accuracy bounds what can be read below about 1e-4.
