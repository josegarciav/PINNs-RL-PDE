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
- **Fronts** (the last table only):
  - **Allen–Cahn**: $u_t = 2 \cdot 10^{-4}\, u_{xx} + u - u^3$ on $[-1, 1] \times [0, 5]$, $u_0 = x^2 \cos \pi x$, $u(\pm 1, t) = -1$. Reference: method of lines on a grid 32 times finer than the 257 by 101 output grid, RK45 at a relative tolerance of $10^{-9}$.
  - **Sharp Burgers**: $\nu = 0.003/\pi$: the usual front 3.3 times thinner. Reference: Cole–Hopf on a 513 by 101 grid.
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
   - **Hybrid, warm**: the same with $P$ started at $(J^\top J + I)^{-1}$ on one full sample at the switch.
   - **Covariance per layer**: $P$ kept only within each layer (4 blocks, the largest 1,056 weights), one shared $J P J^\top + I$.
   - **Covariance per neuron**: $P$ kept only among the weights that enter one neuron (97 blocks of at most 33).
   - **Forgetting**: $f = 0.95$, $0.98$, $0.9$, $0.999$ in place of $0.995$; and a **growing memory**, $f_k = 1 - 1/\max(20,\, k/25)$ at step $k$: the filter remembers 1 batch in 25 of those it has seen, and never fewer than 20.
   - **Low rank**: no $n \times n$ matrix. The information is kept as $P^{-1} = c\,I + V \operatorname{diag}(s^2) V^\top$ with $V$ orthonormal, $n \times k$. Each batch adds $J^\top J$ through one SVD of a $(k + 64)$-square matrix and keeps the $k$ strongest directions. Cost per step $n k (k + 64)$ against $n^2 \cdot 64$.
7. **M8**: multipliers. Initial and boundary rows as constraints with an augmented Lagrangian at penalty 1, inside Levenberg–Marquardt. Its control is Levenberg–Marquardt at loss weights 1.
8. **Baselines**: Adam at $10^{-3}$, L-BFGS (strong Wolfe), Adam for 30 % of the budget then L-BFGS.
9. **Self-scaled quasi-Newton** (Urbán, Stefanou and Pons, 2024, arXiv 2405.04230): Adam for 30 % of the budget, then a dense inverse Hessian $H$ with a strong Wolfe line search on all 2,512 rows, redrawn every 500 steps:
   $H \leftarrow \tfrac{1}{\tau}\big(H - \tfrac{H y\, y^\top H}{y^\top H y} + \phi\, v v^\top\big) + \tfrac{s s^\top}{y^\top s}$, $\quad v = \sqrt{y^\top H y}\,\big(\tfrac{s}{y^\top s} - \tfrac{H y}{y^\top H y}\big)$.
   - **$s$, $y$**: the step and the change of the gradient.
   - **$\tau$**: the scaling, at most 1. Dividing by it widens $H$.
   - **$\phi$**: the member of the Broyden family.
   - **Self-scaled BFGS**: $\phi = 1$ and $\tau = \min(1, y^\top s / s^\top H^{-1} s)$ (the paper's eq. 11).
   - **Self-scaled Broyden**: $\tau$ and $\phi$ both from the step (eqs. 13–23).
   - **Our additions**: $H = (s^\top y / y^\top y)\, I$ at the first update; $H = I$ again after a failed line search.
   - **Adam first**: for 30 % of the budget as in the paper, for 10 s, or not at all.
   - **After the filter**: the growing-memory filter for 30 s, then self-scaled Broyden from its weights. $H$ starts from the scaled identity, or from the filter's $P$ scaled by $s^\top y / y^\top P y$.

```bash
D=experiments/D10_least_squares_optimizer
python $D/run.py --budget 60 --seeds 3 --methods lm flow block32 block128 varpro varpro60 \
    continue4 continue8 continue16 filter64 filter256 al lm_w1 adam lbfgs adam_lbfgs
python $D/run.py --budget 180 --seeds 3 --out $D/results/long \
    --methods lm filter64 lmfilter20 lmfilter20w dfilter64layer dfilter64node sfilter64r64 adam_lbfgs
python $D/run.py --budget 600 --seeds 2 --width 64 --depth 4 --every 10 --out $D/results/wide \
    --methods lm dfilter64layer sfilter64r64 adam_lbfgs
python $D/plot.py && python $D/plot.py $D/results/long/runs_3x32.json
python $D/plot.py $D/results/wide/runs_4x64.json
python $D/run.py --budget 180 --seeds 3 --methods lm filter64 filter64f95 filter64g25 --out $D/results/forget
python $D/run.py --budget 300 --seeds 1 --width 64 --depth 4 --every 10 --out $D/results/wide_grow \
    --methods lm filter64g25 sfilter64r64g25
python $D/run.py --budget 180 --seeds 3 --out $D/results/baselines \
    --methods lm filter64g25 adam_ssbroyden adam_ssbfgs adam_lbfgs \
    adam10_ssbroyden adam0_ssbroyden filter_ss30 filter_ss30p filter64g25m adam0_ssbroydenf
python $D/plot.py $D/results/baselines/runs_3x32.json $D/results/baselines/runs_3x32.png \
    lm,filter64g25,filter_ss30,adam0_ssbroyden,adam10_ssbroyden,adam_ssbroyden,adam_lbfgs
python $D/run.py --budget 180 --seeds 3 --problems allen_cahn --out $D/results/fronts \
    --methods lm filter64g25 filter64g25m adam0_ssbroyden adam0_ssbroydenf adam10_ssbroyden \
    adam_ssbroyden adam_lbfgs
python $D/run.py --budget 180 --seeds 3 --problems burgers_sharp --out $D/results/fronts \
    --methods lm filter64g25 filter64g25m adam0_ssbroyden adam10_ssbroyden
python $D/plot.py $D/results/fronts/runs_3x32.json
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
| heat | The same, covariance started from the last linearization | 2.4e-05 (1e-05–4e-05) | 3.3e-06 (2e-06–3e-06) | 2.6e-06 (2e-06–3e-06) | 1.7e-08 (2e-08–2e-08) | 2.2e-09 | 8737 |
| heat | M3 filter, covariance per layer | 1.1e-04 (1e-04–2e-04) | 2.1e-06 (2e-06–3e-06) | 1.5e-08 (1e-08–2e-08) | 3.8e-09 (3e-09–5e-09) | 1.6e-09 | 17017 |
| heat | M3 filter, 64 directions kept | 7.7e-05 (7e-05–1e-04) | 1.5e-06 (1e-06–2e-06) | 1.1e-06 (7e-07–1e-06) | 4.9e-07 (4e-07–1e-06) | 4.5e-07 | 22318 |
| heat | M3 filter, covariance per neuron | 5.1e-04 (5e-04–6e-04) | 5.1e-05 (3e-05–7e-05) | 2.6e-05 (7e-06–3e-05) | 7.6e-06 (5e-06–2e-05) | 4.8e-06 | 20780 |
| heat | Adam, then L-BFGS | 8.4e-03 (3e-03–1e-02) | 6.6e-04 (5e-04–7e-04) | 1.2e-04 (9e-05–1e-04) | 5.7e-05 (5e-05–8e-05) | 5.2e-05 | 14811 |
| burgers | Levenberg–Marquardt | 4.3e-02 (3e-02–1e-01) | 4.8e-03 (3e-03–8e-03) | 2.8e-03 (3e-03–5e-03) | 1.3e-03 (9e-04–3e-03) | 1.3e-03 | 1333 |
| burgers | M3 Kalman filter, batches of 64 | 2.3e-01 (2e-01–3e-01) | 3.8e-02 (6e-03–1e-01) | 4.8e-04 (3e-04–1e-03) | 3.8e-05 (3e-05–2e-04) | 2.0e-05 | 11020 |
| burgers | Levenberg–Marquardt for 20 s, then the filter | 5.7e-02 (2e-02–1e-01) | 2.2e-03 (1e-03–7e-03) | 2.1e-04 (2e-04–4e-04) | 3.1e-05 (2e-05–1e-04) | 2.8e-05 | 9729 |
| burgers | The same, covariance started from the last linearization | 4.8e-02 (2e-02–1e-01) | 2.5e-03 (2e-03–4e-03) | 3.3e-04 (3e-04–5e-04) | 1.4e-04 (5e-05–2e-04) | 3.9e-05 | 8164 |
| burgers | M3 filter, covariance per layer | 7.0e-02 (5e-02–1e-01) | 7.7e-03 (5e-03–9e-03) | 1.7e-03 (1e-03–3e-03) | 4.3e-04 (3e-04–8e-04) | 4.0e-04 | 16554 |
| burgers | M3 filter, 64 directions kept | 2.0e-01 (7e-02–2e-01) | 1.5e-02 (1e-02–2e-02) | 1.8e-03 (2e-03–2e-03) | 7.7e-04 (5e-04–1e-03) | 9.1e-04 | 21724 |
| burgers | M3 filter, covariance per neuron | 2.1e-01 (2e-01–3e-01) | 1.2e-01 (3e-02–2e-01) | 1.1e-01 (3e-02–3e-01) | 9.1e-01 (2e-02–9e-01) | 9.8e-01 | 20949 |
| burgers | Adam, then L-BFGS | 3.1e-01 (3e-01–4e-01) | 7.0e-02 (4e-02–2e-01) | 4.0e-02 (2e-02–1e-01) | 8.7e-03 (2e-03–2e-02) | 7.0e-03 | 14839 |
| kdv | Levenberg–Marquardt | 1.5e-04 (1e-04–5e-04) | 9.9e-06 (6e-06–1e-05) | 4.8e-06 (2e-06–6e-06) | 1.5e-06 (6e-07–2e-06) | 9.6e-07 | 1071 |
| kdv | M3 Kalman filter, batches of 64 | 9.8e-03 (7e-03–1e-02) | 5.0e-04 (3e-04–6e-04) | 1.3e-05 (1e-05–2e-05) | 2.3e-07 (9e-08–2e-07) | 1.0e-07 | 9029 |
| kdv | Levenberg–Marquardt for 20 s, then the filter | 1.4e-04 (1e-04–5e-04) | 1.5e-05 (1e-05–2e-05) | 1.5e-05 (1e-05–2e-05) | 2.6e-07 (2e-07–5e-07) | 1.1e-07 | 8008 |
| kdv | The same, covariance started from the last linearization | 1.5e-04 (1e-04–6e-04) | 2.2e-05 (1e-05–3e-05) | 2.0e-05 (1e-05–2e-05) | 1.2e-06 (7e-07–5e-06) | 1.9e-07 | 6114 |
| kdv | M3 filter, covariance per layer | 4.4e-03 (3e-03–6e-03) | 1.1e-04 (7e-05–2e-04) | 2.2e-06 (2e-06–4e-06) | 2.9e-08 (2e-08–6e-08) | 4.2e-09 | 12608 |
| kdv | M3 filter, 64 directions kept | 1.9e-03 (1e-03–2e-03) | 3.7e-05 (3e-05–1e-04) | 1.8e-05 (4e-06–3e-05) | 1.2e-05 (3e-06–1e-05) | 6.5e-06 | 15923 |
| kdv | M3 filter, covariance per neuron | 1.1e-02 (7e-03–2e-02) | 1.1e-03 (1e-03–2e-03) | 5.4e-04 (3e-04–1e-03) | 1.1e-04 (1e-04–1e-03) | 9.5e-05 | 14694 |
| kdv | Adam, then L-BFGS | 1.2e-01 (1e-01–2e-01) | 3.4e-02 (2e-02–6e-02) | 3.8e-03 (3e-03–4e-03) | 1.1e-03 (8e-04–1e-03) | 9.8e-04 | 3869 |

### A larger network, 600 seconds

4 hidden layers of 64, 12,737 weights (I17's network), 2 seeds, the error read every 10 s.

| Equation | Method | 30 s | 60 s | 180 s | 300 s | Last 100 s, mean | Steps |
|---|---|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 1.2e-04 (1e-04–1e-04) | 2.2e-05 (2e-05–2e-05) | 2.1e-06 (2e-06–2e-06) | 6.5e-07 (6e-07–7e-07) | 3.4e-07 | 853 |
| heat | M3 filter, covariance per layer | 1.2e-03 (1e-03–1e-03) | 5.3e-04 (4e-04–7e-04) | 1.3e-04 (1e-04–1e-04) | 7.8e-05 (7e-05–8e-05) | 1.4e-05 | 1824 |
| heat | M3 filter, 64 directions kept | 6.8e-05 (5e-05–9e-05) | 1.3e-06 (1e-06–1e-06) | 6.7e-06 (6e-07–1e-05) | 4.4e-07 (4e-07–5e-07) | 4.1e-07 | 26373 |
| heat | Adam, then L-BFGS | 4.2e-03 (3e-03–5e-03) | 1.5e-03 (1e-03–2e-03) | 5.3e-03 (5e-04–1e-02) | 5.0e-05 (5e-05–5e-05) | 5.0e-05 | 13634 |
| burgers | Levenberg–Marquardt | 7.6e-02 (3e-02–1e-01) | 1.2e-02 (3e-03–2e-02) | 3.9e-04 (3e-04–5e-04) | 6.1e-04 (2e-04–1e-03) | 5.9e-04 | 1372 |
| burgers | M3 filter, covariance per layer | 1.5e-01 (1e-01–2e-01) | 1.9e-01 (1e-01–3e-01) | 1.8e-01 (1e-01–2e-01) | 3.2e-02 (3e-02–4e-02) | 2.5e-03 | 2849 |
| burgers | M3 filter, 64 directions kept | 7.7e-02 (2e-02–1e-01) | 3.1e-02 (1e-02–5e-02) | 1.8e-03 (1e-03–2e-03) | 1.9e-03 (5e-04–3e-03) | 5.4e-04 | 16882 |
| burgers | Adam, then L-BFGS | 3.7e-01 (4e-01–4e-01) | 3.2e-01 (3e-01–3e-01) | 4.2e-02 (3e-02–5e-02) | 1.7e-02 (8e-03–3e-02) | 3.1e-03 | 8799 |
| kdv | Levenberg–Marquardt | 3.8e-03 (3e-03–5e-03) | 2.9e-04 (2e-04–3e-04) | 1.9e-05 (1e-05–2e-05) | 6.3e-06 (5e-06–8e-06) | 2.8e-06 | 638 |
| kdv | M3 filter, covariance per layer | 1.4e-01 (1e-01–1e-01) | 3.1e-02 (2e-02–4e-02) | 3.7e-03 (4e-03–4e-03) | 1.1e-03 (9e-04–1e-03) | 1.3e-04 | 1715 |
| kdv | M3 filter, 64 directions kept | 3.9e-03 (3e-03–5e-03) | 3.5e-04 (2e-04–5e-04) | 4.2e-06 (3e-06–5e-06) | 4.9e-06 (4e-06–5e-06) | 3.3e-06 | 13230 |
| kdv | Adam, then L-BFGS | 7.5e-02 (4e-02–1e-01) | 3.2e-02 (2e-02–5e-02) | 5.8e-03 (5e-03–6e-03) | 8.5e-04 (8e-04–9e-04) | 8.5e-04 | 8783 |

### The forgetting factor, 180 seconds

Run later, on battery in Low Power Mode: every method takes about half the steps it takes above, so this table is read against itself only. Levenberg–Marquardt and the filter at 0.995 are rerun as its controls.

| Equation | Method | 10 s | 30 s | 60 s | 120 s | Last 30 s, mean | Steps |
|---|---|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 7.6e-05 (7e-05–1e-04) | 4.5e-06 (4e-06–5e-06) | 1.6e-06 (8e-07–2e-06) | 4.6e-07 (3e-07–5e-07) | 2.2e-07 | 877 |
| heat | M3 Kalman filter, batches of 64 | 6.1e-04 (5e-04–6e-04) | 8.2e-05 (7e-05–1e-04) | 1.1e-05 (1e-05–1e-05) | 2.8e-07 (2e-07–3e-07) | 2.1e-08 | 5837 |
| heat | M3 filter, forgetting 0.95 | 1.2e-05 (1e-05–1e-05) | 1.6e-08 (8e-09–3e-08) | 8.1e-09 (5e-09–1e-08) | 8.6e-09 (2e-09–1e-08) | 2.9e-09 | 5745 |
| heat | M3 filter, memory of 1 batch in 25 seen, at least 20 | 1.4e-05 (1e-05–2e-05) | 1.5e-08 (6e-09–2e-08) | 5.7e-09 (5e-09–1e-08) | 1.7e-09 (1e-09–5e-09) | 1.5e-09 | 5895 |
| burgers | Levenberg–Marquardt | 1.3e-01 (1e-01–3e-01) | 1.5e-02 (1e-02–3e-02) | 2.6e-03 (1e-03–5e-03) | 1.2e-03 (1e-03–2e-03) | 1.9e-03 | 778 |
| burgers | M3 Kalman filter, batches of 64 | 3.2e-01 (3e-01–3e-01) | 1.3e-01 (5e-02–2e-01) | 3.9e-02 (5e-03–1e-01) | 4.5e-04 (4e-04–5e-04) | 1.7e-04 | 5608 |
| burgers | M3 filter, forgetting 0.95 | 1.4e-01 (7e-02–2e-01) | 1.3e-02 (4e-03–2e-01) | 2.3e-03 (1e-03–3e-03) | 2.4e-04 (2e-04–2e-03) | 5.2e-04 | 5570 |
| burgers | M3 filter, memory of 1 batch in 25 seen, at least 20 | 1.3e-01 (9e-02–2e-01) | 1.9e-02 (4e-03–1e-01) | 1.5e-03 (1e-03–4e-03) | 3.3e-04 (6e-05–5e-04) | 1.1e-04 | 5704 |
| kdv | Levenberg–Marquardt | 4.5e-04 (3e-04–2e-03) | 2.5e-05 (2e-05–3e-05) | 9.1e-06 (4e-06–1e-05) | 2.8e-06 (2e-06–4e-06) | 1.5e-06 | 663 |
| kdv | M3 Kalman filter, batches of 64 | 2.9e-02 (1e-02–3e-02) | 2.9e-03 (2e-03–4e-03) | 3.6e-04 (3e-04–4e-04) | 6.7e-06 (7e-06–9e-06) | 4.8e-07 | 4957 |
| kdv | M3 filter, forgetting 0.95 | 1.8e-04 (2e-04–2e-04) | 4.5e-07 (2e-07–6e-07) | 1.4e-07 (9e-08–3e-07) | 1.4e-07 (1e-07–2e-07) | 1.0e-07 | 4727 |
| kdv | M3 filter, memory of 1 batch in 25 seen, at least 20 | 2.2e-04 (2e-04–4e-04) | 6.5e-07 (4e-07–8e-07) | 9.2e-08 (4e-08–2e-07) | 8.3e-08 (4e-08–2e-07) | 1.3e-07 | 4489 |

A first scan at 60 s, one seed, same session:

| Forgetting | Heat | KdV | Burgers | Refused updates |
|---|---|---|---|---|
| 0.995 | 9.2e-06 | 2.7e-04 | 2.6e-02 | 0 |
| 0.98 | 6.9e-09 | 1.2e-06 | 1.4e-03 | 0–1 |
| 0.95 | 7.5e-09 | 3.8e-07 | 5.1e-04 | 1–3 |
| 0.9 | 6.3e-09 | 1.1e-07 | 9.5e-03 | 6–22 |

### The growing memory on the larger network, 300 seconds

One seed. The machine went from battery in Low Power Mode to mains power during this run, so only the three methods of one equation, run one after another, compare with each other.

| Equation | Method | 30 s | 60 s | 180 s | 300 s | Last 50 s, mean | Steps |
|---|---|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 1.1e-04 (1e-04–1e-04) | 1.9e-05 (2e-05–2e-05) | 2.1e-06 (2e-06–2e-06) | 7.5e-07 (7e-07–7e-07) | 9.1e-07 | 404 |
| heat | M3 filter, memory of 1 batch in 25 seen, at least 20 | 3.1e-03 (3e-03–3e-03) | 6.6e-04 (7e-04–7e-04) | 5.5e-05 (6e-05–6e-05) | 1.7e-05 (2e-05–2e-05) | 1.8e-05 | 304 |
| heat | M3 filter, 64 directions, growing memory | 1.6e-05 (2e-05–2e-05) | 8.1e-06 (8e-06–8e-06) | 2.0e-06 (2e-06–2e-06) | 1.3e-06 (1e-06–1e-06) | 1.5e-06 | 8009 |
| burgers | Levenberg–Marquardt | 2.3e-01 (2e-01–2e-01) | 1.1e-02 (1e-02–1e-02) | 3.8e-04 (4e-04–4e-04) | 3.8e-03 (4e-03–4e-03) | 1.4e-03 | 376 |
| burgers | M3 filter, memory of 1 batch in 25 seen, at least 20 | 4.7e-01 (5e-01–5e-01) | 3.7e-01 (4e-01–4e-01) | 2.9e-01 (3e-01–3e-01) | 8.9e-02 (9e-02–9e-02) | 1.0e-01 | 303 |
| burgers | M3 filter, 64 directions, growing memory | 3.4e-01 (3e-01–3e-01) | 4.8e-02 (5e-02–5e-02) | 1.7e-03 (2e-03–2e-03) | 5.3e-04 (5e-04–5e-04) | 5.8e-04 | 8737 |
| kdv | Levenberg–Marquardt | 1.5e-03 (2e-03–2e-03) | 1.0e-04 (1e-04–1e-04) | 4.9e-06 (5e-06–5e-06) | 1.7e-06 (2e-06–2e-06) | 1.9e-06 | 572 |
| kdv | M3 filter, memory of 1 batch in 25 seen, at least 20 | 1.6e-01 (2e-01–2e-01) | 6.0e-03 (6e-03–6e-03) | 2.6e-05 (3e-05–3e-05) | 1.6e-06 (2e-06–2e-06) | 3.2e-06 | 573 |
| kdv | M3 filter, 64 directions, growing memory | 4.9e-05 (5e-05–5e-05) | 9.7e-06 (1e-05–1e-05) | 4.9e-06 (5e-06–5e-06) | 2.9e-06 (3e-06–3e-06) | 3.1e-06 | 12003 |

### The strongest published baseline, 180 seconds

Every method run fresh, in three sessions on the same day, all on mains power with Low Power Mode off and nothing else running. This is the cleanest table of the study. The figure shows seven of the rows.

| Equation | Method | 10 s | 30 s | 60 s | 120 s | Last 30 s, mean | Steps |
|---|---|---|---|---|---|---|---|
| heat | Levenberg–Marquardt | 2.4e-05 (1e-05–4e-05) | 2.3e-06 (1e-06–2e-06) | 5.5e-07 (4e-07–8e-07) | 1.9e-07 (1e-07–2e-07) | 1.1e-07 | 1520 |
| heat | M3 filter, memory of 1 batch in 25 seen, at least 20 | 3.0e-07 (2e-07–4e-07) | 8.0e-09 (5e-09–2e-08) | 1.7e-09 (2e-09–6e-09) | 8.3e-10 (6e-10–2e-09) | 9.6e-10 | 10892 |
| heat | The filter for 30 s, then self-scaled Broyden | 3.2e-07 (3e-07–4e-07) | 8.3e-09 (6e-09–2e-08) | 3.2e-09 (2e-09–4e-09) | 2.5e-10 (2e-10–4e-10) | 8.6e-11 | 12521 |
| heat | The same, Broyden's first H from the filter's P | 3.3e-07 (2e-07–4e-07) | 6.1e-09 (5e-09–2e-08) | 9.8e-10 (7e-10–4e-09) | 2.8e-10 (2e-10–3e-10) | 1.8e-10 | 18348 |
| heat | Adam for 10 s, then self-scaled Broyden | 2.1e-03 (1e-03–3e-03) | 2.9e-07 (1e-07–3e-07) | 6.0e-09 (3e-09–6e-09) | 2.4e-10 (2e-10–4e-10) | 2.4e-10 | 18547 |
| heat | Self-scaled Broyden from the start | 5.0e-06 (4e-06–5e-06) | 6.9e-08 (6e-08–8e-08) | 2.7e-09 (2e-09–5e-09) | 3.5e-10 (3e-10–4e-10) | 3.4e-10 | 20707 |
| heat | The same, the filter's weighting (5/3) | 3.7e-06 (4e-06–5e-06) | 7.8e-08 (5e-08–1e-07) | 4.7e-09 (2e-09–9e-09) | 3.5e-10 (3e-10–5e-10) | 3.5e-10 | 19390 |
| heat | M3 filter, growing memory, the full loss's weighting (10) | 2.7e-07 (2e-07–3e-07) | 1.9e-08 (1e-08–2e-08) | 2.3e-09 (2e-09–3e-09) | 1.2e-09 (1e-09–2e-09) | 8.8e-10 | 11111 |
| heat | Adam, then self-scaled Broyden | 2.4e-03 (1e-03–8e-03) | 1.2e-02 (6e-04–1e-02) | 1.3e-05 (9e-06–2e-05) | 2.0e-09 (1e-09–2e-09) | 2.0e-10 | 8866 |
| heat | Adam, then self-scaled BFGS | 2.2e-03 (1e-03–8e-03) | 3.0e-03 (1e-03–5e-03) | 3.0e-05 (2e-05–3e-05) | 1.1e-08 (7e-09–1e-08) | 1.8e-09 | 5977 |
| heat | Adam, then L-BFGS | 2.2e-03 (1e-03–4e-03) | 1.8e-03 (7e-04–2e-02) | 1.3e-04 (9e-05–1e-04) | 6.4e-05 (4e-05–7e-05) | 6.4e-05 | 15826 |
| burgers | Levenberg–Marquardt | 6.1e-02 (2e-02–1e-01) | 2.3e-03 (8e-04–4e-03) | 1.1e-03 (7e-04–6e-03) | 1.0e-03 (4e-04–2e-03) | 1.1e-03 | 1417 |
| burgers | M3 filter, memory of 1 batch in 25 seen, at least 20 | 5.4e-02 (2e-02–2e-01) | 1.5e-03 (6e-04–6e-03) | 2.5e-04 (2e-04–3e-04) | 9.6e-05 (5e-05–2e-04) | 5.6e-05 | 11018 |
| burgers | The filter for 30 s, then self-scaled Broyden | 5.4e-02 (2e-02–2e-01) | 1.8e-03 (2e-04–6e-03) | 4.0e-03 (3e-03–1e-02) | 5.4e-03 (3e-03–6e-03) | 1.6e-03 | 7340 |
| burgers | The same, Broyden's first H from the filter's P | 5.3e-02 (2e-02–2e-01) | 1.4e-03 (2e-04–6e-03) | 1.1e-03 (5e-04–5e-03) | 9.6e-03 (3e-03–1e-02) | 2.1e-03 | 7349 |
| burgers | Adam for 10 s, then self-scaled Broyden | 3.2e-01 (3e-01–4e-01) | 2.7e-02 (6e-03–1e-01) | 4.5e-03 (2e-03–5e-02) | 1.0e-03 (7e-04–7e-03) | 6.7e-04 | 8123 |
| burgers | Self-scaled Broyden from the start | 9.7e-02 (8e-02–2e-01) | 5.2e-02 (3e-03–1e-01) | 4.0e-03 (8e-04–4e-03) | 3.6e-03 (4e-03–4e-03) | 7.0e-04 | 8559 |
| burgers | The same, the filter's weighting (5/3) | 3.2e-01 (2e-01–4e-01) | 2.9e-01 (2e-01–3e-01) | 2.3e-01 (6e-03–3e-01) | 1.8e-01 (7e-04–2e-01) | 1.4e-01 | 8314 |
| burgers | M3 filter, growing memory, the full loss's weighting (10) | 1.8e-02 (2e-02–5e-02) | 1.8e-03 (3e-04–4e-03) | 3.5e-04 (2e-04–4e-04) | 9.3e-05 (3e-05–2e-04) | 5.1e-05 | 10961 |
| burgers | Adam, then self-scaled Broyden | 3.2e-01 (3e-01–4e-01) | 9.5e-02 (6e-02–2e-01) | 4.1e-02 (2e-02–1e-01) | 2.2e-03 (1e-03–4e-03) | 8.2e-04 | 6005 |
| burgers | Adam, then self-scaled BFGS | 3.2e-01 (3e-01–4e-01) | 1.0e-01 (7e-02–2e-01) | 2.2e-02 (2e-02–1e-01) | 1.0e-03 (1e-03–2e-02) | 9.1e-04 | 6232 |
| burgers | Adam, then L-BFGS | 3.2e-01 (3e-01–4e-01) | 1.0e-01 (6e-02–2e-01) | 3.5e-02 (3e-02–4e-02) | 1.0e-02 (8e-03–3e-02) | 5.5e-03 | 15603 |
| kdv | Levenberg–Marquardt | 1.4e-04 (1e-04–5e-04) | 9.7e-06 (5e-06–1e-05) | 4.8e-06 (2e-06–6e-06) | 1.3e-06 (5e-07–2e-06) | 8.9e-07 | 1111 |
| kdv | M3 filter, memory of 1 batch in 25 seen, at least 20 | 6.8e-06 (6e-06–1e-05) | 8.6e-08 (6e-08–2e-07) | 8.3e-08 (3e-08–2e-07) | 1.1e-07 (8e-08–1e-07) | 2.8e-08 | 8932 |
| kdv | The filter for 30 s, then self-scaled Broyden | 8.6e-06 (7e-06–1e-05) | 9.0e-08 (5e-08–1e-07) | 2.0e-07 (5e-08–2e-07) | 3.8e-08 (3e-08–1e-07) | 1.5e-08 | 6664 |
| kdv | The same, Broyden's first H from the filter's P | 1.2e-05 (5e-06–1e-05) | 8.6e-08 (5e-08–2e-07) | 1.9e-07 (3e-08–2e-07) | 4.9e-08 (6e-09–2e-07) | 4.1e-08 | 8031 |
| kdv | Adam for 10 s, then self-scaled Broyden | 7.1e-02 (3e-02–8e-02) | 1.5e-06 (1e-06–2e-06) | 2.3e-08 (1e-08–4e-08) | 5.1e-10 (3e-10–8e-10) | 3.5e-10 | 11482 |
| kdv | Self-scaled Broyden from the start | 3.9e-05 (3e-05–8e-05) | 2.7e-07 (2e-07–1e-06) | 8.0e-09 (2e-09–2e-08) | 1.1e-09 (7e-10–2e-09) | 1.1e-09 | 12545 |
| kdv | The same, the filter's weighting (5/3) | 2.0e-05 (2e-05–6e-05) | 1.9e-07 (1e-07–8e-07) | 5.1e-09 (3e-09–2e-08) | 3.6e-10 (3e-10–4e-10) | 3.6e-10 | 12779 |
| kdv | M3 filter, growing memory, the full loss's weighting (10) | 8.1e-06 (4e-06–9e-06) | 5.0e-07 (4e-07–8e-07) | 4.9e-07 (4e-07–8e-07) | 5.8e-08 (5e-08–9e-08) | 8.6e-08 | 8948 |
| kdv | Adam, then self-scaled Broyden | 6.7e-02 (5e-02–8e-02) | 1.5e-02 (1e-02–2e-02) | 1.5e-04 (1e-04–2e-04) | 8.4e-09 (4e-09–1e-08) | 3.4e-10 | 6277 |
| kdv | Adam, then self-scaled BFGS | 6.9e-02 (4e-02–8e-02) | 2.1e-02 (1e-02–3e-02) | 2.6e-04 (2e-04–4e-04) | 4.3e-08 (4e-08–8e-08) | 8.7e-09 | 4744 |
| kdv | Adam, then L-BFGS | 6.9e-02 (3e-02–8e-02) | 2.0e-02 (1e-02–3e-02) | 2.6e-03 (2e-03–3e-03) | 8.0e-04 (8e-04–9e-04) | 7.7e-04 | 8183 |

With Adam for 30 % of the budget the Adam phase lasts 54 s, so those self-scaled rows at 10 s and 30 s are Adam's. Steps count the second method only, for the methods made of two.

### Fronts, 180 seconds

| Equation | Method | 10 s | 30 s | 60 s | 120 s | Last 30 s, mean | Steps |
|---|---|---|---|---|---|---|---|
| allen_cahn | Levenberg–Marquardt | 5.8e-01 (5e-01–6e-01) | 7.7e-02 (7e-02–2e-01) | 1.7e-02 (2e-03–2e-02) | 4.3e-04 (1e-04–8e-04) | 2.0e-04 | 1412 |
| allen_cahn | M3 filter, memory of 1 batch in 25 seen, at least 20 | 6.4e-02 (3e-02–1e-01) | 2.5e-05 (1e-05–3e-05) | 5.3e-06 (4e-06–8e-06) | 3.8e-06 (3e-06–4e-06) | 3.3e-06 | 10792 |
| allen_cahn | M3 filter, growing memory, the full loss's weighting (10) | 1.1e-02 (3e-03–5e-02) | 2.3e-05 (2e-05–7e-05) | 1.5e-05 (1e-05–4e-05) | 1.2e-05 (5e-06–1e-05) | 6.9e-06 | 10777 |
| allen_cahn | Self-scaled Broyden from the start | 5.1e-01 (5e-01–5e-01) | 1.8e-02 (8e-03–3e-01) | 2.2e-04 (7e-05–3e-01) | 2.9e-05 (1e-05–3e-01) | 1.1e-05 | 7992 |
| allen_cahn | The same, the filter's weighting (5/3) | 5.1e-01 (3e-01–5e-01) | 4.7e-03 (5e-03–6e-03) | 9.0e-05 (7e-05–1e-04) | 1.6e-05 (1e-05–3e-05) | 9.1e-06 | 8470 |
| allen_cahn | Adam for 10 s, then self-scaled Broyden | 1.0e+00 (9e-01–1e+00) | 1.5e-01 (4e-02–4e-01) | 1.2e-03 (2e-04–4e-01) | 1.1e-04 (3e-05–4e-01) | 1.9e-05 | 7789 |
| allen_cahn | Adam, then self-scaled Broyden | 1.0e+00 (9e-01–1e+00) | 9.3e-01 (9e-01–1e+00) | 5.6e-01 (5e-01–6e-01) | 2.7e-01 (1e-04–4e-01) | 3.2e-01 | 5661 |
| allen_cahn | Adam, then L-BFGS | 1.0e+00 (9e-01–1e+00) | 9.3e-01 (9e-01–1e+00) | 5.6e-01 (6e-01–6e-01) | 5.2e-01 (4e-01–5e-01) | 5.2e-01 | 15532 |
| burgers_sharp | Levenberg–Marquardt | 2.4e-01 (1e-01–3e-01) | 1.4e-01 (9e-02–2e-01) | 1.5e-01 (7e-02–2e-01) | 1.5e-01 (7e-02–2e-01) | 5.5e-02 | 1410 |
| burgers_sharp | M3 filter, memory of 1 batch in 25 seen, at least 20 | 3.1e-01 (3e-01–5e-01) | 7.9e-01 (5e-01–1e+00) | 9.4e-01 (8e-01–2e+00) | 9.5e-01 (9e-01–1e+00) | 1.1e+00 | 11513 |
| burgers_sharp | M3 filter, growing memory, the full loss's weighting (10) | 1.7e-01 (2e-01–3e-01) | 2.1e-01 (2e-01–1e+00) | 1.7e-01 (2e-01–1e+00) | 1.4e-01 (9e-02–1e+00) | 1.0e+00 | 11270 |
| burgers_sharp | Self-scaled Broyden from the start | 2.4e-01 (2e-01–2e-01) | 2.7e-01 (2e-01–3e-01) | 2.5e-01 (2e-01–3e-01) | 2.4e-01 (2e-01–3e-01) | 2.4e-01 | 8016 |
| burgers_sharp | Adam for 10 s, then self-scaled Broyden | 3.7e-01 (3e-01–4e-01) | 2.3e-01 (2e-01–3e-01) | 2.2e-01 (2e-01–3e-01) | 2.3e-01 (2e-01–3e-01) | 2.5e-01 | 7560 |

All on mains power with Low Power Mode off, but not on an idle machine: ordinary desktop use ran alongside. Because the training path does not depend on the clock except through the length of a timed Adam phase, the step count of a rerun measures its speed; runs found more than 10 % slow were rerun. One was slow twice and is kept: Adam then self-scaled Broyden on Allen–Cahn, seed 2, at 0.85 of full speed (5,104 steps against 5,975).

The figures are `results/runs_3x32.png`, `results/long/runs_3x32.png`, `results/wide/runs_4x64.png`, `results/forget/runs_3x32.png`, `results/wide_grow/runs_4x64.png`, `results/baselines/runs_3x32.png` and `results/fronts/runs_3x32.png` after `plot.py`.

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
19. **Starting the covariance from the last linearization does not shorten the pause.** With $P = (J^\top J + I)^{-1}$ at the switch the hybrid is level or behind the cold restart at every reading: at 120 s, 1.7e-08 against 5.8e-09 on heat, 1.4e-04 against 3.1e-05 on Burgers, 1.2e-06 against 2.6e-07 on KdV. A tight $P$ makes the first updates small; the wide prior is what lets the filter move.
20. **At 180 s the filter is far below the usual baseline on all three.** Adam then L-BFGS ends at 5.2e-05 on heat, 7.0e-03 on Burgers and 9.8e-04 on KdV. The full filter is 35,000, 350 and 9,800 times lower. Against self-scaled Broyden, a stronger baseline, the result is mixed (findings 33–35).

### Stage 2b: scaling to 12,737 weights

21. **The per-layer filter does not scale.** On the larger network it takes 1,700 to 2,850 steps in 600 s and ends 40 to 50 times above Levenberg–Marquardt on heat and KdV, and 4 times above on Burgers.
22. **The low-rank filter takes the steps, and stalls.** With 64 directions it runs 13,000 to 26,000 steps in 600 s at 12,737 weights.
    - **Early**: at 60 s it is 17 times below Levenberg–Marquardt on heat (1.3e-06 against 2.2e-05); at 180 s 4.5 times below on KdV.
    - **Late**: level with Levenberg–Marquardt at 600 s on all three (4.1e-07, 5.4e-04, 3.3e-06 against 3.4e-07, 5.9e-04, 2.8e-06), and 120, 6 and 260 times below Adam then L-BFGS.
    - **On the small network** it stops near the same levels (4.5e-07, 9.1e-04, 6.5e-06 over the last 30 s), where the full filter goes on to 1.5e-09, 2.0e-05, 1.0e-07.
23. **What is lost is the long memory.** Keeping only the strongest directions throws away the weak information that thousands of batches build up, and the full filter's late gains come from it. Rank in a 30 s scan on one seed: 64 best; 16 and 32 noisier; 128 and 256 slower, with 256 equal to the full filter.
    - **Neither more directions nor a diagonal brings it back** (one seed, 180 s). With 256 directions: 3e-07 on heat, 8.6e-07 on KdV. With the dropped information kept on a diagonal: 2.2e-06 and 1.5e-05, worse than without, because the extra information makes the steps too short.
24. **The low-rank filter is still the best method found at 12,737 weights in the first minutes**, and the only filter that keeps up with Levenberg–Marquardt at that size. Closing the gap to the full filter's end point is the open problem of M3.

### Stage 2b: the forgetting factor

25. **Forgetting sets the filter's pace, not its memory.** Dividing $P$ by the factor $f$ each step lets the variance grow back in the directions the stream stops informing; $1/(1-f)$ batches is the memory.
    - **A longer memory is worse**: at $f = 0.999$ (1,000 batches), 1.8e-05 on heat and 4.2e-04 on KdV after 180 s (one seed), against 1.5e-09 and 1.0e-07 at 0.995.
    - **A shorter memory is faster** and ends level: at $f = 0.98$, 2.8e-09 on heat and 1.8e-07 on KdV after 180 s (one seed).
26. **At $f = 0.95$ the slow first minute is gone on heat and KdV.** The filter is ahead of Levenberg–Marquardt from the first reading: at 30 s, 1.6e-08 against 4.5e-06 on heat (280 times) and 4.5e-07 against 2.5e-05 on KdV (55 times). Over the last 30 s it is 76 and 15 times below, and 7 and 5 times below the filter at 0.995.
27. **Burgers wants the long memory late.** At $f = 0.95$ it is level with Levenberg–Marquardt to 60 s and ends 3.7 times below it (5.2e-04), but 3 times above the filter at 0.995 (1.7e-04), with one seed at 1.8e-03. At 0.9 it degrades, with 22 refused updates.
28. **So the factor should change along the run**: short memory while the model is far off, long memory near the end.
29. **A memory that grows with the stream gets both.** Remembering 1 batch in 25 of those seen, at least 20, it is as fast as $f = 0.95$ for the first 30 s on all three equations, and over the last 30 s it is the best or level with the best:
    - **Heat**: 1.5e-09, 150 times below Levenberg–Marquardt.
    - **Burgers**: 1.1e-04, 17 times below Levenberg–Marquardt, 1.5 times below $f = 0.995$ and 5 times below $f = 0.95$.
    - **KdV**: 1.3e-07, 12 times below Levenberg–Marquardt, level with $f = 0.95$ and 4 times below $f = 0.995$.

    It has no constant tied to the budget. The 20 and the 25 were set once, from the two fixed factors, and not scanned.
30. **At 12,737 weights the growing memory does not rescue the full filter.** It takes 300 to 570 steps in 300 s and ends 20 times behind Levenberg–Marquardt on heat, 70 times on Burgers, level on KdV.
31. **The low-rank filter with the growing memory wins the first minute and ends level.** At 30 s it is 7 times ahead of Levenberg–Marquardt on heat and 30 times on KdV; over the last 50 s it is within a factor of 2.5 of it, either way, on all three.
32. **With this optimizer the small network is the better model.** The 2,241-weight network trained by the filter for 180 s (on battery) is 230 times (heat), 5 times (Burgers) and 20 times (KdV) below the 12,737-weight network trained by Levenberg–Marquardt for 600 s on mains power. The filter's accuracy needs many cheap steps, and a small network gives them.

### The strongest published baseline

33. **Self-scaled Broyden is a far stronger baseline than L-BFGS.** After the same 54 s of Adam, over the last 30 s:
    - **Self-scaled Broyden**: 2.0e-10 on heat, 8.2e-04 on Burgers, 3.4e-10 on KdV.
    - **Self-scaled BFGS**: 1.8e-09, 9.1e-04, 8.7e-09.
    - **L-BFGS**: 6.4e-05, 5.5e-03, 7.7e-04.

    Only the quasi-Newton update differs. Finding 20 and the R1 verdict were measured against the weakest of the three.
34. **Late on the smooth equations it beats the filter.** Over the last 30 s it is 5 times below the filter on heat (2.0e-10 against 9.6e-10) and 80 times below on KdV (3.4e-10 against 2.8e-08). It stays below the filter from 85 s on KdV and from 129 s on heat, that is 31 s and 75 s after its own start.
35. **Early, and on Burgers, the filter wins.**
    - **Early**: at 30 s the filter is at 8.0e-09 on heat and 8.6e-08 on KdV, while self-scaled Broyden is still in Adam at 1e-02. At 60 s, 6 s after the switch, it is 1.3e-05 and 1.5e-04 against the filter's 1.7e-09 and 8.3e-08.
    - **Burgers**: the filter ends at 5.6e-05, below what the reference can resolve and at least 8 times below self-scaled Broyden (8.2e-04). Self-scaled Broyden ends level with Levenberg–Marquardt (1.1e-03).
36. **The filter flattens where self-scaled Broyden does not.** On KdV the filter holds near 1e-07 from 30 s to 120 s and only then falls to 2.8e-08. Self-scaled Broyden goes through that level without slowing. One explanation, not yet tested: each filter step sees 64 fresh rows, so its steps carry the sampling noise of the batch, while self-scaled Broyden steps on one fixed set of 2,512 rows for 500 steps.
37. **The two strengths look as if they fit together.** The filter reaches in 20 to 30 s the level that self-scaled Broyden reaches 30 to 70 s after its start, and self-scaled Broyden is what goes on falling from there. Findings 39–41 test the filter first, then self-scaled Broyden from its weights.
38. **Self-scaled Broyden needs no Adam phase here.** From the first step it is at 6.9e-08 on heat and 2.7e-07 on KdV after 30 s, and ends at 3.4e-10 and 1.1e-09. After 10 s of Adam it ends at 2.4e-10 and 3.5e-10. On Burgers all three versions end between 6.7e-04 and 8.2e-04. The paper's Adam phase of 30 % only delays it. Against the best of the three at each moment:
    - **Heat**: the filter is ahead until 68 s, 30 times at 20 s and 7 times at 30 s.
    - **KdV**: the filter is ahead until 37 s, 4 times at 20 s. After that self-scaled Broyden leads, by up to 200 times.
    - **Burgers**: the filter is ahead at every reading, by 7 to 60 times.
39. **The filter for 30 s, then self-scaled Broyden, is the best method on heat.** It holds the filter's lead early, stays within 2 times of the best self-scaled Broyden from 60 to 120 s, and ends at 8.6e-11, 2 to 4 times below every version of it. With the filter's $P$ as the first $H$ it ends at 1.8e-10: no gain.
40. **On KdV and Burgers the hybrid fails.**
    - **KdV**: it ends at 1.5e-08 (seeds 1.0e-09 to 4.8e-08), against 3.5e-10 for self-scaled Broyden after 10 s of Adam.
    - **Burgers**: the error rises after the switch, from 1.8e-03 at 30 s to 5.4e-03 at 120 s. It ends 30 times above the filter alone.
41. **Why the handoff fails** (diagnostics on one seed; not a timed run):
    - **Not the loss weights**: self-scaled Broyden given the filter's own weighting stalls the same way on KdV (1.7e-07 to 2.3e-07 in 2,400 steps).
    - **Not a fresh $H$**: on its own path, self-scaled Broyden restarted from the scaled identity at 2.8e-07 converges as fast as without the restart (2.8e-09 against 2.4e-09 after 2,400 steps).
    - **Not the size of the weights on KdV**: after 5 s of the filter the weight norm is Adam's (7.4), and self-scaled Broyden is slow from there too (3.9e-08 after 2,400 steps).
    - **The filter trains a network of higher rank.** At the same error on KdV (about 2e-07) the Jacobian on 2,512 rows has 213 singular values above $10^{-8}$ after the filter, against 109 after Adam then self-scaled Broyden. The residual left sits a decade lower in the spectrum: 73 % of it in $\sigma \in (10^{-8}, 10^{-6}]$, against 92 % in $(10^{-7}, 10^{-5}]$.
    - **From there no second method moves the error.** Damped Gauss–Newton steps by SVD on fresh samples of 5,024 rows, the damping chosen on a second sample, lower the loss by a fifth in 6 steps and leave the error at 1.6e-07. Undamped steps raise the loss on the second sample.
    - **On Burgers the filter's weights reach a norm of 2,359**, against 7 to 19 elsewhere: saturated units that draw the front. Self-scaled Broyden on its fixed 2,512 points then raises the loss on fresh points 8 times. It fits the points and not the equation between them.
42. **So the filter is fast because it uses more of the network, and that is also why it is hard to finish.** It reaches low errors early by moving weakly determined weights, and leaves the last error in directions that are weak for every method tried here.
43. **The filter's lead on Burgers is the optimizer's, not its weighting's.** With its rows weighted 60 so that it minimizes the same loss as every other method, the filter ends at 5.1e-05 on Burgers. That is level with its own weighting (5.6e-05) and 14 times below self-scaled Broyden from the start (7.0e-04), and it is ahead at every reading. On heat the weighting changes nothing (8.8e-10). On KdV the full loss's weighting makes the filter slower and 3 times worse at the end (8.6e-08 against 2.8e-08): there the light weighting helped.
44. **Self-scaled Broyden with the filter's light weighting** fails on Burgers on two seeds of three (1.4e-01 and 1.7e-01; the third 1.5e-03), is level on heat (3.5e-10), and is its best on KdV (3.6e-10, against 1.1e-09 with the weights of 10).

### Fronts

45. **On Allen–Cahn the filter is the best method at every reading.** The strongest baseline at every moment is self-scaled Broyden from the start with the filter's light weighting, so the two minimize the same loss:

    | | 20 s | 30 s | 60 s | 120 s | 180 s |
    |---|---|---|---|---|---|
    | Filter | 1.2e-04 | 2.5e-05 | 5.3e-06 | 3.8e-06 | 3.0e-06 |
    | Best self-scaled Broyden | 3.1e-02 | 4.7e-03 | 9.0e-05 | 1.6e-05 | 7.0e-06 |
    | Ratio | 250 | 190 | 17 | 4 | 2.3 |

    Over the last 30 s: 3.3e-06 against 9.1e-06, and 2.0e-04 for Levenberg–Marquardt, 60 times above. All three filter seeds end between 2.7e-06 and 3.5e-06.
46. **On the full loss's weighting the order holds.** The filter weighted 10 is ahead of self-scaled Broyden weighted 10 at every reading and ends 2 times below it (6.9e-06 against 1.1e-05). Both do better with the light weighting late, the filter by 2 times.
47. **Self-scaled Broyden sometimes lands on a wrong solution; the filter never does.** Runs that end above 0.09, out of three seeds:

    | Method | Wrong solution |
    |---|---|
    | Filter, either weighting | 0 of 6 |
    | Levenberg–Marquardt | 0 (but at 1e-04 to 2e-04) |
    | Self-scaled Broyden from the start | 1 |
    | The same, the filter's weighting | 0 |
    | Adam for 10 s, then self-scaled Broyden | 1 |
    | Adam for 54 s, then self-scaled Broyden | 2 |
    | Adam, then L-BFGS | 3 |

    The wrong solution is the unstable equilibrium $u = 0$ (diagnosed on seed 2 of self-scaled Broyden from the start, not a timed run):
    - It fits the initial line to 5e-05. Then the middle decays to 0 by $t \approx 1$, where the true solution grows from 0.04 to 1 near $x = \pm 0.25$.
    - $u = 0$ solves the equation exactly, so the loss pays only in thin layers: near $t = 0$, and where the middle meets the $-1$ outside. The error grows with time: 8e-02 at $t = 0.2$, 0.5 at $t = 3$, 0.8 at $t = 5$.
    - This is the causality failure of PINNs (Wang, Sankaran and Perdikaris, 2022, arXiv 2203.07404).
    - The filter on the same seed is within 2e-06 to 5e-06 at every time. Why it avoids the equilibrium is not tested.
48. **Whether Broyden fails depends on where it starts.** A timed Adam phase gives fewer Adam steps on a slower machine, and so a different start. Seed 2 after 10 s of Adam ended at 0.57 in a run 25 % slow and at 1.5e-05 at full speed. Seed 2 after 54 s of Adam reached 2.1e-05 at full speed in the first sweep and failed in both slowed reruns.
49. **The first reference hid the gap.** Graded on a solver grid 8 times finer than the output grid, whose own error is about 6e-06, the filter and self-scaled Broyden both ended near 7e-06 and looked level. On a grid 32 times finer (error about 4e-07) the filter ends at 3.3e-06 and Broyden at 9.1e-06. A tie near a reference's own error is not a tie.
50. **Burgers at $\nu = 0.003/\pi$ defeats every method at this size.**
    - No method ends below 3e-02. Levenberg–Marquardt is the least bad: 5.5e-02 over the last 30 s, one seed at 0.28.
    - Self-scaled Broyden sits at 0.22 to 0.27 from 20 s to the end, with or without Adam.
    - The filter is the worst. It is at 0.31 at 10 s and near 1 from 40 s on, with either weighting. Why its error rises is not diagnosed.
    - The front, $u \approx -\tanh(x / 2\nu)$, is about $4\nu \approx 4 \cdot 10^{-3}$ wide. The 2,000 points are about 0.03 apart, 8 times more: a full sample puts about 3 points in the front, a filter batch one point every 15 batches.
    - So at this viscosity the question is where the points go before how the optimizer steps. A stream drawn from the residual is the next test.

## Verdict against the plan

| Move | Gate | Result |
|---|---|---|
| Stage 0, Levenberg–Marquardt | $10^{-6}$ on heat on one core | Passed |
| M1, residual flow | Beats stage 0 on two of three | Dropped: no gain on any |
| M4, block Kaczmarz | Beats stage 0 on two of three | Dropped as the main optimizer. Kept as a candidate opening move for the agent (M9) |
| M5, variable projection | Beats stage 0 on two of three; a gain on the linear problem | Kept, with a warm start: a large gain on heat, level on the other two |
| M3, filtering | Beats stage 0 on two of three; works without a full-rank covariance | **Kept, and the lead candidate**: ahead on all three at 180 s. Without the full $P$: a block per layer holds on heat and KdV and loses a factor of 20 on Burgers, and is too slow at 12,737 weights; a block per neuron fails; 64 directions keep up with Levenberg–Marquardt at 12,737 weights but lose the full filter's late lead. A memory that grows with the stream (1 batch in 25, at least 20) removes the slow start and is best or level at the end on all three. Against self-scaled Broyden with little or no Adam: ahead until 68 s on heat and 37 s on KdV, and over the whole run on Burgers on the same loss (14 times at the end). Followed by self-scaled Broyden it is the best method on heat; that handoff fails on KdV and Burgers. On Allen–Cahn ahead at every reading, 2.3 times at the end, and never on a wrong solution; at $\nu = 0.003/\pi$ every method fails and the filter most |
| M6, continuation | A cold run is not as good | Dropped at this viscosity |
| M8, multipliers | Beats stage 0 on two of three | Dropped |

Roadmap bet R1 asks for 100 times below Adam then L-BFGS. At 60 s Levenberg–Marquardt meets it on heat and KdV and not on Burgers. At 180 s the filter meets it on all three (finding 20), on this network and these three equations only.

That bar is too low. Self-scaled Broyden, published in 2024, meets it on heat and KdV by itself (findings 33–34). The bar R1 should be held to is the best of the published methods at every moment of the run:

$$
\varepsilon_{\text{new}}(T) \le \min_{b \in \text{baselines}} \varepsilon_b(T) \quad \text{for every } T
$$

The strongest baseline is self-scaled Broyden with little or no Adam first (finding 38), not the paper's 30 % of Adam. Against it, on the same loss:

| Equation | Best method here | Meets the bar |
|---|---|---|
| Heat | The filter for 30 s, then self-scaled Broyden | Yes: within 2 times of the best at every reading, 2 to 4 times below it at the end |
| Burgers | The filter alone | Yes: 7 to 60 times below at every reading, 14 times at the end |
| KdV | The filter until 37 s | No: after 37 s self-scaled Broyden leads, by up to 200 times |
| Allen–Cahn | The filter alone | Yes: 2.3 to 250 times below at every reading from 20 s, and no wrong solution on any seed (finding 47) |
| Burgers, $\nu = 0.003/\pi$ | None | No method below 3e-02. Levenberg–Marquardt is the least bad; the filter is the worst |

No single method meets it on all five. The filter's lead is real on both equations with a moderate front and in the first half-minute of the smooth ones. On KdV it is not yet an optimizer that finishes. A front 3.3 times thinner than Burgers' usual one defeats every method at this size and with these points.

## What it does not show

- **Three seeds** (two at 12,737 weights), two network sizes, three equations in one space dimension. The plan asks for ten seeds and the nine equations of I17.
- **Dense linear algebra.** The cost per step grows with the cube of the smaller of rows and weights. Larger networks need the matrix-free solve.
- **Double precision in PyTorch.** The engine runs in single precision; errors near $10^{-8}$ will not survive it unchanged.
- **Baselines at their defaults.** Adam's rate and the L-BFGS history were not tuned per equation, and neither was anything in the new methods.
- **The hybrid's 20 s** was set once and not scanned.
- **The warm-up length of 60** was chosen once, on a 30 s test, and not scanned.
- **The filter's constants** (forgetting 0.995, prior 1, cap $10^{12}$, the factor 2 in the check) were set once. The cap was chosen from one seed at three values.
- **The full filter's cost grows with the square of the number of weights** per step and its memory likewise (40 MB here). The low-rank form scales and loses accuracy late; a Kronecker-factored form is untested.
- **The rank of 64** comes from one 30 s scan on one seed. **The forgetting of 0.95** comes from one 60 s scan on one seed; the growing memory's 20 and 25 were not scanned.
- **Low Power Mode**: the forgetting runs were made on battery at about half speed. They are compared only with controls from the same session.
- **Two runs were slowed by other load** (Burgers, per-layer filter, seed 1 at 4×64: 2,413 steps against 3,286; heat, low-rank filter, seed 1 at 4×64: 20,291 against 32,455). Both are kept.
- **SOAP is not run.** Self-scaled Broyden and BFGS follow the paper's protocol except the line search's constants, which the paper does not give (PyTorch's strong Wolfe search is used), and the two additions in the setup. The restart after a failed line search fired 800 to 5,200 times per run for self-scaled Broyden on heat and KdV, at most once on Burgers, never for self-scaled BFGS. Whether the method does better or worse without it is not tested.
- **The Adam phase of 30 %** is the length used for every hybrid here, not a tuned one. A shorter phase would start the self-scaled methods sooner.
- **The filter does not minimize the same loss as the others.** Its batch is 48 equation rows and 8 initial and 8 boundary rows, unscaled, with weight 10 on the last two. Per point, that is a weight of $80/48 \approx 1.7$ on the initial and boundary terms, against 10 in the loss of every other method:

  $$
  \mathbb{E}\big[\lVert r_{\text{batch}} \rVert^2\big] = 48\,\big(\mathbb{E}\, r_{\text{pde}}^2 + 1.7\, \mathbb{E}\, r_{\text{ic}}^2 + 1.7\, \mathbb{E}\, r_{\text{bc}}^2\big), \qquad L = \mathbb{E}\, r_{\text{pde}}^2 + 10\, \mathbb{E}\, r_{\text{ic}}^2 + 10\, \mathbb{E}\, r_{\text{bc}}^2
  $$

  Tested in finding 43: with rows weighted to match the full loss, the filter's Burgers and heat results hold; on KdV it is 3 times worse. The tables before the self-scaled baselines use the light weighting.
- **The Burgers reference** is the package's Cole–Hopf solution on a 256 by 101 grid; its own accuracy bounds what can be read below about 1e-4.
- **The Allen–Cahn reference** has its own error of about 4e-07 (a solver grid 32 times finer than the output; against 64 times finer, 2.9e-07). The best errors read here are 8 times above it.
- **The fronts runs shared the machine with desktop use.** Runs more than 10 % slow by their step count were rerun; one kept is 15 % slow (finding 48). The heat, Burgers and KdV tables were made on an idle machine.
