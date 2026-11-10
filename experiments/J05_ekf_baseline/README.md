# J05: a Kalman-filter baseline against the PINN twin

## Question

How does the classical estimator, an extended Kalman filter on a lumped model of the evaporator, compare with the PINN twin of J03 on the same records: estimation error, latency, tuning effort, and sensitivity to the model mismatches J04 identified?

## Setup

- **Plant and records:** exactly J03's: the J01 evaporator over 20 minutes, the steam wandering by $\pm 1$ K (recorded), the fouling drifting $R_f(t) = 2 \times 10^{-4}\, t/T$, outlet $X$ and $W$ every 5 s with noise 0.002 and 0.001, the same three noise seeds.
- **Estimator:** `pinnrl.process.ExtendedKalmanFilter` on the method-of-lines model with $n_z \in \{6, 11, 21, 51\}$ cells (`FallingFilmEvaporator.fouling_state_space`), $R_f$ appended to the state as a random walk of intensity $\sigma_{rw}^2$ per second, $\sigma_{rw} \in \{10^{-7}, 10^{-6}, 10^{-5}\}$ m²K/W per $\sqrt{\text{s}}$ (the truth's slope is $1.7 \times 10^{-7}$ per second), started at $R_f = 10^{-4}$ with a prior standard deviation of $10^{-4}$, like the PINN. The Jacobians by central finite differences, the covariance propagated with the matrix exponential, the state with LSODA; the Rauch–Tung–Striebel smoother gives the batch estimate, which is what the PINN (a whole-window fit) is comparable to. $R_f$ is clipped at zero.
- **Model mismatch:** the filter's steam record biased by $+0.3$ K, or its feed flow by $+1$ %, with 11 cells and $\sigma_{rw} = 10^{-6}$.
- **Metrics:** relative L2 of $R_f(t)$ after a 2-minute warm-up; the error at the end of the window; the lag of the filter behind the ramp (the mean of (truth − estimate) over the slope after 5 minutes); the fraction of the window inside the $\pm 2\sigma$ band; the run time on one CPU core.
- **Code:** `run.py` (14 s for all 42 runs), `plot.py`.

Rerun with:

```bash
python experiments/J05_ekf_baseline/run.py --seeds 3
python experiments/J05_ekf_baseline/plot.py
```

## Result

![J05](../../docs/assets/benchmarks/j05_ekf_baseline.png)

Mean over 3 seeds.

| Cells | $\sigma_{rw}$ | Filter rel. L2 | Filter end error | Lag | $\pm 2\sigma$ coverage | Smoother rel. L2 | Smoother end error | Time |
|---|---|---|---|---|---|---|---|---|
| 6 | $10^{-7}$ | 0.264 | 0.270 | +196 s | 0.08 | 0.202 | 0.270 | 0.1 s |
| 6 | $10^{-6}$ | 0.072 | 0.052 | −40 s | 0.90 | 0.130 | 0.052 | 0.1 s |
| 6 | $10^{-5}$ | 0.172 | 0.180 | −75 s | 1.00 | 0.155 | 0.180 | 0.1 s |
| 11 | $10^{-7}$ | 0.326 | 0.310 | +249 s | 0.04 | 0.211 | 0.310 | 0.2 s |
| 11 | $10^{-6}$ | **0.035** | **0.022** | +6 s | 1.00 | 0.064 | 0.022 | 0.2 s |
| 11 | $10^{-5}$ | 0.140 | 0.151 | −33 s | 1.00 | 0.103 | 0.151 | 0.2 s |
| 21 | $10^{-7}$ | 0.356 | 0.329 | +273 s | 0.01 | 0.223 | 0.329 | 0.3 s |
| 21 | $10^{-6}$ | 0.048 | 0.008 | +28 s | 1.00 | 0.036 | 0.008 | 0.3 s |
| 21 | $10^{-5}$ | 0.131 | 0.137 | −13 s | 1.00 | 0.086 | 0.137 | 0.4 s |
| 51 | $10^{-7}$ | 0.374 | 0.339 | +287 s | 0.00 | 0.232 | 0.339 | 0.8 s |
| 51 | $10^{-6}$ | 0.062 | 0.007 | +40 s | 0.98 | **0.023** | **0.007** | 0.8 s |
| 51 | $10^{-5}$ | 0.129 | 0.128 | −1 s | 1.00 | 0.081 | 0.128 | 0.8 s |

With model mismatch (11 cells, $\sigma_{rw} = 10^{-6}$): steam $+0.3$ K gives a filter rel. L2 of 0.129 and an end error of 0.116 (smoother 0.196); feed $+1$ % gives 0.134 and 0.081 (smoother 0.073).

For reference, the PINN twin of J03 on the same records: rel. L2 0.09–0.16, end error 0.10–0.49, field error 2 %, 290–340 s per run.

## Conclusion

1. **With the model known, the Kalman filter beats the PINN twin by a wide margin at a thousandth of the cost.** Eleven cells and a tuned random walk track the drift to 3.5 % in L2 and 2 % at the end of the window in 0.2 s, against the PINN's 9–16 % and 10–49 % in five minutes; the smoother on 51 cells reaches 2.3 % and 0.7 %, close to J04's 5 % data limit for the end value (the smoother uses the whole window, as the Fisher bound assumes, and its model is nearly the plant's). The PINN's deficit is the surrogate limit of J03: the filter has no surrogate, it has the equations.
2. **The lumped model is good enough, and the tuning is one number.** Six cells (the coarsest, a 4 % error on the step response in J01) still give 7 %, and 11 cells are within a factor of two of 51. The only knob is $\sigma_{rw}$: a decade too small and the filter lags the ramp by 4 minutes (30 % error, the band misses the truth); a decade too large and it follows the noise (13–17 %). The right value is the drift's expected rate over the sampling interval, which a plant engineer knows roughly; the PINN's recipe has more knobs (steps, weights, architecture, the unknown's bounds) and no such rule.
3. **Both estimators are only as good as their inputs.** A 0.3 K steam bias costs the filter 12 % at the end (J04 predicted 18 % for the batch estimate; the random walk lets the filter absorb part of it as a slow drift) and a 1 % feed bias 8 %. These are the same numbers that bound the PINN, because they are the data's, not the estimator's: the sensors, not the method, set the floor once the model is right.
4. **Where the PINN earns its place is not here.** This benchmark has a known model structure with one unknown signal, which is the Kalman filter's home ground. The twin's case is the model that is partly unknown: a drying-kinetics closure (J07/P20), a heat-transfer correlation off its range, a sensor model with an unknown delay, or a field in space that a lumped model cannot carry. The next comparisons should be made there, with the EKF kept as the control on every one.

**Consequences for the backlog:**
- J06 (calibrated intervals): the filter's $\pm 2\sigma$ band is a candidate interval (coverage 1.00 at the tuned setting, over-wide) to put next to the PINN's Laplace or swarm interval.
- J07 (spray dryer): include an unknown closure, so the comparison tests what the PINN is for.
- K15/J09 (measurement design): the Fisher machinery of J04 applies to both estimators.
- The EKF is a `pinnrl.process` module; moving-horizon estimation (the roadmap's other classical estimator) can reuse its state-space form.

## Moving-horizon estimation (J05b)

The roadmap's other classical estimator on the same records: `pinnrl.process.MovingHorizonEstimator`, which re-solves the last $N$ records at every solve as a bounded least-squares problem (the state at the window's start plus one disturbance of $R_f$ per record, the arrival cost from an EKF run alongside). What it can do that the filter cannot is constrain: here, that the fouling only grows (disturbances $\ge 0$).

- **Setup:** J05's plant, records and seeds; both estimators on the 11-cell model with $R_f$ a random walk, the same $Q$, $R$ and prior (the physical states without process noise, so that inside a window they follow the model); the MHE solved at every second record. Cases: *tuned* ($\sigma_{rw} = 10^{-6}$, $N$ = 6, 12, 24 records, i.e. 30 s to 2 min); *monotone* (the same with the growth constraint, $N = 12$); *noisy* ($\sigma_{rw} = 10^{-5}$, where the filter follows the noise, with and without the constraint); *bad start* ($R_f$ started at $4 \times 10^{-4}$ with a prior standard deviation of $2 \times 10^{-5}$, a confident wrong prior).
- **Metrics:** J05's, with the 90 % band's coverage along the window, and the *roughness* of the estimate: the standard deviation of its increments in units of the truth's increment per step (0 for a perfect ramp).
- **Code:** `run_mhe.py`.

```bash
python experiments/J05_ekf_baseline/run_mhe.py --seeds 3 --workers 3
```

Mean over 3 seeds:

| Case | Estimator | Rel. L2 | End error | 90 % band coverage | Roughness | Time |
|---|---|---|---|---|---|---|
| tuned | EKF | 0.035 | 0.022 | 0.99 | 2.5 | 0.4 s |
| | MHE, $N = 6$ | 0.034 | 0.021 | 0.99 | 1.7 | 4.1 s |
| | MHE, $N = 12$ | 0.034 | 0.021 | 0.99 | 1.7 | 7.9 s |
| | MHE, $N = 24$ | 0.034 | 0.021 | 0.99 | 1.7 | 13.7 s |
| monotone | MHE, $N = 12$, growth only | 0.033 | 0.021 | 1.00 | 1.6 | 8.9 s |
| noisy | EKF | 0.140 | 0.151 | 1.00 | 22.9 | |
| | MHE, $N = 12$ | 0.138 | 0.152 | 1.00 | 13.8 | |
| | MHE, $N = 12$, growth only | 0.126 | 0.169 | 1.00 | 8.0 | |
| bad start | EKF | 0.035 | 0.022 | 0.99 | 2.5 | |
| | MHE, $N = 12$ / $24$ | 0.034 | 0.021 | 0.99 | 1.7 | |

The MHE is solved at 120 of the 240 records and takes about three linearizations of the window per solve.

Both estimators are within $0.2 \times 10^{-4}$ of the truth 35 s (EKF) and 40–50 s (MHE) after the bad start.

### Conclusion (J05b)

1. **On this model the moving horizon is the Kalman filter, at 10 to 35 times the cost.** Error, end error and band are the same to the second digit for every horizon: the evaporator's response to $R_f$ is nearly linear over a window (J04's profile is the Fisher parabola), and for a linear model the two estimators coincide exactly, which the package tests for horizons of 1, 5 and 40. Re-linearizing along the window buys nothing when one linearization per record was already right. The horizon is not a tuning knob here either: with the arrival cost from the filter, 30 s and 2 min give the same estimate.
2. **The growth constraint smooths, it does not correct.** With the tuned random walk the constraint is rarely active and changes nothing. With the walk ten times too loose it removes two thirds of the noise-following (roughness 23 → 8, against 14 without the constraint) and a tenth of the L2 error, but the end error is no better (0.17 against 0.15): a one-sided constraint on noisy increments ratchets upward, and the bias it adds offsets the variance it removes. A constraint is not a substitute for the right process noise.
3. **A confident wrong prior costs both about half a minute**, the residence time of the tube twice over: the outlet records overrule the prior as soon as the wrong fouling has been flushed through, and the window does not get there sooner because its arrival cost is the filter's.
4. **Where MHE should earn its cost is a model whose nonlinearity matters within a window**, such as the droplet's critical-moisture transition (J07), or constraints that bind in normal operation. For the fouling monitor the EKF with the trend model of J06 is the estimator; the MHE stays in the package as the control for those cases and as the online form of the batch estimator.
