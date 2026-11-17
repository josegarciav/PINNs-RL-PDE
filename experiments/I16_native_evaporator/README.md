# I16: the evaporator's reference solve in C++

**Question.** The method-of-lines right-hand sides of the process models are Python callbacks inside `solve_ivp`, and a batch fit of J06 took 4–14 s. What does a compiled integrator buy per solve, per Gauss–Newton fit and per filter run?

**Short answer.** Per solve, 3–8× on smooth inputs and 50–100× on a plant record, because SciPy's BDF collapses on the kinks of a linearly interpolated record (0.7–6.5 s a solve) where the compiled integrator does not (7–108 ms). A J06-style batch fit goes from 106 s to 0.7 s with the same estimate. The fouling monitor gains only 1.6× (1.9 → 1.2 s for 360 records), because its time is now in the filter's finite-difference Jacobian and matrix exponential, not in the solve. The time was never in the right-hand side itself (7 µs a call): SciPy's BDF spent it in dense factorisations of a Jacobian that is block lower-bidiagonal, and in the steps the kinks force.

## Setup

`pinnrl/native/evaporator.cpp`, compiled on first use like the jets kernel: the upwind model of `FallingFilmEvaporator.mol_rhs` integrated by the L-stable SDIRK method of order 4 with 5 stages and γ = 1/4 (Hairer & Wanner, table IV.6.5) and its embedded third-order estimate. Because cell *i* depends on itself and on cell *i − 1* only, each implicit stage is solved exactly by one sweep downstream with a 2 × 2 Newton iteration per cell (2–3 iterations, started from the previous stage's slope), and the error estimate is filtered by (I − hγJ)⁻¹ through the same sweep. The inputs `W_in`, `X_in`, `T_s`, `R_f` are sampled on a grid (every 5 s for `reference`, every 1 s inside a filter step) and interpolated by a natural cubic spline: with linear tables the forcing has a kink at every knot, the method loses its order there and the controller rejected a third of its steps; the spline removed every rejection.

One hour of a 10 m tube with a fouling ramp, a steam temperature either smooth (a 2 K sine) or a *record* with 10 s knots and 0.05 K of noise interpolated linearly (as plant data are), one thread, Apple M4, best of three.

1. **Per solve:** `reference(t_max=3600, nz, nt=61)` at 11, 51 and 201 cells; SciPy's BDF with a dense Jacobian (the path before I16), BDF with the banded pattern declared (what `reference` now does above 64 cells), and the compiled integrator; relative tolerances 1e-8 and 1e-6. The error column is against BDF at 1e-11.
2. **Batch fit:** `least_squares` over four Legendre modes of the fouling and a steam offset (prior 0.5 K), from an hour of outlet records every 10 s at 51 cells, each residual evaluation one solve, through either path.
3. **Monitor:** `monitor_fouling` (trend filter at 51 cells, the steam offset as a state) over 360 records, with `integrator="scipy"` and `"native"`.

```bash
python experiments/I16_native_evaporator/run.py     # ~4 min
```

## Results

**Per solve, ms (relative error of X against BDF at 1e-11):**

| Inputs | nz | BDF dense 1e-8 | BDF banded 1e-8 | native 1e-8 | BDF dense 1e-6 | native 1e-6 |
|---|---|---|---|---|---|---|
| smooth | 11 | 10.4 (8e-9) | | **1.8** (4e-9) | 5.2 (1e-6) | **1.3** (3e-7) |
| smooth | 51 | 15.6 (1e-8) | | **5.0** (7e-9) | 10.2 (1e-6) | **2.3** (4e-7) |
| smooth | 201 | 77.8 (1e-8) | 22.1 (1e-8) | **16.2** (6e-9) | 48.2 (6e-6) | **5.7** (4e-7) |
| record | 11 | 716 (9e-8) | | **6.7** (4e-5) | 196 (5e-6) | **3.5** (4e-5) |
| record | 51 | 1,502 (2e-7) | | **30** (5e-5) | 355 (9e-6) | **10** (5e-5) |
| record | 201 | 6,518 (3e-7) | 1,652 (3e-7) | **108** (6e-5) | 1,464 (2e-5) | **39** (6e-5) |

On the record the native path's 4–6e-5 against BDF is not integration error (it is the same at both tolerances) but the spline through the noisy record against SciPy's linear interpolation of it: a modelling difference of 2 × 10⁻⁵ in the outlet solids, a hundredth of the record noise the fits and filters work at.

**Batch fit** (4 modes + offset, 51 cells, 720 records): BDF 106 s in 66 solves, native **0.68 s** in 49 solves; the estimates agree to the third decimal (modes 0.62, 0.53, −0.13, 0.07 and an offset of 0.03 K in both; the truth 0.8, 0.6, −0.1, 0.05 and 0.4 K: the fit trades fouling against the steam offset as J04 says it must, equally through either path).

**Monitor** (51 cells, 360 records, the steam offset as a state): SciPy 1.87 s, native **1.20 s**; the two fouling tracks differ by 6 × 10⁻⁹ m²K/W at most (the estimate is 1.92 × 10⁻⁴ for a truth of 2 × 10⁻⁴), the misfits are identical.

## Conclusion

1. **The right-hand side was never the cost.** At 7 µs a call it is 3 ms of a 78 ms dense BDF solve; the rest was LU of a 402 × 402 matrix 48 times. Declaring the band (`jac_sparsity`) alone gives 3.5× at 201 cells and nothing below 64, where SciPy's sparse path costs more than it saves; `reference` now declares it above 64 cells.
2. **A plant record is the hard case, and the one that matters.** Linearly interpolated data have a kink every 10 s; a variable-order BDF restarts at each and takes seconds per solve, which is where J06's 4–14 s fits came from. The compiled integrator's step is microseconds, its spline keeps the order, and the same fit takes under a second: the batch estimator through the reference solver is now as cheap as the filter was.
3. **The filter is next limited by its own linear algebra.** With the propagation at a tenth of its cost, `monitor_fouling` spends its 1.2 s on the Jacobian of the dynamics by finite differences (103 right-hand-side evaluations a record) and the matrix exponential of a 103 × 103 transition matrix; an analytic Jacobian from the same sweep would take those too.
4. **What the integrator does not do:** the dryer and the droplet (their reference solves are already milliseconds), other interpolations than the spline, or an event (a cleaning); the SciPy path stays the default of `reference` and the fallback where the C++ does not build.

## What changed in the package

- `pinnrl/native/evaporator.cpp`, `pinnrl.native.evaporator` (`evaporator_solve`, `evaporator_available`), and the loader of `pinnrl.native` generalised to several libraries.
- `FallingFilmEvaporator.reference(method="native", rtol, atol)`, `native_propagator(nz, steam_bias, trend)` for the filters' state layouts, the banded Jacobian pattern for SciPy's BDF above 64 cells.
- `ExtendedKalmanFilter(..., propagate=...)` and `MovingHorizonEstimator(..., propagate=...)`; `monitor_fouling(..., integrator="auto" | "native" | "scipy")`, `auto` taking the compiled integrator where it builds.
- `tests/unit_tests/test_native_evaporator.py`: the integrator against a tight BDF solve, the propagator against `solve_ivp` for both state layouts, the monitor and the filters agreeing between the integrators.
