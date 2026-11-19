# Process twins

This page shows how to build the twin of a process unit with `pinnrl.process`: the model of each unit, what a plant measures on it, and which estimator recovers what it cannot measure. It is for readers who know machine learning and want the process-engineering side; the full results are in [Benchmarks](benchmarks.md) (experiments J01–J13).

- **Twin**: a model of a running unit, kept in step with its records, that reports quantities no instrument measures.
- **Reference solver**: a classical numerical solver of the unit's model. It plays the plant in the experiments and the forward model in the classical estimators.
- **Classical estimator**: one that needs no neural network (a Kalman filter, a least-squares fit through the reference solver). Every PINN result is compared with one.

`pinnrl.process` holds reference simulators of the units, the same models as [`CustomPDE`](custom_equations.md) problems for the PINN, the classical estimators, and an adapter for plant-historian data.

## The units

| Unit | Class | Unknown a plant cannot measure |
|---|---|---|
| Falling-film evaporator tube | [`FallingFilmEvaporator`](api/process.md#process.evaporator.FallingFilmEvaporator) | the fouling resistance $R_f(t)$ |
| Drying droplet | [`DryingDroplet`](api/process.md#process.droplet.DryingDroplet) | the drying-rate closure $f(X)$ |
| Spray-dryer tower (co-current, steady) | [`SprayDryer`](api/process.md#process.dryer.SprayDryer) | the same closure, seen through the air |
| Counter-current tower (steady and transient) | [`CounterCurrentDryer`](api/process.md#process.counter_dryer.CounterCurrentDryer) | the curve's parameters and the wall loss, from a transient |
| The line: evaporator passes feeding the dryer | [`ProcessLine`](api/process.md#process.line.ProcessLine) | every unit's unknowns, and which one the powder inherits |

Each class has two faces:

- `reference`: the reference solver, an ODE integrator or the **method of lines** (space derivatives replaced by differences on a grid, which leaves ODEs in time).
- `custom_pde()`: the same equations with their inputs, scales and sensors, for the PINN.

### The falling-film evaporator

- **Falling film**: a thin liquid film that runs down the inside of a steam-heated tube and boils as it goes, which concentrates the liquid.
- **Fouling**: a deposit that grows on the heated wall and adds a resistance $R_f$ to the heat flow. The tube is cleaned when $R_f$ reaches a limit.

$$
\begin{aligned}
W_t + v\,W_z &= -\frac{v\,P\,q}{\lambda}, \\
X_t + v\,X_z &= \frac{v\,X\,P\,q}{\lambda\,W}, \\
q &= \frac{T_s(t) - T_b(X)}{1/U + R_f(t)}, \qquad T_b(X) = T_{sat} + k_{bpe}\,X .
\end{aligned}
$$

- $W(z, t)$: liquid mass flow, kg/s. $X(z, t)$: solids mass fraction.
- $z$: distance down the tube of length $L$ (10 m). $v$: film velocity (0.5 m/s, 20 s of residence).
- $q$: wall heat flux, W/m². $P$: wetted perimeter. $\lambda$: latent heat of evaporation.
- $T_s(t)$: steam temperature. $T_b$: boiling point of the film. $T_{sat}$: the solvent's saturation temperature. $k_{bpe}$: boiling-point elevation, the rise of the boiling point per unit of solids.
- $U$: clean heat-transfer coefficient (2,000 W/m²K). $R_f(t)$: fouling resistance, m²K/W.
- Inlet: $W(0, t) = W_{in}(t)$, $X(0, t) = X_{in}(t)$, the feed.

Measured:

- the inputs $T_s(t)$, $W_{in}(t)$, $X_{in}(t)$;
- the outlet $X(L, t)$ and $W(L, t)$.

Unknown:

- $R_f(t)$. Only the sum $1/U + R_f$ enters, so a monitor estimates the change of resistance since a reference state ([J04](benchmarks.md#fouling-from-outlet-sensors-identifiability-experiment-j04)).

Headline (J05): an extended Kalman filter on an 11-cell model recovers $R_f(t)$ to 3.5 % in L2 in 0.2 s; the PINN twin reaches 9–16 % in five minutes and fails to train on 22 of 36 runs (J03, J06). Every estimator is in [Benchmarks](benchmarks.md#calibrated-fouling-intervals-experiment-j06).

Finding: with the model's structure known, the classical estimators reach the data limit at a thousandth of the PINN twin's cost. The steam temperature is the input that limits every estimator (J04).

The other units have their model where the page uses them: the [droplet](#checking-a-learned-closure), the [co-current tower](#from-the-rig-to-the-tower), the [counter-current tower](#the-counter-current-tower) and [the line](#the-line).

The plant used in the rest of the page: a fouling ramp, a recorded steam signal, noisy outlet records.

```python
import numpy as np
from pinnrl.process import FallingFilmEvaporator

T = 600.0
steam = lambda t: 355.15 + np.sin(2 * np.pi * t / 400.0)   # recorded, K
plant = FallingFilmEvaporator(T_s=steam, R_f=lambda t: 1e-4 * t / T)
ref = plant.reference(t_max=T, nz=51, nt=121)               # one output every 5 s
times = ref.t[1:]
rng = np.random.default_rng(0)
records = np.stack(                                         # outlet solids and flow, with noise
    [ref.X.u[-1, 1:] + 0.002 * rng.standard_normal(times.size),
     ref.W.u[-1, 1:] + 0.001 * rng.standard_normal(times.size)], axis=1)
```

## Estimating a drifting parameter

$$
\frac{ds}{dt} = f(t, s, \theta), \qquad y_k = h(t_k, s) + v_k, \qquad v_k \sim \mathcal{N}(0, R) .
$$

- $s$: the model's state (here $W$ and $X$ on a coarse grid). $\theta$: the drifting unknown (here $R_f$).
- $y_k$: the record at time $t_k$. $R$: its noise covariance.
- **Extended Kalman filter (EKF)**: at each record, corrects the model's prediction by the record, each weighted by its uncertainty; the model is linearized at the current estimate.
- **Smoother**: a backward pass that estimates each time from the whole record (the hindsight estimate).
- **Process noise** $Q$: how much each state may wander between records.

With the structure known and one slowly drifting unknown, this filter on a coarse method-of-lines model is the estimator to beat (J05). The unknown joins the state in one of two ways:

| Drift model | Equations | Suits | Limit |
|---|---|---|---|
| Random walk (`augment_random_walk`) | $\dot\theta = w$ | a parameter that wanders | on a steady drift it lags, or needs so much process noise that its band is several times wider than its error |
| Local linear trend (`augment_local_linear_trend`) | $\dot\theta = \rho,\ \dot\rho = w$ | a parameter that drifts steadily | honest only while the rate changes slowly |

- $w$: white noise of intensity $Q$. $\rho$: the drift's rate.
- The trend follows a ramp without lag and gives the rate a prognosis needs.

```python
from pinnrl.process import ExtendedKalmanFilter, augment_local_linear_trend

nz = 11                                                     # a coarse model is enough for the filter
model = FallingFilmEvaporator(T_s=steam, R_f=0.0)
f, h, s0 = model.fouling_state_space(nz)                    # f(t, s, [R_f]), h(t, s), steady start
n = s0.size
ekf = ExtendedKalmanFilter(
    augment_local_linear_trend(f, n, 1),                    # state = [s, R_f, dR_f/dt]
    lambda t, x: h(t, x[:n]),
    Q=np.concatenate([np.zeros(n + 1), [(1e-9) ** 2]]),     # only the rate wanders
    R=np.array([0.002**2, 0.001**2]),
)
x0 = np.concatenate([s0, [0.5e-4, 0.0]])
P0 = np.diag(np.concatenate([np.full(n, 1e-6), [(1e-4) ** 2, (1e-6) ** 2]]))
res = ekf.run(times, records, x0, P0, smooth=True)
lo, hi = res.interval(0.9)                                  # (K, n + 2) each
print(f"R_f(T) = {res.filtered[-1, n]:.2e} in [{lo[-1, n]:.2e}, {hi[-1, n]:.2e}]  (truth 1.00e-04)")
```

The level and the rate give the time until the unknown reaches a limit $\theta^*$, here the fouling at which the tube is cleaned:

$$
\tau = \frac{\theta^* - \theta}{\rho} .
$$

- [`time_to_threshold`](api/process.md#process.ekf.time_to_threshold) samples $\tau$ from the level, the rate and their block of the covariance. The interval is asymmetric.
- When the rate is not clearly positive, part of the draws never arrive: `probability < 1` and an infinite upper end.

```python
from pinnrl.process import time_to_threshold

idx = [n, n + 1]
tt = time_to_threshold(res.filtered[-1, n], res.filtered[-1, n + 1],
                       res.filtered_cov[-1][np.ix_(idx, idx)], threshold=2e-4)
print(f"cleaning in {tt.median / 60:.0f} min, 90 % interval "
      f"[{tt.lower / 60:.0f}, {tt.upper / 60:.0f}] min (truth 10 min)")
```

Such an interval can be trusted under three conditions ([J06](benchmarks.md#calibrated-fouling-intervals-experiment-j06) measures each):

1. **The drift model fits the drift.** A random walk's band is honest only where it is wide; a trend's only while the rate changes slowly.
2. **The model's own error is below the interval's width.** A coarse grid biases the estimate by more than a tight interval allows.
3. **Every input the unknown can be traded against is measured to that accuracy or estimated alongside.** For the evaporator that is the steam temperature ([J04](benchmarks.md#fouling-from-outlet-sensors-identifiability-experiment-j04)), which enters the state as one more constant with its prior.

[`monitor_fouling`](api/process.md#process.monitor.monitor_fouling) packages this for the evaporator: the trend filter on a 51-cell model, optionally with the steam record's offset as one more state. It returns the fouling with its band, the hindsight estimate, the records against the filter's predictions, and the time to clean.

```python
from pinnrl.process import monitor_fouling

mon = monitor_fouling(FallingFilmEvaporator(T_s=steam), times, records,
                      steam_bias_std=0.3, threshold=2e-4)      # the steam record trusted to 0.3 K
print(mon.summary())
```

The estimators spend their time in the reference solve. Experiment [I16](benchmarks.md#the-evaporators-reference-solve-in-c-experiment-i16) put the evaporator's method-of-lines model in C++ ([`pinnrl.native.evaporator`](api/native.md)), with an L-stable fourth-order SDIRK integrator.

- `reference(..., method="native")` uses it, with the inputs sampled every 5 s and interpolated by a spline.
- `FallingFilmEvaporator.native_propagator(nz, steam_bias, trend)` gives the filters a `propagate(t0, t1, x)` (`ExtendedKalmanFilter(..., propagate=...)`).
- `monitor_fouling(..., integrator="auto")` takes it wherever it builds.
- SciPy's BDF remains the default of `reference` and the fallback everywhere.

## Moving-horizon estimation

$$
\min_{x_0,\ w_1..w_N}\ \lVert x_0 - \bar x_0 \rVert^2_{P_0^{-1}} + \sum_{k=1}^{N} \lVert w_k \rVert^2_{Q^{-1}} + \sum_{k=1}^{N} \lVert y_k - h(t_k, x_k) \rVert^2_{R^{-1}}
\quad \text{with} \quad x^{lo} \le x_0 \le x^{hi},\quad w^{lo} \le w_k \le w^{hi} .
$$

- **Moving-horizon estimation (MHE)**: at every record, re-solve this least-squares problem over the last $N$ records (`horizon`).
- $x_0$: the state at the window's start. $\bar x_0$, $P_0$: what the earlier records say about it (the arrival cost).
- $w_k$: one disturbance per record for each state with process noise. $x_k$ follows the model from $x_0$ and the $w_k$.
- Bounds: `lower` and `upper` on $x_0$, `noise_lower` and `noise_upper` on the $w_k$.

For a linear model without bounds, [`MovingHorizonEstimator`](api/process.md#process.mhe.MovingHorizonEstimator) is the Kalman filter, whatever the horizon. Use it for what the filter cannot express: bounds on the states and on the disturbances, such as a fouling resistance that only grows.

```python
from pinnrl.process import MovingHorizonEstimator, augment_random_walk

Q = np.concatenate([np.zeros(n), [(1e-6) ** 2]])            # R_f a random walk
mhe = MovingHorizonEstimator(augment_random_walk(f, n, 1), lambda t, x: h(t, x[:n]),
                             Q=Q, R=np.array([0.002**2, 0.001**2]), horizon=12)
lower = np.r_[np.full(n, -np.inf), 0.0]                     # R_f >= 0
grow = mhe.run(times, records, np.concatenate([s0, [0.5e-4]]),
               np.diag(np.concatenate([np.full(n, 1e-6), [(1e-4) ** 2]])),
               lower=lower, noise_lower=0.0, stride=12)     # disturbances >= 0, one solve a minute
print(grow.t[-1], grow.filtered[-1, -1])
```

Results on the evaporator ([J05b](benchmarks.md#moving-horizon-estimation-experiment-j05b)):

- **Cost**: 10 to 35 times the filter, about three linearizations of the window per solve. `stride` solves every few records.
- **Accuracy**: the filter's, because the evaporator's nonlinearity is mild.
- **What it adds**: the constraint.

## Plant data: a historian export and a tag map

- **Historian**: the plant's database of instrument readings. Its export is a table of readings in plant units, with gaps and bad readings.
- **Tag**: the name of one instrument signal, such as `TT205.PV`.
- **Tag map**: which tag is which signal of the model, and how to convert it.

[`Historian.from_file`](api/process.md#process.historian.Historian) reads:

- a column per tag, or a row per reading (with an optional quality flag);
- CSV or Parquet (Parquet needs `pyarrow`);
- numeric or dated timestamps.

```python
import tempfile
from pathlib import Path
from pinnrl.process import Historian

rows = ["timestamp,TT205.PV,DT310.PV,FT320.PV"]            # degC, % solids, kg/h
for k, t in enumerate(times):
    rows.append(f"{t:.0f},{steam(t) - 273.15:.3f},{100 * records[k, 0]:.4f},{3600 * records[k, 1]:.3f}")
export = Path(tempfile.mkdtemp()) / "evaporator.csv"
export.write_text("\n".join(rows) + "\n")

tag_map = {
    "inputs": {"T_s": {"tag": "TT205.PV", "offset": 273.15}},                 # degC -> K
    "sensors": [
        {"tag": "DT310.PV", "name": "outlet X", "location": 10.0, "expression": "X",
         "gain": 0.01, "noise": 0.002},                                        # % -> fraction
        {"tag": "FT320.PV", "name": "outlet W", "location": 10.0, "expression": "W",
         "gain": 1 / 3600, "noise": 0.001},                                    # kg/h -> kg/s
    ],
}
data = Historian.from_file(export).apply(tag_map)
print(data.t_max, data.inputs["T_s"], data.sensors)
```

The result feeds both kinds of estimator:

- **The PINN twin**: the inputs are `TimeSeries` and the sensors carry their records, as `custom_pde` takes them.
- **A process model**: `series.numpy` is the same signal as a plain function.
- **A filter**: `data.records()` is the array it runs on.

```python
model = FallingFilmEvaporator(T_s=data.inputs["T_s"].numpy, R_f=0.0)
hist_times, hist_records = data.records(["outlet X", "outlet W"])
pde = model.custom_pde(t_max=data.t_max, sensors=data.sensors,
                       unknown_fouling={"bounds": [0.0, 5e-4], "positive": False})
print(hist_records.shape, len(pde.sensors))
```

Shortcuts and options:

- [`monitor_from_historian(data)`](api/process.md#process.monitor.monitor_from_historian) runs the fouling monitor on such a window.
- The [dashboard's Process view](dashboard.md#process) does the same from the two file paths.
- `write_demo_export(folder)` writes a synthetic export and its tag map to start from.
- `window: [t0, t1]` in the map cuts an interval and shifts it to $t = 0$.
- `every` and `aggregate` (`interp`, `mean`, `last`) put a tag on a regular grid, which sensors need to share for `records()`.
- The map can be a YAML file with the same keys.

## Checking a learned closure

The unit is the drying droplet: one droplet of a solids solution in hot air, as on a single-droplet rig.

- **Closure**: a relation the balance equations need but do not supply, here the drying rate as a function of the moisture.
- **Dry-basis moisture** $X$: kg of water per kg of solids.
- **Characteristic drying curve** $f(X)$: the drying rate divided by that of a fully wet surface. It is 1 above the critical moisture $X_{cr}$ and 0 at and below the equilibrium moisture $X_{eq}$.

$$
\begin{aligned}
\frac{dX}{dt} &= -r, \qquad r = \frac{h_m A\,\big(\rho_{vs}(T) - Y \rho_a\big)}{m_s}\, f(X), \\
\frac{dT}{dt} &= \frac{h A\,(T_a - T) - \lambda\, m_s\, r}{m_s\,(c_s + c_w X)}, \\
f(X) &= \min\!\left(1,\ \left(\frac{X - X_{eq}}{X_{cr} - X_{eq}}\right)^{n}\right) .
\end{aligned}
$$

- $T$, $T_a$: droplet and air temperature. $Y$: humidity ratio of the air, kg of water per kg of dry air. $\rho_a$: air density.
- $r$: drying rate per unit of solids. $m_s$: solids mass of the droplet. $A$: its surface.
- $h$, $h_m$: heat- and mass-transfer coefficients (Ranz–Marshall correlations of the droplet's diameter).
- $\rho_{vs}(T)$: vapour density at saturation. $c_s$, $c_w$: heat capacities of solids and water.
- $n$: exponent of the falling-rate part of the curve. Defaults: $X_{eq} = 0.05$, $X_{cr} = 1.0$, $n = 1.5$.

Measured:

- the droplet's mass $m = m_s (1 + X)$;
- its temperature $T$.

Unknown:

- the closure $f(X)$.

Headline (J07d): shooting with a monotone piecewise-linear closure on 24 knots recovers $f(X)$ to 1.5–1.6 % on every seed in 8–10 s; a PINN with $f$ an unknown function, from a cold start, collapses to a constant on every seed (J07). Every estimator is in [Benchmarks](benchmarks.md#a-drying-droplet-and-its-closure-experiment-j07).

- **Shooting**: adjust the unknowns until the reference solver's output matches the record.

Finding: with an ODE model and a reference solver that takes a closure, an unknown function is a few dozen bounded parameters, and shooting beats the PINN.

The network that produced a closure is not the judge of it: in J07b a closure 20 % off came with fields that fitted the record to 1 %. The check is independent of the network:

1. Solve the model with the closure by the reference solver.
2. Compare with the record by the reduced chi-square.

$$
\chi^2 = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{y_i - \hat y_i}{\sigma_i} \right)^2 .
$$

- $y_i$: a reading. $\hat y_i$: the model's value. $\sigma_i$: the reading's noise.
- **Reduced chi-square**: about 1 when the closure explains the record to its noise.
- [`DryingDroplet.record_misfit`](api/process.md#process.droplet.DryingDroplet) returns it for the mass, the temperature and both.

```python
from pinnrl.process import DryingDroplet

dr = DryingDroplet()
run = dr.reference(200.0, nt=2001)
t_rec = np.arange(1.0, 201.0)
mass = np.interp(t_rec, run.t, run.m) + 0.005 * dr.m0 * rng.standard_normal(t_rec.size)
temp = np.interp(t_rec, run.t, run.T) + 0.5 * rng.standard_normal(t_rec.size)
candidate = lambda X: np.clip((X - dr.X_eq) / (1.3 - dr.X_eq), 0.0, 1.0) ** 1.5   # a critical moisture 30 % off
print(dr.record_misfit(t_rec, mass, temp, 0.005 * dr.m0, 0.5)["total"],              # about 1
      dr.record_misfit(t_rec, mass, temp, 0.005 * dr.m0, 0.5, closure=candidate)["total"])
```

The rule from [J07c](benchmarks.md#a-drying-droplet-and-its-closure-experiment-j07):

1. Reject a refined closure whose misfit is above its start's.
2. Read the level: only a misfit near 1 says the closure reproduces the record.

With the network class, give the closure what is known of it:

- **Anchor**: `unknown_rate={"bounds": None, "anchor": {"at": dr.X_eq, "value": 0.0}}` in `custom_pde` makes the closure zero at the equilibrium moisture by construction. That alone takes a pointwise estimate's misfit from 60–90 to about 2 (J07e).
- **The network's own fit**: `pde.sensor_misfit(trainer.model)` is the same test without a reference solver.
- **Restarts**: [`train_with_restarts`](api/training.md#training.restarts.train_with_restarts) restarts on that misfit.

## Which measurement to add

$$
\Sigma = \left( P_0^{-1} + J^\top R^{-1} J \right)^{-1} .
$$

- $\Sigma$: posterior covariance of the unknowns around the operating point.
- $P_0$: prior covariance. Each instrument error is an unknown with its accuracy as the prior.
- $J$: sensitivities of the records to the unknowns. $R$: measurement-noise covariance.

$\Sigma$ does not depend on what the records turn out to be, so a sensor or a calibration is priced from the sensitivities alone, before buying it.

- A calibration is a tighter prior.
- A new measurement is more rows of $J$.

```python
from pinnrl.analysis import greedy_design, information, posterior_covariance, target_std

def outlet(theta):                                  # records for theta = [R_f mean (1e-4), R_f slope, steam offset]
    ev = FallingFilmEvaporator(T_s=lambda t: steam(t) - theta[2],
                               R_f=lambda t: 1e-4 * (theta[0] + theta[1] * (2 * t / T - 1)))
    r = ev.reference(t_max=T, nz=51, nt=121)
    return r.X.u[-1, 1:], r.W.u[-1, 1:]

nominal, steps = np.array([0.5, 0.5, 0.0]), np.array([0.02, 0.02, 0.02])
cols = []
for j in range(3):                                  # central differences through the reference solver
    up, dn = nominal.copy(), nominal.copy()
    up[j] += steps[j]
    dn[j] -= steps[j]
    cols.append([(a - b) / (2 * steps[j]) for a, b in zip(outlet(up), outlet(dn))])
JX, JW = (np.stack([c[i] for c in cols], axis=1) for i in range(2))
F = information(JX, 0.002) + information(JW, 0.001)
end = np.array([1e-4, 1e-4, 0.0])                   # R_f(T) = 1e-4 (mean + slope)
for label, steam_std in (("steam trusted to 0.3 K", 0.3), ("calibrated to 0.03 K", 0.03)):
    cov = posterior_covariance(F, [np.inf, np.inf, steam_std])
    print(f"{label}: R_f(T) to {target_std(cov, end) / 1e-4:.1%}")
lab = greedy_design(list(JX[11::12]), 0.0005, 2, prior_std=[np.inf, np.inf, 0.3],
                    base_information=F, target=end)  # two lab assays of the outlet solids: when?
print([float(times[11::12][i]) for i in lab.chosen], [f"{c / 1e-4:.1%}" for c in lab.criterion])
```

Experiment [J09](benchmarks.md#measurement-design-for-the-fouling-estimate-experiment-j09) does this for the evaporator with six instrument errors:

- The estimate is limited by the steam offset and, jointly, by the four mass-balance instruments.
- A condensate meter and a larger, faster steam swing are worth more than any calibration.
- Check a design that matters by fitting simulated records, as J09 does.

## From the rig to the tower

A droplet rig records the droplet itself; a tower records the air. [`SprayDryer`](api/process.md#process.dryer.SprayDryer) is the droplet coupled to its air: a plug flow of equal droplets that travel down the tower with the air (co-current), at steady state.

$$
\begin{aligned}
\frac{dX}{d\tau} &= -r, \\
\frac{dT_p}{d\tau} &= \frac{h A\,(T_a - T_p) - \lambda\, m_s\, r}{m_s\,(c_s + c_w X)}, \\
\frac{dY}{d\tau} &= \frac{F_s}{G}\, r, \\
\frac{dT_a}{d\tau} &= -\frac{\dfrac{F_s}{m_s}\, h A\,(T_a - T_p) + G\, c_v\,(T_a - T_p)\,\dfrac{dY}{d\tau} + U_w\, \pi D\, v_p\,(T_a - T_{amb})}{G\,(c_a + c_v Y)} .
\end{aligned}
$$

- $\tau = z / v_p$: residence time of the droplets. $z$: distance below the atomiser. $v_p$: droplet velocity.
- $X$, $T_p$: moisture and temperature of the droplets. $Y$, $T_a$: humidity ratio and temperature of the air.
- $r$: the droplet's drying rate, with the same closure $f(X)$ and the local $Y$.
- $F_s$: solids flow, kg/s. $G$: dry-air flow, kg/s. $c_a$, $c_v$: heat capacities of dry air and vapour.
- $U_w$: wall loss coefficient. $D$: tower diameter. $T_{amb}$: ambient temperature.
- Start: the droplets' state at the atomiser and the inlet air ($T_{in}$, $Y_{in}$).

Measured (`records()`, one operating point):

- air temperatures at six heights;
- the outlet air's humidity;
- the powder's moisture, from the lab.

Unknown:

- the closure $f(X)$, or the two parameters $X_{cr}$ and $n$ of a known form;
- the nuisances: the wall loss and the droplet size.

```python
from pinnrl.process import SprayDryer

tower = SprayDryer()                                         # 8 m, air at 180 degC, 250 um droplets
profile = tower.reference()                                  # X, T_p, Y, T_a along the height; 5 ms
print(profile.outlet(), tower.balances(profile)["energy"])
slow = tower.with_closure(lambda X: 0.7 * tower.droplet.drying_curve(X))
print(tower.records()["T_air"] - 273.15, slow.records()["T_air"] - 273.15)
print(tower.operating_point(T_in=438.15, solids_flow=0.012).records()["X_out"])
print(tower.stickiness_margin(profile)[-1])                  # T_p - T_g(X) at the outlet, K
```

What the twin gives that no instrument does is the stickiness margin:

$$
\Delta T = T_p - T_g(X), \qquad T_g = \frac{w_s\, T_{gs} + k\, w_w\, T_{gw}}{w_s + k\, w_w}, \qquad w_w = \frac{X}{1 + X},\quad w_s = 1 - w_w .
$$

- **Glass transition** $T_g$: the temperature above which an amorphous solid softens. Water lowers it (the Gordon–Taylor relation above).
- **Stickiness margin** $\Delta T$: how far a particle is above its glass transition. Particles stick to the wall and cake when it passes a material limit.
- $T_{gs}$, $T_{gw}$: glass transitions of the dry solids and of water. $k$: the Gordon–Taylor constant of the pair.

Headline ([J08](benchmarks.md#a-soft-sensor-for-the-stickiness-margin-experiment-j08)): the twin, run forward with the identified closure, gives the outlet margin to 1.0–1.5 K, against 1.6–3.2 K for a lab assay hours later. With the curve's form known, five operating points identify its two parameters and predict the powder moisture at a new operating point to 1–2 % ([J11](benchmarks.md#the-closure-at-dryer-scale-experiment-j11)).

Finding: identify the closure on the rig and carry it to the tower; at dryer scale estimate a few numbers and the nuisances.

J12 prices the measurements at dryer scale:

- **Pay**: better thermocouples, more operating points, replicate lab assays.
- **Does not pay**: a droplet-size measurement.

## The counter-current tower

The air enters at the bottom against the falling droplets. The droplets' state is known at the top and the air's at the bottom.

$$
\begin{aligned}
X_t + v_p\, X_z &= -r, \\
T_t + v_p\, T_z &= \frac{h A\,(T_a - T) - \lambda\, m_s\, r}{m_s\,(c_s + c_w X)}, \\
Y_t - v_a\, Y_z &= \frac{v_a}{G}\, S, \\
T_{a,t} - v_a\, T_{a,z} &= -\frac{v_a}{G}\, \frac{Q + c_v\,(T_a - T)\, S + U_w\, \pi D\,(T_a - T_{amb})}{c_a + c_v Y} .
\end{aligned}
$$

- $z$: distance down from the atomiser. $T$: droplet temperature. The other fields are those of the co-current tower.
- $v_p$: droplet velocity, downward. $v_a = G / (\rho_a A_{tower})$: air velocity, upward.
- $S = (F_s / v_p)\, r$: water the droplets give to the air per metre of height.
- $Q = \dfrac{F_s}{m_s v_p}\, h A\,(T_a - T)$: heat they take from it per metre.
- Boundaries: $X$, $T$ at the top ($z = 0$); $Y$, $T_a$ at the bottom, the inlet air $T_{in}(t)$, $Y_{in}(t)$.
- Time scales: the droplets cross the tower in 5.3 s, the air in 106 s.

Measured (`records`, over a transient):

- air temperatures at five heights;
- the humidity of the air leaving at the top;
- the moisture of the powder leaving at the bottom.

Unknown:

- the curve's parameters $X_{cr}$ and $n$;
- the wall loss $U_w$.

Solvers:

- [`CounterCurrentDryer.steady`](api/process.md#process.counter_dryer.CounterCurrentDryer) solves the steady state, a boundary-value problem, by shooting on the air's state at the top: about 100 ms, against 5 ms for the co-current tower.
- `transient` integrates the transport equations by the method of lines with a sparse Jacobian. Ten minutes take 0.04–0.13 s for a smooth step and 0.34–1.3 s for an inlet record with 10 s knots, on 51 to 401 cells (J13).
- `custom_pde` writes the same four fields in $(z, t)$ for the PINN: the inlet air as time-series inputs, the steady profiles as the initial condition, each equation divided by its own scale. `identify=` names the parameters to estimate.

```python
import numpy as np
from pinnrl.process import CounterCurrentDryer

tower = CounterCurrentDryer(T_in=lambda t: 433.15 + (15.0 if t > 10.0 else 0.0))   # a step of the inlet air
profile = tower.steady()                                   # X, T_p, Y, T_a down the tower; the BVP by shooting
print(profile.outlet(), tower.balances(profile)["energy"], tower.stickiness_margin(profile)[-1])
response = tower.transient(600.0, nz=201)                  # the step response by the method of lines
records = tower.records(response, probe_heights=(0.5, 2.0, 3.5, 5.0, 6.5))
print(records["T_air"][:, -1] - 273.15, records["X_out"][[0, -1]])
pde = tower.custom_pde(600.0, identify=["Xcr", "ncdc"], initial_guesses={"Xcr": 0.7, "ncdc": 1.0})
```

Headline ([J13](benchmarks.md#the-counter-current-tower-experiment-j13)): bounded least squares through one transient record takes 10 s and recovers $X_{cr}$, $n$, $U_w$ within the linear prediction (±0.011, ±0.015, ±0.003) on all three data sets; the PINN twin on the same record takes 110–140 s and all six runs are rejected by the misfit test (reduced chi-square 5–9).

Finding: the steady state has no cheap initial-value solve, the case the roadmap set for the PINN twin, but the transient does, and the classical fit through it wins.

## The line

[`ProcessLine`](api/process.md#process.line.ProcessLine) chains evaporator passes and hands the concentrate to the dryer.

$$
\begin{aligned}
W^{(k+1)}_{in}(t) &= W^{(k)}(L, t), \qquad X^{(k+1)}_{in}(t) = X^{(k)}(L, t), \\
F_s(t) &= W_c(t)\, X_c(t), \qquad \text{droplet solids fraction} = X_c(t), \\
\sigma_y^2 &= \sum_j \left( \frac{\partial y}{\partial \theta_j}\, \sigma_j \right)^2, \qquad \text{share}_j = \frac{1}{\sigma_y^2} \left( \frac{\partial y}{\partial \theta_j}\, \sigma_j \right)^2 .
\end{aligned}
$$

- $W^{(k)}$, $X^{(k)}$: flow and solids of pass $k$, each pass an evaporator tube as above.
- $W_c$, $X_c$: the concentrate, the outlet of the last pass. $F_s$: the dryer's solids flow.
- $y$: a target (the concentrate's solids, the powder's moisture, its stickiness margin).
- $\theta_j$, $\sigma_j$: an uncertain quantity of a unit and its standard deviation.
- The dryer is quasi-steady: its residence time is seconds against the evaporator's hours of fouling.

Methods:

- `steady`: the line at an instant.
- `reference`: the shift. The passes by the method of lines, the dryer at every output time.
- `records`: what the line's instruments report.
- `propagate`: linear and Monte-Carlo propagation of each unit's uncertainty to the three targets, with each source's share of the variance.

Measured:

- each pass's outlet solids and flow;
- the dryer's air temperatures and outlet humidity;
- lab assays of the powder.

Unknown:

- every unit's unknowns (twelve in J10), and which one the powder inherits.

```python
from pinnrl.process import ProcessLine

line = ProcessLine.default(fouling=[lambda t: 2e-5 * t / 14400, lambda t: 4e-5 * t / 14400, 0.0])
print(line.steady()["X"], line.steady(t=14400.0)["powder"])      # 10 % -> 41 % solids; the powder after four hours
shift = line.reference(4 * 3600.0, nz=51, dryer_every=4)          # three passes, then the dryer every 2 min
budget = line.propagate({"R_f_0": 6e-6, "R_f_1": 5e-6, "X_cr": 0.1, "d0": 25e-6, "W_feed": 0.002}, t=14400.0)
print(budget["linear"]["powder"])                                  # std of the powder moisture and the shares
```

Headline ([J10](benchmarks.md#the-line-level-twin-experiment-j10)): at the end of a shift, the standard deviation of the powder moisture is 0.094 from the fouling monitor on each pass followed by propagation, 0.0054 from the dryer's own online records, and 0.0011 from every record of the line.

Findings:

- **The information flows downstream only.** The dryer's instruments determine its powder but not the fouling behind it.
- **The first pass's fouling is one unknown with the feed's calibration.**

## Which estimator

| Situation | Use | Evidence |
|---|---|---|
| Known model, a drifting scalar, online | EKF on a coarse model, trend-augmented | J05, J06 |
| The same with bounds or monotone growth | Moving-horizon estimation | J05b |
| Known model, whole record at once, an interval to defend | Gauss–Newton on the reference solver with a few modes of the unknown | J04, J06 |
| An unknown function of the state in an ODE model | Shooting with a monotone interpolant through the reference solver | J07d |
| A closure seen only through a unit's outlet and profile records | A few parameters of a known form (or a scale factor) by least squares through the reference; report the prediction, not the closure | J11 |
| A few parameters from a transient of a unit whose steady state is a boundary-value problem | Least squares through the method-of-lines transient (a tenth of a second a solve) | J13 |
| A line of units | Monitor each unit on its own records, then propagate; use the downstream unit's records for its own product | J10 |
| No reference solver for the unknown's form, or a field in space | The PINN twin, started from a pointwise estimate and checked with a misfit guard | J03, J07b, J07c |
