# Process twins

`pinnrl.process` holds what the twin of a process unit is built from: reference simulators of the units, the same models as [`CustomPDE`](custom_equations.md) problems for the PINN, the classical estimators every PINN result is compared with, and an adapter for plant-historian data. This page shows how the pieces fit; the numbers behind the recommendations are in [Benchmarks](benchmarks.md) (experiments J01–J07).

## The units

| Unit | Class | Unknown a plant cannot measure | Records |
|---|---|---|---|
| Falling-film evaporator tube | [`FallingFilmEvaporator`](api/process.md#process.evaporator.FallingFilmEvaporator) | the fouling resistance $R_f(t)$ | outlet solids and flow |
| Drying droplet | [`DryingDroplet`](api/process.md#process.droplet.DryingDroplet) | the drying-rate closure $f(X)$ | mass and temperature |
| Spray-dryer tower (co-current, steady) | [`SprayDryer`](api/process.md#process.dryer.SprayDryer) | the same closure, seen through the air | air temperatures at a few heights, outlet humidity, powder moisture |

Each class has a reference solver (`reference`, method of lines or an ODE integrator), which plays the plant in the experiments and the forward model in the classical estimators, and `custom_pde()`, which writes the same equations with their inputs, scales and sensors for the PINN.

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

With the model's structure known and one slowly drifting unknown, a Kalman filter on a coarse method-of-lines model is the estimator to beat: experiment J05 has it at 3.5 % in 0.2 s against the PINN twin's 9–16 % in five minutes. The unknown is appended to the state. A *random walk* (`augment_random_walk`) suits a parameter that wanders; for one that drifts steadily it either lags or needs so much process noise that its band is several times wider than its error. A *local linear trend* (`augment_local_linear_trend`: a level and a rate, the rate a random walk) follows a ramp without lag and gives the rate a prognosis needs.

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

The level and the rate, with their block of the covariance, give the time until the unknown reaches a limit, here the fouling at which the tube is cleaned. The time is a ratio of two uncertain numbers, so [`time_to_threshold`](api/process.md#process.ekf.time_to_threshold) samples it: the interval is asymmetric, and when the rate is not clearly positive part of the draws never arrive (`probability < 1`, an infinite upper end).

```python
from pinnrl.process import time_to_threshold

idx = [n, n + 1]
tt = time_to_threshold(res.filtered[-1, n], res.filtered[-1, n + 1],
                       res.filtered_cov[-1][np.ix_(idx, idx)], threshold=2e-4)
print(f"cleaning in {tt.median / 60:.0f} min, 90 % interval "
      f"[{tt.lower / 60:.0f}, {tt.upper / 60:.0f}] min (truth 10 min)")
```

Three conditions decide whether such an interval can be trusted; experiment [J06](benchmarks.md#calibrated-fouling-intervals-experiment-j06) measures each. The drift model has to fit the drift (a random walk's band is honest only where it is wide; a trend's only while the rate changes slowly). The model's own error has to be below the interval's width (a coarse grid biases the estimate by more than a tight interval allows). And every input the unknown can be traded against has to be either measured to that accuracy or estimated alongside: for the evaporator that is the steam temperature ([J04](benchmarks.md#fouling-from-outlet-sensors-identifiability-experiment-j04)), which enters the state as one more constant with its prior.

[`monitor_fouling`](api/process.md#process.monitor.monitor_fouling) packages this estimator for the evaporator: the trend filter on a 51-cell model, optionally with the steam record's offset as one more state, returning the fouling with its band at every record, the hindsight estimate, the records against the filter's predictions and the time to clean.

```python
from pinnrl.process import monitor_fouling

mon = monitor_fouling(FallingFilmEvaporator(T_s=steam), times, records,
                      steam_bias_std=0.3, threshold=2e-4)      # the steam record trusted to 0.3 K
print(mon.summary())
```

## Moving-horizon estimation

[`MovingHorizonEstimator`](api/process.md#process.mhe.MovingHorizonEstimator) takes the same `f`, `h`, `Q` and `R` as the filter and re-solves the last `horizon` records at every record, as a bounded least-squares problem in the state at the window's start and one disturbance per record for each state with process noise. For a linear model without bounds it is the Kalman filter, whatever the horizon. Use it for what the filter cannot express: bounds on the states, and bounds on the disturbances, such as a fouling resistance that only grows.

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

It costs 10 to 35 times the filter on the evaporator (about three linearizations of the window per solve; `stride` solves every few records). On the evaporator, whose nonlinearity is mild, it gives the filter's accuracy; the constraint is what it adds ([J05b](benchmarks.md#moving-horizon-estimation-experiment-j05b)).

## Plant data: a historian export and a tag map

A historian export is a table of tag readings in plant units, with gaps and bad readings. [`Historian.from_file`](api/process.md#process.historian.Historian) reads the two usual layouts, a column per tag or a row per reading (with an optional quality flag), from CSV or Parquet (Parquet needs `pyarrow`), with numeric or dated timestamps. A *tag map* then says which tag is which signal of the model and how to convert it.

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

The result feeds both kinds of estimator. The inputs are `TimeSeries` and the sensors carry their records, which is what `custom_pde` takes for the PINN twin; `series.numpy` is the same signal as a plain function for a process model, and `data.records()` is the array a filter runs on.

```python
model = FallingFilmEvaporator(T_s=data.inputs["T_s"].numpy, R_f=0.0)
hist_times, hist_records = data.records(["outlet X", "outlet W"])
pde = model.custom_pde(t_max=data.t_max, sensors=data.sensors,
                       unknown_fouling={"bounds": [0.0, 5e-4], "positive": False})
print(hist_records.shape, len(pde.sensors))
```

[`monitor_from_historian(data)`](api/process.md#process.monitor.monitor_from_historian) runs the fouling monitor on such a window directly, and the [dashboard's Process view](dashboard.md#process) does the same from the two file paths; `write_demo_export(folder)` writes a synthetic export and its tag map to start from.

A `window: [t0, t1]` in the map cuts an interval and shifts it to $t = 0$; `every` and `aggregate` (`interp`, `mean`, `last`) put a tag on a regular grid, which sensors need to share for `records()`. The map can be a YAML file with the same keys.

## Checking a learned closure

When the unknown is a function, the network that produced it is not the judge of it: in experiment J07b a closure 20 % off came with fields that fitted the record to 1 %. Solve the model with the closure by the reference solver and compare with the record. [`DryingDroplet.record_misfit`](api/process.md#process.droplet.DryingDroplet) returns the reduced chi-square, about 1 when the closure explains the record to its noise.

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

The rule from [J07c](benchmarks.md#a-drying-droplet-and-its-closure-experiment-j07): reject a refined closure whose misfit is above its start's, and read the level, since only a misfit near 1 says the closure reproduces the record.

With the network class, give the closure what is known of it: `unknown_rate={"bounds": None, "anchor": {"at": dr.X_eq, "value": 0.0}}` in `custom_pde` makes it zero at the equilibrium moisture by construction, which alone takes a pointwise estimate's misfit from 60–90 to about 2 (J07e). For the network's own fit, `pde.sensor_misfit(trainer.model)` is the same test without a reference solver, and [`train_with_restarts`](api/training.md#training.restarts.train_with_restarts) restarts on it.

## Which measurement to add

Before buying a sensor or a calibration, price it. Around the operating point the posterior covariance of the unknowns is $(P_0^{-1} + J^\top R^{-1} J)^{-1}$ whatever the records turn out to be, so a design is judged from the sensitivities alone. Put every instrument error in the parameter vector with its accuracy as the prior; a calibration is then a tighter prior and a new measurement more rows of $J$.

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

Experiment [J09](benchmarks.md#measurement-design-for-the-fouling-estimate-experiment-j09) does this for the evaporator with six instrument errors: the estimate is limited by the steam offset and, jointly, by the four mass-balance instruments; a condensate meter and a larger, faster steam swing are worth more than any calibration. Check a design that matters by fitting simulated records, as J09 does.

## From the rig to the tower

A droplet rig records the droplet itself; a tower records the air. [`SprayDryer`](api/process.md#process.dryer.SprayDryer) is the droplet coupled to its air, and its `records()` are what a plant has at one operating point.

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

Experiment [J11](benchmarks.md#the-closure-at-dryer-scale-experiment-j11) measures what those records determine: the two parameters of a known curve form from five operating points, about half the directions of a free-form closure, and, either way, the powder moisture at a new operating point to 1–2 %. Identify the closure on the rig and carry it to the tower; at dryer scale estimate a few numbers and the nuisances (J12 prices them: better thermocouples, more operating points and replicate lab assays pay; a droplet-size measurement does not). What the twin then gives that no instrument does is the stickiness margin, the particle temperature above its glass transition: to 1.0–1.5 K at the outlet, against 1.6–3.2 K for a lab assay hours later and tens of kelvin from the air's water balance ([J08](benchmarks.md#a-soft-sensor-for-the-stickiness-margin-experiment-j08)).

## Which estimator

| Situation | Use | Evidence |
|---|---|---|
| Known model, a drifting scalar, online | EKF on a coarse model, trend-augmented | J05, J06 |
| The same with bounds or monotone growth | Moving-horizon estimation | J05b |
| Known model, whole record at once, an interval to defend | Gauss–Newton on the reference solver with a few modes of the unknown | J04, J06 |
| An unknown function of the state in an ODE model | Shooting with a monotone interpolant through the reference solver | J07d |
| A closure seen only through a unit's outlet and profile records | A few parameters of a known form (or a scale factor) by least squares through the reference; report the prediction, not the closure | J11 |
| No reference solver for the unknown's form, or a field in space | The PINN twin, started from a pointwise estimate and checked with a misfit guard | J03, J07b, J07c |
