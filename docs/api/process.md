# Process models

Reference simulators of process units, their `CustomPDE` forms and the classical estimators they are compared with (`pinnrl.process`). The falling-film evaporator is the first unit (roadmap v0.7, experiments J01–J06) and the drying droplet, with its drying-rate closure, the second (J07); the spray-dryer tower couples that droplet to its air (J11). The estimators are an extended Kalman filter with a smoother and moving-horizon estimation; `Historian` turns a plant-historian export into the inputs and sensor records they and the PINN twin take.

::: process.evaporator.FallingFilmEvaporator

::: process.evaporator.EvaporatorReference

::: process.droplet.DryingDroplet

::: process.dryer.SprayDryer

::: process.dryer.DryerReference

::: process.droplet.DropletReference

::: process.droplet.saturation_pressure

::: process.ekf.ExtendedKalmanFilter

::: process.ekf.FilterResult

::: process.ekf.augment_random_walk

::: process.ekf.augment_local_linear_trend

::: process.ekf.time_to_threshold

::: process.ekf.ThresholdTime

::: process.ekf.finite_difference_jacobian

::: process.mhe.MovingHorizonEstimator

::: process.historian.Historian

::: process.historian.TwinData

::: process.monitor.monitor_fouling

::: process.monitor.FoulingMonitor

::: process.monitor.monitor_from_historian

::: process.monitor.monitor_from_files

::: process.monitor.write_demo_export
