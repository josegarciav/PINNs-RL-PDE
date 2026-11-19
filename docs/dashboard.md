# Dashboard

This page describes the browser app that launches runs, follows them as they train and compares their results, with no plotting code to write. Read it if you prefer a form to the command line, or want the fouling monitor on plant data.

```bash
pinnrl-dashboard                       # http://127.0.0.1:8050/, opens a browser tab
pinnrl-dashboard --port 8051 --no-browser
```

- **Run**: one training, stored as a folder of files.
- **Runs folder**: `./runs/`, relative to the directory you start the dashboard from. It is the same folder `pinnrl-train` uses, and the dashboard lists and writes runs there.
- **Launched runs**: ordinary `pinnrl-train` processes. They keep going if you close the browser.
- **Port**: if the port is taken, the next two are tried.

![Landing page](assets/screenshots/dashboard-landing.png)

## Live Training

**New Training** is a form over every `pinnrl-train` option. Defaults come from `config.yaml` for the selected PDE.

It covers the PDE, architecture, budget, sampling strategy, optimizer, LR schedule, adaptive weights, loss function and mode.

- **Collocation point**: a point where the equation's residual is evaluated.
- **Sampling strategy**: the rule that places the collocation points. The RL bandit is an experimental one.
- **LR schedule**: how the learning rate changes over training.
- **Adaptive weights**: weights of the loss terms that are adjusted during training.
- **Inverse mode**: training that also identifies parameters of the equation. The form asks for the parameters to identify, their initial guesses, and the observation source: synthetic noisy samples of the exact solution, or an `.npz` file.
- **Well dataset**: a simulation dataset of [The Well](datasets.md). Choosing one fills in its recommended mode and architecture.

![New training](assets/screenshots/dashboard-new-training.png)

**Monitor** follows a run as it trains. It shows:

- the loss components and the learning rate;
- the parameter trajectory in inverse mode, against the true value when known;
- interactive 3-D surfaces of the predicted `u(x, t)` and the PDE residual.

The surfaces are read from the run's `live_snapshot.npz`, written every `validation_frequency` epochs plus at the start and the end. 2-D PDEs are shown as a time slice.

## Collocation & Solution

For a finished run, this tab shows:

- the predicted and exact solution surfaces, with a time slider;
- the point-wise error;
- where the collocation points were placed over training.

The last view is how you check what an adaptive sampler did. The exact-solution panel is empty for PDEs without a closed form.

![Collocation and solution](assets/screenshots/dashboard-collocation.png)

## Process

This tab is the soft sensor of the evaporator twin on plant data. It runs the [fouling monitor](process_twins.md#estimating-a-drifting-parameter). No network is trained, so the result is there in a few seconds.

- **Soft sensor**: an estimate of a quantity the plant does not measure, computed from the sensors it has.
- **Fouling resistance**: the extra resistance to heat transfer of the deposit on the tube.
- **Fouling monitor**: a Kalman filter on the evaporator model, with the fouling resistance as a level and a rate.
- **Kalman filter**: an estimator that corrects a model's state with each new record and carries its uncertainty.
- **Historian export**: the plant's recorded sensor data, as CSV or Parquet.
- **Tag map**: a YAML file that says which column of the export is which signal ([format](process_twins.md#plant-data-a-historian-export-and-a-tag-map)).

To run it, give the historian export and the tag map. The view shows:

| Output | Content |
|---|---|
| Fouling resistance | The estimate at each record with its 90 % band (what was known at the time), the hindsight estimate from the whole record, the cleaning threshold, and the projected time of reaching it with its interval |
| Outlet records | The records against the filter's one-step predictions |
| Prediction misfit (in the summary) | In units of the sensor noise. Near 1 when the model explains the records; a larger value says the band should not be trusted |

- **Open interval**: an interval for the time to clean whose upper end is open means the fouling rate is not yet clearly positive.

- **Steam record uncertainty**: a steam-temperature offset mimics fouling (0.5 K for 30 %, experiment J04). Enter how well the steam record is trusted and the offset is estimated alongside. This widens the band to what the data support ([J06](benchmarks.md#calibrated-fouling-intervals-experiment-j06)).
- **Rate walk**: how fast the fouling rate itself may change.

$$
q = \frac{r}{\sqrt{\tau}}
$$

- **$q$**: the rate walk.
- **$r$**: the fouling rate.
- **$\tau$**: the time over which the rate may change by its own size.
- **Default**: suits a steady drift. On a curve that bends it gives a narrow band around a biased estimate.
- **Larger value**: the safe choice, at the price of a wider time to clean.

- **Write demo data** puts a synthetic 20-minute export and its tag map in `runs/process_demo/` and fills the form. The two files are a template for a real export.
- A `model:` block in the tag map overrides the tube's parameters (`length`, `U`, `velocity`, ...).

![Process view](assets/screenshots/dashboard-process.png)

## Comparison

Select several finished runs to overlay their loss curves and compare final metrics side by side.

![Comparison](assets/screenshots/dashboard-comparison.png)

- **Delete**: runs can be deleted from the selector, with a confirmation step. Only folders inside the runs directory can be removed.
- **Load in Python**: use `pinnrl.training.train.load_run` (see [Training](training.md#python)).
