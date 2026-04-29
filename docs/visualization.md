# 3D Visualization

`pinnrl` ships interactive 3D visualizations in two places: the dashboard's **Live Solution Snapshot** (Monitor sub-tab) and the **Collocation & Solution** tab. Both render Plotly `go.Surface` plots that you can rotate, zoom, and probe in the browser.

---

## Live training snapshot (Monitor sub-tab)

While a training run is in progress, the trainer periodically writes a small grid of model predictions and PDE residuals to `live_snapshot.npz` inside the experiment directory. The dashboard's Monitor tab polls the file every 10 seconds and renders two side-by-side surfaces:

| Panel | Meaning |
|---|---|
| **Predicted u(x, t)** (left) | The current network prediction over a fixed grid. |
| **PDE residual** (right) | Pointwise residual `\|F[u_θ]\|` on the same grid. Hot spots are where the network still violates the PDE. |

A small caption above the surfaces shows the latest epoch number, so you can tell whether the snapshot is fresh.

### What the file contains

`live_snapshot.npz` is written by `PDETrainer._save_live_snapshot` and contains:

| Key | Description |
|---|---|
| `axis_x`, `axis_y` | 1D coordinate arrays for the two plotted axes |
| `u_pred` | `(grid_size, grid_size)` predicted field |
| `residual` | `(grid_size, grid_size)` PDE residual field |
| `epoch` | Integer training epoch the snapshot was taken at |
| `dimension` | 1 (axes are `x` and `t`) or 2 (axes are `x1` and `x2` at fixed `t`) |
| `x_label`, `y_label` | Axis labels for the rendered plots |
| `fixed_t` | For 2D PDEs, the time slice the surface was rendered at (`NaN` for 1D) |

The default grid is 60 × 60 — small enough that snapshots write in milliseconds and the dashboard refresh stays smooth. Adjust by passing a different `grid_size` to `_save_live_snapshot`.

### How often snapshots are written

Snapshots refresh every epoch, with a final snapshot saved when training finishes — so the surface you see at the end reflects the trained model, not an early-epoch view.

If a snapshot fails to render (e.g. because the residual computation hit a non-differentiable point), the trainer logs a debug message and continues training. The surface in the dashboard will simply show the last successful snapshot.

### 1D vs 2D PDEs

- **1D PDE** (`dimension=1`): `u_pred` and `residual` are surfaces over `(x, t)` — the natural visualization of a scalar field evolving in time.
- **2D PDE** (`dimension=2`): the trainer fixes `t` at the midpoint of `time_domain` and renders surfaces over `(x1, x2)`. To see a different time, edit `fixed_t = 0.5 * (time_lo + time_hi)` in `_save_live_snapshot` (or, in a future version, expose a slider).

---

## Collocation & Solution tab

The third dashboard tab provides post-hoc visualization of any completed experiment.

| Panel | Description |
|---|---|
| **Collocation Evolution** | 2D scatter showing where collocation points landed over training. Multiple snapshots are colour-coded so you can see how the RL agent (or RAR) concentrated points around shocks, interfaces, or boundary layers. |
| **Exact Solution** (3D, left) | Analytical reference `u(x, t)` over the chosen time slice. |
| **Predicted Solution** (3D, right) | Network output `u_θ(x, t)` on the same grid. |
| **Time slider** | Scrub through the time domain; both the exact and predicted surfaces update in lockstep. |

This tab loads the saved model checkpoint (`final_model.pt`) and reconstructs the surfaces on a dense grid, so render quality is higher than the live snapshot.

### When the exact-solution panel is empty

For Well datasets in `data_only` mode there is no closed-form analytical solution to compare against. The exact-solution panel falls back to a placeholder figure with a friendly message; the predicted-solution panel still works as the network's reproduction of the Well reference field.

---

## Working with snapshots programmatically

```python
import numpy as np
from pathlib import Path

experiment = Path("experiments") / "20260430_120000_Heat Equation_fourier_no_rl"
snapshot = np.load(experiment / "live_snapshot.npz")

print("epoch:", int(snapshot["epoch"]))
print("dimension:", int(snapshot["dimension"]))
print("u_pred shape:", snapshot["u_pred"].shape)

import matplotlib.pyplot as plt
fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
ax_u.imshow(snapshot["u_pred"], extent=[*snapshot["axis_x"][[0, -1]], *snapshot["axis_y"][[0, -1]]],
            origin="lower", aspect="auto", cmap="viridis")
ax_u.set_title(f"u_pred at epoch {int(snapshot['epoch'])}")
ax_r.imshow(snapshot["residual"], extent=[*snapshot["axis_x"][[0, -1]], *snapshot["axis_y"][[0, -1]]],
            origin="lower", aspect="auto", cmap="magma")
ax_r.set_title("residual")
plt.tight_layout()
plt.show()
```

This is convenient for slide decks and notebook exports where the interactive Plotly surfaces are not needed.

---

## Inverse-mode parameter trajectory

The Monitor sub-tab also includes a separate plot, **Identified-parameter trajectories**, that appears when the experiment was launched in inverse mode. It shows each trainable PDE parameter's value at every logged epoch — so you can watch (e.g.) `alpha` move from its initial guess toward the truth in real time. See the [inverse problems guide](inverse_problems.md) for the full inverse workflow.

---

## Tips

- **Start coarse, refine later.** Default 60 × 60 grids give smooth surfaces without slowing the trainer; only increase if you need publication-grade renders.
- **Watch the residual surface.** A bumpy residual that does not decrease alongside the loss is a sign the network is fitting the boundary/initial conditions but not the interior — try enabling RL adaptive sampling or RAR.
- **Snapshots survive process restarts.** If you re-launch the dashboard while a training run is ongoing, the live snapshot will pick up where it left off — no need to restart the trainer.
- **Multiple experiments side-by-side:** open multiple browser tabs pointing at the dashboard and select different experiments in each. The auto-refresh interval is independent per tab.

---

## See also

- [Dashboard guide](dashboard.md) — full layout reference.
- [Architecture overview](ARCHITECTURE.md) — where snapshot writing fits into the training loop.
- [Inverse problems](inverse_problems.md) — parameter trajectories and observation overlays.
- [Benchmark datasets](datasets.md) — visualizing Well-driven training runs.
