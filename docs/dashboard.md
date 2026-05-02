# Dashboard Guide

The pinnrl dashboard is a browser-based interface for configuring, launching, and monitoring PINN training experiments. It replaces the need for manual CLI invocations and plotting scripts.

## Launching the dashboard

```bash
pinnrl-dashboard
```

Open [http://127.0.0.1:8050/](http://127.0.0.1:8050/) in your browser.

---

## Layout overview

The dashboard is organized into three tabs:

| Tab | Purpose |
|-----|---------|
| **Live Training** | Launch new experiments and monitor running/completed ones |
| **Comparison** | Compare architectures and PDEs side by side |
| **Collocation & Solution** | Visualize collocation point distributions and solution surfaces |

---

## Live Training

The Live Training tab has two sub-tabs: **Monitor** and **New Training**.

### Monitor

Select any experiment from the dropdown to see its training progress, loss curve, current status, and a live 3D snapshot of the solution.

![Dashboard monitor view](assets/screenshots/dashboard-landing.png)

The monitor displays:

- **Experiment selector** — dropdown listing all experiments in the `experiments/` directory
- **Progress bar** — shows epoch progress for running experiments
- **Loss graph** — total loss, residual loss, boundary loss, and initial condition loss over epochs (and `data_loss` in inverse / data_only / data_augmented modes)
- **Live Solution Snapshot** — two side-by-side interactive 3D Plotly surfaces (predicted `u(x, t)` and PDE residual) refreshed every epoch from the experiment's `live_snapshot.npz`. See [Visualization](visualization.md) for details.
- **Identified-parameter trajectories** — shown only for inverse-mode runs; plots each trainable PDE parameter's value over training epochs. See [Inverse Problems](inverse_problems.md).

Experiments are auto-detected from the `experiments/` directory. Running experiments show a `.running` indicator and refresh automatically.

### New Training

Configure and launch a new training run directly from the browser.

![New Training form](assets/screenshots/dashboard-new-training.png)

**Model Configuration** (left column):

- **PDE** — select from 9 supported PDEs. Changing the PDE auto-selects the recommended architecture.
- **Architecture** — choose from 7 neural architectures (feedforward, resnet, siren, fourier, fno, attention, autoencoder).
- **Device** — CPU, MPS (Apple Silicon), or CUDA.
- **RL Adaptive Sampling** — toggle to enable the DQN-based collocation agent.
- **Mode** — `Forward (solve)` is the default. `Inverse (identify)` reveals the parameter-identification panel. `Data-augmented (residual + data)` and `Data-only (regression)` are for training against benchmark datasets — see the Dataset section below.

**Dataset** (optional, left column):

- **Train on a Well benchmark dataset** — toggle to expose the dataset controls.
- **Dataset** — pick from `pinnrl.datasets.WELL_REGISTRY`. Selecting a dataset auto-fills the PDE selector, recommended architecture, and Mode dropdown from the registry entry.
- **Source** — `Hugging Face streaming` (default) reads from `hf://datasets/polymathic-ai/`; `Local download dir` reads from a path you have populated with `the-well-download`.
- **Trajectories / Sampled points** — how much of the dataset to flatten into training observations.
- See [The Well datasets](datasets.md) for a full description of each dataset and its scientific use cases.

**Inverse Problem panel** (revealed when Mode = Inverse):

- **Parameters to identify** — checklist of trainable PDE parameters auto-populated from the selected PDE.
- **Initial guesses** — one numeric input per checked parameter; far-from-truth guesses verify the inversion actually moves.
- **Observation source** — `Synthetic` generates noisy points from the analytical solution; `File` accepts an `.npz` with `x`, `t`, `u` keys.
- **Observation points / Noise std / File path** — fine controls for the synthetic / file paths.
- See [Inverse Problems](inverse_problems.md) for the full workflow.

**Hyperparameters** (right column):

- **Epochs** — number of training epochs (default: 3000)
- **Learning Rate** — optimizer learning rate (default: 0.005)
- **Batch Size** — training batch size (default: 2048)
- **Collocation Points** — interior sampling points (default: 5000)
- **Boundary Points** — boundary condition points (default: 500)
- **Initial Points** — initial condition points (default: 500)
- **Optimizer** — Adam, L-BFGS, or two-phase Adam → L-BFGS.
- **Loss Function** — MSE, MAE, or Huber (configurable `huber_delta`).

Click **Start Training** to launch. Training runs as a background process — you can close the browser and results are saved automatically to a timestamped directory under `experiments/`.

---

## Comparison

Compare training results across architectures or PDEs.

![Comparison tab](assets/screenshots/dashboard-comparison.png)

Select completed experiments to overlay their loss curves and accuracy metrics. This is useful for:

- Benchmarking architectures on the same PDE
- Comparing convergence rates across different hyperparameter choices
- Identifying which architecture best suits a particular equation type

---

## Collocation & Solution

Visualize the trained solution surface and collocation point distribution for any completed experiment.

![Collocation & Solution tab](assets/screenshots/dashboard-collocation.png)

- **Collocation points** — 2D scatter showing where training points were sampled, colour-coded across snapshots so you can see how the RL agent (or RAR) concentrated points around shocks and interfaces.
- **Exact solution** (3D, left) — analytical reference surface. For Well-dataset runs in `data_only` mode, this panel falls back to a placeholder since no closed-form reference exists.
- **Predicted solution** (3D, right) — network output `u_θ(x, t)` over the domain.
- **Time slider** — scrub through time; both 3D surfaces update in lockstep.

This tab loads the saved model checkpoint (`final_model.pt`) and reconstructs the solution on a dense grid for visualization. See [Visualization](visualization.md) for the file formats and programmatic access.

---

## Experiment directory structure

Each training run creates a directory under `experiments/`:

```
experiments/
  20260310_143000_Heat Equation_fourier_no_rl/
    config.yaml          # full config snapshot
    metadata.json        # status, timing, PDE, architecture
    final_model.pt       # trained model weights
    loss_curve.png       # training loss plot
    solution_plot.png    # predicted vs exact solution
    metrics.json         # L2 error, max error, mean error
    visualizations/      # additional plots
```

While training is in progress, a `.running` marker file is present. It is removed when training completes or fails.

---

## Tips

- **Start simple**: Use the Heat Equation with Fourier Features and 3000 epochs for your first run. It converges reliably and lets you verify the setup works.
- **Architecture recommendations**: The PDE dropdown auto-selects a recommended architecture. These defaults come from extensive benchmarking and are a good starting point.
- **RL sampling**: Enable RL adaptive sampling for nonlinear PDEs with sharp gradients (Burgers, Allen-Cahn). For smooth problems (Heat, Wave), uniform sampling is sufficient.
- **Monitor convergence**: If the loss plateaus early, try increasing collocation points or switching architectures before increasing epochs.
