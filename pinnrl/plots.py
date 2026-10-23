"""Plots of PINN solutions and training runs.

One module for every figure pinnrl draws:

* :func:`plot_solution` — exact and predicted solution surfaces (the dashboard).
* :func:`save_training_plots` — the figures :class:`~pinnrl.training.trainer.PDETrainer`
  writes into a run's ``visualizations/`` folder: training history, solution
  comparison and collocation-point evolution.

Model and exact solution are evaluated on a grid by :func:`solution_grid`,
shared by all solution plots. Plotly figures are always saved as HTML; a PNG
copy is written only when ``kaleido`` is installed.
"""

import importlib.util
import logging
import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

_HAS_KALEIDO = importlib.util.find_spec("kaleido") is not None
_LOGGER = logging.getLogger(__name__)


def _write_plotly(fig, save_path):
    """Write ``fig`` as HTML, plus the static image at ``save_path`` if possible."""
    html_path = os.path.splitext(save_path)[0] + ".html"
    fig.write_html(html_path)
    if _HAS_KALEIDO:
        fig.write_image(save_path)
    _LOGGER.info(f"Saved {html_path}")


def solution_grid(
    model, pde, n: int, t: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``model`` and ``pde.exact_solution`` on a regular grid.

    1-D PDEs use an ``(x, t)`` grid over the whole time domain, or the single
    time ``t`` when given. 2-D PDEs use an ``(x1, x2)`` grid at time ``t``
    (default: the start of the time domain).

    Args:
        model: Trained network; its parameters' device is used.
        pde: PDE providing the domain and ``exact_solution``.
        n: Points per axis.
        t: Optional time of the slice.

    Returns:
        ``(A, B, u_pred, u_exact)``, each of shape ``(n, n)`` (or ``(n, 1)``
        for a single-time 1-D slice), with ``A, B`` the grid coordinates and
        ``u_pred`` the model's output channel 0. ``u_exact`` is ``None`` when
        the PDE has no exact solution.
    """
    device = next(model.parameters()).device
    t_lo, t_hi = float(pde.time_domain[0]), float(pde.time_domain[1])
    if pde.dimension == 1:
        a = torch.linspace(pde.domain[0][0], pde.domain[0][1], n, device=device)
        b = (
            torch.tensor([float(t)], device=device)
            if t is not None
            else torch.linspace(t_lo, t_hi, n, device=device)
        )
        A, B = torch.meshgrid(a, b, indexing="ij")
        x = A.reshape(-1, 1)
        tt = B.reshape(-1, 1)
    else:
        a = torch.linspace(pde.domain[0][0], pde.domain[0][1], n, device=device)
        b = torch.linspace(pde.domain[1][0], pde.domain[1][1], n, device=device)
        A, B = torch.meshgrid(a, b, indexing="ij")
        x = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
        tt = torch.full((x.shape[0], 1), t_lo if t is None else float(t), device=device)
    with torch.no_grad():
        u_pred = model(torch.cat([x, tt], dim=1))[:, 0:1]
        u_exact = pde.exact_solution(x, tt)
    shape = A.shape
    return (
        A.cpu().numpy(),
        B.cpu().numpy(),
        u_pred.reshape(shape).cpu().numpy(),
        None if u_exact is None else u_exact.reshape(shape).cpu().numpy(),
    )


def _log_min_error(pred: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """log10 of the smaller of the absolute and relative error."""
    abs_error = np.abs(pred - exact)
    rel_error = np.abs((pred - exact) / (exact + 1e-10))
    return np.log10(np.minimum(abs_error, rel_error) + 1e-10)


# ---------------------------------------------------------------------------
# Dashboard: exact vs predicted surfaces
# ---------------------------------------------------------------------------


def plot_solution(
    model,
    pde,
    num_points: int = 1000,
    save_path: str | None = None,
    return_figs: bool = False,
    time_point: float | None = None,
):  # pragma: no cover
    """Exact and predicted solution as two interactive 3-D surfaces.

    Args:
        model: The trained PINN model.
        pde: The PDE instance.
        num_points: Total grid points (``sqrt(num_points)`` per axis, and as many
            time frames for an animated 2-D plot).
        save_path: Optional ``.html`` path; writes ``*_exact.html`` and ``*_predicted.html``.
        return_figs: Return ``(exact_fig, predicted_fig)`` instead of saving/showing.
        time_point: Time of the slice for 2-D PDEs (animated over time when
            omitted). 1-D PDEs always show the whole ``(x, t)`` surface: a
            single-time slice of a 1-D field is a line, which a surface cannot draw.
    """
    n = int(np.sqrt(num_points))
    if pde.dimension == 2:
        times = (
            [float(time_point)]
            if time_point is not None
            else list(np.linspace(float(pde.time_domain[0]), float(pde.time_domain[1]), n))
        )
        grids = [solution_grid(model, pde, n, t) for t in times]
        A, B = grids[0][0], grids[0][1]
        preds = [g[2] for g in grids]
        exacts = [g[3] for g in grids]
    else:
        A, B, pred, exact = solution_grid(model, pde, n)
        times, preds, exacts = [None], [pred], [exact]

    if exacts[0] is None:
        exact_fig = go.Figure()
        exacts = [None] * len(exacts)
    else:
        exact_fig = go.Figure(
            data=[go.Surface(x=A, y=B, z=exacts[0], colorscale="viridis", showscale=False)]
        )
    predicted_fig = go.Figure(
        data=[go.Surface(x=A, y=B, z=preds[0], colorscale="plasma", showscale=False)]
    )
    if len(times) > 1:
        labels = [f"t={t:.2f}" for t in times]
        if exacts[0] is not None:
            exact_fig.frames = [
                go.Frame(data=[go.Surface(x=A, y=B, z=z)], name=label)
                for z, label in zip(exacts, labels, strict=False)
            ]
        predicted_fig.frames = [
            go.Frame(data=[go.Surface(x=A, y=B, z=z)], name=label)
            for z, label in zip(preds, labels, strict=False)
        ]
        sliders = [
            dict(
                steps=[dict(method="animate", args=[[label]], label=label) for label in labels],
                currentvalue=dict(visible=True, prefix="Time: "),
                len=0.9,
            )
        ]
        exact_fig.update_layout(sliders=sliders)
        predicted_fig.update_layout(sliders=sliders)

    axis = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
        showspikes=False,
        showbackground=True,
        backgroundcolor="rgba(240, 240, 240, 0.5)",
    )
    scene = dict(
        xaxis=dict(title="x", **axis),
        yaxis=dict(title="t" if pde.dimension == 1 else "y", **axis),
        zaxis=dict(title="u(x,t)" if pde.dimension == 1 else "u(x,y,t)", **axis),
        camera=dict(
            up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        dragmode="turntable",
        aspectmode="cube",
    )
    margin = dict(l=0, r=0, b=0, t=30)
    exact_title = "Exact Solution" if exacts[0] is not None else "No exact solution configured"
    exact_fig.update_layout(title=exact_title, scene=scene, margin=margin)
    predicted_fig.update_layout(title="PINN Prediction", scene=scene, margin=margin)

    if return_figs:
        return exact_fig, predicted_fig
    if save_path:
        exact_fig.write_html(save_path.replace(".html", "_exact.html"))
        predicted_fig.write_html(save_path.replace(".html", "_predicted.html"))
    else:
        exact_fig.show()
        predicted_fig.show()


# ---------------------------------------------------------------------------
# Post-training figures for a PDETrainer run
# ---------------------------------------------------------------------------


def loss_curve_epochs(history, validation_frequency):
    """1-based x positions for the per-epoch losses and for ``val_loss``.

    Every ``*_loss`` series except ``val_loss`` has one value per epoch. The
    trainer validates at 0-based epochs ``0, f, 2f, ...``, i.e. 1-based epochs
    ``1, f + 1, 2f + 1, ...``.
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    val = [1 + k * validation_frequency for k in range(len(history.get("val_loss", [])))]
    return epochs, val


def plot_training_history(trainer, save_path=None):  # pragma: no cover
    """Loss curves on a log axis, plus loss-weight curves when adaptive weights are on.

    Only the ``*_loss`` series are drawn: ``val_loss`` at the epochs it was
    measured, the others once per epoch. The learning rate and parameter
    trajectories are not losses and are left out.
    """
    history = trainer.history
    epochs, val_epochs = loss_curve_epochs(history, trainer.validation_frequency)
    loss_keys = [key for key, values in history.items() if key.endswith("_loss") and values]
    weights = np.array(history.get("loss_weights", []))
    with_weights = trainer.use_adaptive_weights and len(weights) > 0
    if with_weights:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Losses", "Loss Weights"))
    else:
        fig = go.Figure()
    loss_cell = {"row": 1, "col": 1} if with_weights else {}
    for key in loss_keys:
        x = val_epochs if key == "val_loss" else epochs[: len(history[key])]
        fig.add_trace(go.Scatter(x=x, y=history[key], name=key, mode="lines"), **loss_cell)
    if with_weights:
        for i, component in enumerate(["residual", "boundary", "initial"]):
            if i < weights.shape[1]:
                fig.add_trace(
                    go.Scatter(
                        x=epochs[: len(weights)],
                        y=weights[:, i],
                        name=f"{component}_weight",
                        mode="lines",
                    ),
                    row=2,
                    col=1,
                )
        fig.update_layout(title="Training History", height=800)
        fig.update_yaxes(type="log", title="Loss", row=1, col=1)
        fig.update_yaxes(title="Weight Value", row=2, col=1)
        fig.update_xaxes(title="Epoch", row=2, col=1)
    else:
        fig.update_layout(
            title="Training History", xaxis_title="Epoch", yaxis_title="Loss", yaxis_type="log"
        )
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        _write_plotly(fig, save_path)


def plot_solution_comparison(trainer, num_points=50, save_path=None):  # pragma: no cover
    """Exact, predicted and log-error surfaces side by side.

    ``num_points`` is the grid size per axis. 2-D PDEs get ten time frames
    with a slider and a Play button. Skipped when the PDE has no exact solution.
    """
    pde, model = trainer.pde, trainer.model
    if not pde.has_exact_solution:
        _LOGGER.info("No exact solution configured; skipping the solution comparison plot")
        return
    titles = ("Exact", "PINN Predicted", "Log Min Error")
    specs = [[{"type": "surface"}] * 3]
    if pde.dimension == 2:
        times = np.linspace(float(pde.time_domain[0]), float(pde.time_domain[1]), 10)
        frames = []
        for t in times:
            A, B, pred, exact = solution_grid(model, pde, num_points, t)
            frames.append(
                go.Frame(
                    data=[
                        go.Surface(x=A, y=B, z=exact, colorscale="viridis", showscale=False),
                        go.Surface(x=A, y=B, z=pred, colorscale="viridis", showscale=False),
                        go.Surface(x=A, y=B, z=_log_min_error(pred, exact), colorscale="viridis"),
                    ],
                    traces=[0, 1, 2],
                    name=f"t={t:.2f}",
                )
            )
        fig = make_subplots(rows=1, cols=3, subplot_titles=titles, specs=specs)
        for col, trace in enumerate(frames[0].data, start=1):
            fig.add_trace(trace, row=1, col=col)
        fig.frames = frames
        fig.update_layout(
            title="Solution Comparison (2D)",
            width=1800,
            height=600,
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True},
                            ],
                        }
                    ],
                }
            ],
            sliders=[
                {
                    "currentvalue": {"prefix": "t="},
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ],
                }
            ],
        )
    else:
        A, B, pred, exact = solution_grid(model, pde, num_points)
        fig = make_subplots(rows=1, cols=3, subplot_titles=titles, specs=specs)
        fig.add_trace(
            go.Surface(x=A, y=B, z=exact, colorscale="viridis", showscale=False), row=1, col=1
        )
        fig.add_trace(
            go.Surface(x=A, y=B, z=pred, colorscale="viridis", showscale=False), row=1, col=2
        )
        fig.add_trace(
            go.Surface(
                x=A,
                y=B,
                z=_log_min_error(pred, exact),
                colorscale="viridis",
                colorbar=dict(title="Log Error", x=0.98, y=0.5),
            ),
            row=1,
            col=3,
        )
        axes = dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)")
        fig.update_layout(
            title="Solution Comparison (1D)",
            width=1800,
            height=600,
            scene=axes,
            scene2=axes,
            scene3=axes,
        )
    if save_path:
        _write_plotly(fig, save_path)


def _plot_density_heatmap(ax, points, cmap, title, x_range, y_range, xlabel, ylabel, bins=50):
    """2-D histogram of the first two point coordinates."""
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=[x_range, y_range])
    im = ax.imshow(
        counts.T,
        origin="lower",
        aspect="auto",
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        cmap=cmap,
    )
    plt.colorbar(im, ax=ax).set_label("Number of Points", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)


def _snapshot_indices(n: int) -> Sequence[int]:
    """Up to five evenly spaced indices into ``n`` snapshots, always ending at the last."""
    if n < 5:
        return list(range(n))
    step = n // 5
    indices = [i * step for i in range(5)]
    indices[-1] = n - 1
    return indices


def plot_collocation_evolution(trainer, save_path=None):  # pragma: no cover
    """2×2 grid: point snapshots over training, and initial/mid/final densities.

    1-D PDEs are drawn in the ``(x, t)`` plane, 2-D PDEs in the ``(x, y)`` plane.
    """
    history = getattr(trainer, "points_history", None)
    if not history:
        _LOGGER.warning("No collocation points history available")
        return
    pde = trainer.pde
    if pde.dimension == 1:
        x_range, y_range, ylabel = pde.domain[0], pde.time_domain, "t"
        progression_title = "Progression of Points"
    else:
        x_range, y_range, ylabel = pde.domain[0], pde.domain[1], "y"
        progression_title = "Progression of Points (x-y projection)"

    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Evolution of Collocation Points", fontsize=20)

    ax = axs[0, 0]
    for idx in _snapshot_indices(len(history)):
        points = history[idx]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=3,
            alpha=0.5,
            label=f"Snapshot {idx}",
            marker=".",
            edgecolors="none",
        )
    ax.set_title(progression_title, fontsize=16)
    ax.legend(fontsize=10)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(alpha=0.3)

    mid = len(history) // 2
    for ax, points, cmap, title in [
        (axs[0, 1], history[0], plt.cm.Blues, "Initial Distribution"),
        (axs[1, 0], history[mid], plt.cm.Greens, f"Intermediate Distribution (Snapshot {mid})"),
        (axs[1, 1], history[-1], plt.cm.Reds, "Final Distribution"),
    ]:
        _plot_density_heatmap(ax, points, cmap, title, x_range, y_range, "x", ylabel)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        _LOGGER.info(f"Collocation evolution saved to {save_path}")
    plt.close(fig)


def save_training_plots(trainer, save_dir):  # pragma: no cover
    """Write every post-training figure into ``save_dir``.

    Each figure is attempted independently, so one failing plot never loses
    the others or crashes the run.
    """
    os.makedirs(save_dir, exist_ok=True)
    jobs = [
        ("final_training_history.png", plot_training_history, {}),
        ("final_solution_comparison.png", plot_solution_comparison, {"num_points": 100}),
        ("collocation_evolution.png", plot_collocation_evolution, {}),
    ]
    for filename, plot, kwargs in jobs:
        try:
            plot(trainer, save_path=os.path.join(save_dir, filename), **kwargs)
        except Exception as e:
            _LOGGER.warning(f"Could not write {filename}: {e}")
