"""Load `The Well` datasets and flatten them into pinnrl observation tensors.

The Well stores trajectories as ``(n_traj, n_steps, *spatial, n_fields)`` HDF5
arrays. pinnrl's data-fitting loss expects a flat point cloud of
``(x, t, u)`` tuples (see ``PDEBase._compute_data_loss``). The functions here
bridge those two shapes:

* :func:`resolve_path` returns the user-supplied local download dir, or the
  Hugging Face mirror when none is given.
* :func:`load_well_slice` instantiates ``the_well.data.WellDataset`` (lazily —
  the dependency is optional) in full-trajectory mode, draws ``n_points``
  random spatio-temporal locations from a few of its trajectories, reads only
  those, and returns ready-to-use CPU/GPU tensors.

Loaded slices are cached as ``.npz`` under ``~/.cache/pinnrl/well/`` (or
``$PINNRL_WELL_CACHE``) so repeat training runs (and CI smoke tests) do not
re-download or re-flatten.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .registry import get_entry

_LOGGER = logging.getLogger(__name__)
_DEFAULT_HF_BASE = "hf://datasets/polymathic-ai/"


class TheWellNotInstalledError(ImportError):
    """Raised when a Well dataset is requested without ``pinnrl[well]`` installed."""

    def __init__(self) -> None:
        super().__init__(
            "The Well datasets require the optional dependency. Install with:\n"
            "    pip install 'pinnrl[well]'\n"
            "or:\n"
            "    pip install the_well h5py huggingface-hub"
        )


def resolve_path(base: str | None = None) -> str:
    """Return the ``well_base_path`` to hand to ``WellDataset``.

    Args:
        base: Local directory if the user has run ``the-well-download``,
            otherwise ``None`` (or blank) to stream from Hugging Face.

    Returns:
        ``base`` as a string, or ``"hf://datasets/polymathic-ai/"``.
    """
    if base is None or not str(base).strip():
        return _DEFAULT_HF_BASE
    return str(base)


def _cache_dir() -> Path:
    """Where flattened ``.npz`` slices are stored (``$PINNRL_WELL_CACHE`` overrides)."""
    root = os.environ.get("PINNRL_WELL_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "pinnrl" / "well"


def _base_tag(base: str | None) -> str:
    """Stable short tag for the resolved base path.

    Distinguishes HF streaming from local mirrors and from each other so two
    on-disk copies of the same dataset (different versions, different
    download dirs) cannot collide on the same flattened ``.npz``.
    """
    import hashlib

    resolved = resolve_path(base)
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]


# Bump when the flattened (x, t, u) layout or its coordinates change, so slices
# cached by an older loader are not reused. v2: real time/space grids from
# ``input_time_grid``/``output_time_grid``/``space_grid`` (v1 used placeholders).
_CACHE_FORMAT = "v2"


def _cache_file(
    name: str,
    split: str,
    n_traj: int,
    n_points: int,
    seed: int,
    base: str | None,
    n_steps: int,
) -> Path:
    tag = _base_tag(base)
    return (
        _cache_dir()
        / f"{name}__{split}__t{n_traj}_p{n_points}_s{seed}_n{n_steps}__{tag}__{_CACHE_FORMAT}.npz"
    )


def _load_well_dataset(name: str, split: str, base: str | None, n_steps: int):
    """Import ``the_well`` lazily and open ``name`` in full-trajectory mode.

    Each sample is then one whole trajectory from ``t0``: ``input_fields`` holds
    the first step and ``output_fields`` the rest, with ``input_time_grid`` /
    ``output_time_grid`` and ``space_grid`` giving the coordinates. ``the_well``
    loads the first step plus up to ``max_rollout_steps`` more, so reading
    ``n_steps`` steps maps to ``max_rollout_steps = n_steps - 1``.
    """
    try:
        from the_well.data import WellDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise TheWellNotInstalledError() from exc

    return WellDataset(
        well_base_path=resolve_path(base),
        well_dataset_name=name,
        well_split_name=split,
        full_trajectory_mode=True,
        max_rollout_steps=int(n_steps) - 1,
    )


def _extract_arrays(sample: Any, n_spatial_dims: int) -> dict[str, Any]:
    """Collect one full-trajectory ``WellDataset`` sample without copying its fields.

    Returns ``{"segments", "times", "axes"}``: ``segments`` are ``input_fields``
    and ``output_fields`` as stored (``(steps, *spatial, n_fields)`` each, in
    time order), ``times`` is ``input_time_grid`` followed by
    ``output_time_grid``, and ``axes[d]`` is the coordinate of spatial axis
    ``d``, read from ``space_grid[..., d]`` along that axis.

    Raises:
        KeyError: If the sample lacks fields or grids (``return_grid`` must be on).
        ValueError: If shapes disagree with ``n_spatial_dims`` or each other.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"Expected WellDataset sample to be a dict, got {type(sample).__name__}")

    def _as_np(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    required = ("input_fields", "input_time_grid", "space_grid")
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(
            f"WellDataset sample is missing {missing} (got keys {sorted(sample)}); "
            "open the dataset with full_trajectory_mode=True and return_grid=True"
        )

    segments = [_as_np(sample["input_fields"])]
    times = [_as_np(sample["input_time_grid"]).reshape(-1)]
    if "output_fields" in sample:
        segments.append(_as_np(sample["output_fields"]))
        times.append(_as_np(sample["output_time_grid"]).reshape(-1))
    times = np.concatenate(times).astype(np.float32)
    shape = segments[0].shape[1:]
    for seg in segments:
        if seg.ndim != n_spatial_dims + 2:
            raise ValueError(
                f"Expected fields with {n_spatial_dims + 2} dims "
                f"(n_steps, {'x ' * n_spatial_dims}n_fields), got shape {seg.shape}"
            )
        if seg.shape[1:] != shape:
            raise ValueError(f"output_fields {seg.shape} do not match input_fields {shape}")
    n_steps = sum(len(seg) for seg in segments)
    if times.size != n_steps:
        raise ValueError(f"{times.size} time stamps for {n_steps} field steps")

    space_grid = _as_np(sample["space_grid"])
    if space_grid.shape != (*shape[:-1], n_spatial_dims):
        raise ValueError(
            f"space_grid shape {space_grid.shape} does not match fields {(n_steps, *shape)}"
        )
    axes = []
    for d in range(n_spatial_dims):
        index = tuple(slice(None) if i == d else 0 for i in range(n_spatial_dims))
        axes.append(space_grid[index + (d,)].astype(np.float32))

    return {"segments": segments, "times": times, "axes": axes}


def _n_nodes(traj: dict[str, Any]) -> int:
    """Space-time nodes in one trajectory from :func:`_extract_arrays`."""
    return traj["times"].size * int(np.prod(traj["segments"][0].shape[1:-1]))


def _gather(trajectories: list[dict[str, Any]], flat: np.ndarray) -> dict[str, np.ndarray]:
    """Read the ``(x, t, u)`` of flat indices into the stacked trajectories.

    Index ``k`` counts trajectory by trajectory, each in C order over
    ``(time, *spatial)`` — the order of a flattened ``(n_steps, *spatial,
    n_fields)`` array. Only the selected nodes are read.
    """
    offsets = np.concatenate([[0], np.cumsum([_n_nodes(traj) for traj in trajectories])])
    which = np.searchsorted(offsets, flat, side="right") - 1
    n_dims = len(trajectories[0]["axes"])
    n_fields = trajectories[0]["segments"][0].shape[-1]
    x = np.empty((flat.size, n_dims), dtype=np.float32)
    t = np.empty((flat.size, 1), dtype=np.float32)
    u = np.empty((flat.size, n_fields), dtype=np.float32)
    for i, traj in enumerate(trajectories):
        rows = np.flatnonzero(which == i)
        grid = (traj["times"].size, *traj["segments"][0].shape[1:-1])
        step, *nodes = np.unravel_index(flat[rows] - offsets[i], grid)
        t[rows, 0] = traj["times"][step]
        for d, node in enumerate(nodes):
            x[rows, d] = traj["axes"][d][node]
        start = 0
        for seg in traj["segments"]:
            hit = (step >= start) & (step < start + len(seg))
            u[rows[hit]] = seg[(step[hit] - start, *(node[hit] for node in nodes))]
            start += len(seg)
    return {"x": x, "t": t, "u": u}


def load_well_slice(
    name: str,
    split: str = "train",
    n_traj: int = 1,
    n_points: int = 4096,
    seed: int = 0,
    device: str = "cpu",
    base: str | None = None,
    use_cache: bool = True,
    n_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sample ``n_points`` flat ``(x, t, u)`` tuples from a Well dataset.

    Args:
        name: Registry key — must exist in ``WELL_REGISTRY``.
        split: Dataset split (``train``, ``valid``, ``test``).
        n_traj: How many ``WellDataset`` samples to draw (random, without
            replacement, capped at ``len(dataset)``). Points are pooled
            across them before sub-sampling.
        n_points: Number of (space, time) samples to retain; all are kept
            when the pool is smaller.
        seed: RNG seed for the sample choice and the sub-sampling.
        device: Torch device for the returned tensors.
        base: Local download dir; ``None`` = HF streaming.
        use_cache: When True, reuse / write a flattened ``.npz`` under
            the cache dir (``~/.cache/pinnrl/well/`` or ``$PINNRL_WELL_CACHE``).
            Set False to force a fresh load.
        n_steps: Time steps read per trajectory, from its start (fewer when
            the trajectory is shorter); ``None`` uses the entry's
            ``default_loaded_steps`` (101, ``the_well``'s rollout cap; 11 for
            3-D datasets). Memory grows with it: the
            ``n_traj`` trajectories are held as ``the_well`` returns them,
            but only the ``n_points`` selected nodes are copied out.

    Returns:
        ``{"x": (N, d_space), "t": (N, 1), "u": (N, d_fields)}`` on ``device``.

    Raises:
        KeyError: If ``name`` is not registered.
        TheWellNotInstalledError: If the optional ``the_well`` package is missing.
        RuntimeError: If the split is empty.
        ValueError: If ``n_steps`` is below 1.
    """
    entry = get_entry(name)
    n_steps = entry.steps_to_read(n_steps)
    rng = np.random.default_rng(seed)

    cache_path = _cache_file(name, split, n_traj, n_points, seed, base, n_steps)
    if use_cache and cache_path.exists():
        _LOGGER.info("Loading cached Well slice from %s", cache_path)
        with np.load(cache_path) as data:
            return {
                "x": torch.tensor(data["x"], device=device),
                "t": torch.tensor(data["t"], device=device),
                "u": torch.tensor(data["u"], device=device),
            }

    dataset = _load_well_dataset(name, split, base, n_steps=n_steps)
    n_available = len(dataset)
    if n_available == 0:
        raise RuntimeError(f"Well dataset {name!r} split {split!r} is empty")
    take = min(n_traj, n_available)
    traj_indices = rng.choice(n_available, size=take, replace=False)

    trajectories = [
        _extract_arrays(dataset[int(idx)], entry.n_spatial_dims) for idx in traj_indices
    ]
    n_total = sum(_n_nodes(traj) for traj in trajectories)
    # Draw the flat indices first (O(n_points) memory), then read only those nodes.
    if n_points < n_total:
        flat = rng.choice(n_total, size=n_points, replace=False)
    else:
        flat = np.arange(n_total)
    sampled = _gather(trajectories, flat)
    x, t, u = sampled["x"], sampled["t"], sampled["u"]

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, x=x, t=t, u=u)
        _LOGGER.info("Cached Well slice at %s", cache_path)

    return {
        "x": torch.tensor(x, device=device),
        "t": torch.tensor(t, device=device),
        "u": torch.tensor(u, device=device),
    }
