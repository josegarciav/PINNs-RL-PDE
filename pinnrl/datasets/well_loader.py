"""Load `The Well` datasets and flatten them into pinnrl observation tensors.

The Well stores trajectories as ``(n_traj, n_steps, *spatial, n_fields)`` HDF5
arrays. pinnrl's data-fitting loss expects a flat point cloud of
``(x, t, u)`` tuples (see ``PDEBase._compute_data_loss``). The functions here
bridge those two shapes:

* :func:`resolve_path` returns the user-supplied local download dir, or the
  Hugging Face mirror when none is given.
* :func:`load_well_slice` instantiates ``the_well.data.WellDataset`` (lazily —
  the dependency is optional) with its default arguments, flattens a few of
  its samples, keeps ``n_points`` random spatio-temporal locations, and
  returns ready-to-use CPU/GPU tensors.

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
    base: str | None = None,
    n_steps: int | None = None,
) -> Path:
    tag = _base_tag(base)
    steps = "all" if n_steps is None else str(n_steps)
    return (
        _cache_dir()
        / f"{name}__{split}__t{n_traj}_p{n_points}_s{seed}_n{steps}__{tag}__{_CACHE_FORMAT}.npz"
    )


def _load_well_dataset(name: str, split: str, base: str | None, n_steps: int | None = None):
    """Import ``the_well`` lazily and open ``name`` in full-trajectory mode.

    Each sample is then one whole trajectory from ``t0``: ``input_fields`` holds
    the first step and ``output_fields`` the rest, with ``input_time_grid`` /
    ``output_time_grid`` and ``space_grid`` giving the coordinates. ``n_steps``
    caps the number of steps read: ``the_well`` loads the first step plus up to
    ``max_rollout_steps`` more (default 100), so ``n_steps`` maps to
    ``max_rollout_steps = n_steps - 1``.
    """
    try:
        from the_well.data import WellDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise TheWellNotInstalledError() from exc

    kwargs = {"full_trajectory_mode": True}
    if n_steps is not None:
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        kwargs["max_rollout_steps"] = int(n_steps) - 1
    return WellDataset(
        well_base_path=resolve_path(base),
        well_dataset_name=name,
        well_split_name=split,
        **kwargs,
    )


def _flatten_trajectory(
    fields: np.ndarray,
    spatial_axes: dict[str, np.ndarray],
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Flatten ``(n_steps, *spatial, n_fields)`` into ``(N, ...)`` arrays.

    ``spatial_axes`` is an ordered dict mapping axis name -> 1D coordinate
    array (length matches the corresponding spatial dim of ``fields``).
    """
    grids = np.meshgrid(times, *spatial_axes.values(), indexing="ij")
    t_flat = grids[0].reshape(-1, 1).astype(np.float32)
    x_flat = np.stack([g.reshape(-1) for g in grids[1:]], axis=-1).astype(np.float32)
    u_flat = fields.reshape(-1, fields.shape[-1]).astype(np.float32)
    return {"x": x_flat, "t": t_flat, "u": u_flat}


def _extract_arrays(sample: Any, n_spatial_dims: int) -> dict[str, np.ndarray]:
    """Turn one full-trajectory ``WellDataset`` sample into numpy arrays.

    Fields are ``input_fields`` followed by ``output_fields`` along time
    (``(n_steps, *spatial, n_fields)``), times are ``input_time_grid`` followed
    by ``output_time_grid``, and the coordinate of spatial axis ``d`` is read
    from ``space_grid[..., d]`` along that axis.

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

    steps = [_as_np(sample["input_fields"])]
    times = [_as_np(sample["input_time_grid"]).reshape(-1)]
    if "output_fields" in sample:
        steps.append(_as_np(sample["output_fields"]))
        times.append(_as_np(sample["output_time_grid"]).reshape(-1))
    fields = np.concatenate(steps, axis=0).astype(np.float32)
    times = np.concatenate(times).astype(np.float32)
    if fields.ndim != n_spatial_dims + 2:
        raise ValueError(
            f"Expected fields with {n_spatial_dims + 2} dims "
            f"(n_steps, {'x ' * n_spatial_dims}n_fields), got shape {fields.shape}"
        )
    if times.size != fields.shape[0]:
        raise ValueError(f"{times.size} time stamps for {fields.shape[0]} field steps")

    space_grid = _as_np(sample["space_grid"])
    if space_grid.shape != (*fields.shape[1:-1], n_spatial_dims):
        raise ValueError(
            f"space_grid shape {space_grid.shape} does not match fields {fields.shape}"
        )
    spatial_axes: dict[str, np.ndarray] = {}
    for d, axis in enumerate(("x", "y", "z")[:n_spatial_dims]):
        index = tuple(slice(None) if i == d else 0 for i in range(n_spatial_dims))
        spatial_axes[axis] = space_grid[index + (d,)].astype(np.float32)

    return {
        "fields": fields,
        "times": times,
        **{f"axis_{k}": v for k, v in spatial_axes.items()},
    }


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
        n_steps: Cap on the number of time steps read per trajectory (memory
            grows with it); ``None`` keeps ``the_well``'s default of at most 101
            (the first step plus a 100-step rollout).

    Returns:
        ``{"x": (N, d_space), "t": (N, 1), "u": (N, d_fields)}`` on ``device``.

    Raises:
        KeyError: If ``name`` is not registered.
        TheWellNotInstalledError: If the optional ``the_well`` package is missing.
        RuntimeError: If the split is empty.
    """
    entry = get_entry(name)
    rng = np.random.default_rng(seed)

    cache_path = _cache_file(name, split, n_traj, n_points, seed, base=base, n_steps=n_steps)
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

    parts = []
    for idx in traj_indices:
        sample = dataset[int(idx)]
        arrays = _extract_arrays(sample, entry.n_spatial_dims)
        spatial = {k.removeprefix("axis_"): v for k, v in arrays.items() if k.startswith("axis_")}
        parts.append(_flatten_trajectory(arrays["fields"], spatial, arrays["times"]))

    x = np.concatenate([p["x"] for p in parts], axis=0)
    t = np.concatenate([p["t"] for p in parts], axis=0)
    u = np.concatenate([p["u"] for p in parts], axis=0)

    n_total = x.shape[0]
    if n_points < n_total:
        sel = rng.choice(n_total, size=n_points, replace=False)
        x, t, u = x[sel], t[sel], u[sel]

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, x=x, t=t, u=u)
        _LOGGER.info("Cached Well slice at %s", cache_path)

    return {
        "x": torch.tensor(x, device=device),
        "t": torch.tensor(t, device=device),
        "u": torch.tensor(u, device=device),
    }
