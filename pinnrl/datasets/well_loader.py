"""Load `The Well` datasets and flatten them into pinnrl observation tensors.

The Well stores trajectories as ``(n_traj, n_steps, *spatial, n_fields)`` HDF5
arrays. pinnrl's data-fitting loss expects a flat point cloud of
``(x, t, u)`` tuples (see ``PDEBase._compute_data_loss``). The functions here
bridge those two shapes:

* :func:`resolve_path` picks the Hugging Face mirror by default and falls
  back to a user-supplied local download dir.
* :func:`load_well_slice` instantiates ``the_well.data.WellDataset`` (lazily —
  the dependency is optional), pulls a small slice of trajectories, samples
  ``n_points`` random spatio-temporal locations, and returns ready-to-use
  CPU/GPU tensors.

Loaded slices are cached as ``.npz`` under ``~/.cache/pinnrl/well/`` so
repeat training runs (and CI smoke tests) do not re-download or re-flatten.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

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


def resolve_path(base: Optional[str] = None) -> str:
    """Return the ``well_base_path`` to hand to ``WellDataset``.

    Args:
        base: Local directory if the user has run ``the-well-download``,
            otherwise ``None`` to stream from Hugging Face.
    """
    if base is None or not str(base).strip():
        return _DEFAULT_HF_BASE
    return str(base)


def _cache_dir() -> Path:
    """Where flattened ``.npz`` slices are stored."""
    root = os.environ.get("PINNRL_WELL_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "pinnrl" / "well"


def _cache_file(name: str, split: str, n_traj: int, n_points: int, seed: int) -> Path:
    return _cache_dir() / f"{name}__{split}__t{n_traj}_p{n_points}_s{seed}.npz"


def _load_well_dataset(name: str, split: str, base: Optional[str]):
    """Import ``the_well`` lazily and instantiate a ``WellDataset``."""
    try:
        from the_well.data import WellDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise TheWellNotInstalledError() from exc

    return WellDataset(
        well_base_path=resolve_path(base),
        well_dataset_name=name,
        well_split_name=split,
    )


def _flatten_trajectory(
    fields: np.ndarray,
    spatial_axes: Dict[str, np.ndarray],
    times: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Flatten ``(n_steps, *spatial, n_fields)`` into ``(N, ...)`` arrays.

    ``spatial_axes`` is an ordered dict mapping axis name -> 1D coordinate
    array (length matches the corresponding spatial dim of ``fields``).
    """
    grids = np.meshgrid(times, *spatial_axes.values(), indexing="ij")
    t_flat = grids[0].reshape(-1, 1).astype(np.float32)
    x_flat = np.stack([g.reshape(-1) for g in grids[1:]], axis=-1).astype(np.float32)
    u_flat = fields.reshape(-1, fields.shape[-1]).astype(np.float32)
    return {"x": x_flat, "t": t_flat, "u": u_flat}


def _extract_arrays(sample: Any, n_spatial_dims: int) -> Dict[str, np.ndarray]:
    """Coerce a ``WellDataset`` sample into raw numpy arrays.

    The Well's sample dict format has shifted between releases; we accept
    the common shapes and surface a clear error otherwise so users know
    what we needed.
    """
    if not isinstance(sample, dict):
        raise TypeError(
            f"Expected WellDataset sample to be a dict, got {type(sample).__name__}"
        )

    def _as_np(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    fields_key = next(
        (k for k in ("input_fields", "fields", "u", "data") if k in sample), None
    )
    if fields_key is None:
        raise KeyError(
            "WellDataset sample missing a fields tensor. Looked for "
            "'input_fields', 'fields', 'u', or 'data'; got keys: "
            f"{list(sample.keys())}"
        )
    fields = _as_np(sample[fields_key]).astype(np.float32)
    if fields.ndim != n_spatial_dims + 2:
        raise ValueError(
            f"Expected fields tensor with {n_spatial_dims + 2} dims "
            f"(n_steps, {'x ' * n_spatial_dims}n_fields), got shape {fields.shape}"
        )

    n_steps = fields.shape[0]
    times = _as_np(sample.get("time", np.linspace(0.0, 1.0, n_steps))).reshape(-1)
    if times.size != n_steps:
        times = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)

    spatial_axes: Dict[str, np.ndarray] = {}
    axis_names = ("x", "y", "z")[:n_spatial_dims]
    for i, axis in enumerate(axis_names):
        size = fields.shape[1 + i]
        coord = _as_np(sample.get(f"space/{axis}", np.linspace(0.0, 1.0, size))).reshape(-1)
        if coord.size != size:
            coord = np.linspace(0.0, 1.0, size, dtype=np.float32)
        spatial_axes[axis] = coord.astype(np.float32)

    return {"fields": fields, "times": times.astype(np.float32), **{f"axis_{k}": v for k, v in spatial_axes.items()}}


def load_well_slice(
    name: str,
    split: str = "train",
    n_traj: int = 1,
    n_points: int = 4096,
    seed: int = 0,
    device: str = "cpu",
    base: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, torch.Tensor]:
    """Sample ``n_points`` flat ``(x, t, u)`` tuples from a Well dataset.

    Args:
        name: Registry key — must exist in ``WELL_REGISTRY``.
        split: Dataset split (``train``, ``valid``, ``test``).
        n_traj: How many trajectories to draw from. Tensors are aggregated
            across them before sub-sampling.
        n_points: Number of (space, time) samples to retain.
        seed: RNG seed for reproducible sub-sampling.
        device: Torch device for the returned tensors.
        base: Local download dir; ``None`` = HF streaming.
        use_cache: When True, reuse / write a flattened ``.npz`` under
            ``~/.cache/pinnrl/well/``. Set False to force a fresh load.

    Returns:
        ``{"x": (N, d_space), "t": (N, 1), "u": (N, d_fields)}`` on ``device``.
    """
    entry = get_entry(name)
    rng = np.random.default_rng(seed)

    cache_path = _cache_file(name, split, n_traj, n_points, seed)
    if use_cache and cache_path.exists():
        _LOGGER.info("Loading cached Well slice from %s", cache_path)
        with np.load(cache_path) as data:
            return {
                "x": torch.tensor(data["x"], device=device),
                "t": torch.tensor(data["t"], device=device),
                "u": torch.tensor(data["u"], device=device),
            }

    dataset = _load_well_dataset(name, split, base)
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
