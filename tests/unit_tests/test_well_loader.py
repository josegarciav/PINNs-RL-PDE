"""Tests for the Well dataset loader.

We don't depend on ``the_well`` actually being installed: a fake module is
spliced into ``sys.modules`` for these tests so the loader exercises its
own flatten / cache logic against a small synthetic trajectory.
"""

import sys
import types

import numpy as np
import pytest
import torch

from pinnrl.datasets.well_loader import (
    TheWellNotInstalledError,
    load_well_slice,
    resolve_path,
)

# Fake trajectories live on x in [0, 4], y in [-1, 1] and t in [2, 2 + dt * n_steps)
# so the tests can tell real coordinates from placeholders.
X_RANGE, Y_RANGE, T0, DT = (0.0, 4.0), (-1.0, 1.0), 2.0, 0.5


def _install_fake_well_module(monkeypatch, n_traj=2, n_steps=3, hw=4, n_fields=2):
    """Install a fake ``the_well.data`` module mimicking ``WellDataset`` 1.x samples.

    In ``full_trajectory_mode`` the real class returns the first step in
    ``input_fields`` and the rest in ``output_fields``, with ``input_time_grid``
    / ``output_time_grid`` and a ``space_grid`` of shape ``(*spatial, n_dims)``.
    """
    calls = []

    class FakeWellDataset:
        def __init__(self, well_base_path, well_dataset_name, well_split_name, **kwargs):
            calls.append(kwargs)
            self.path = well_base_path
            self.name = well_dataset_name
            self.split = well_split_name
            # Like the real class: the first step plus up to max_rollout_steps more.
            steps = 1 + min(n_steps - 1, kwargs.get("max_rollout_steps", 100))
            rng = np.random.default_rng(0)
            self._fields = rng.standard_normal((n_traj, steps, hw, hw, n_fields)).astype(np.float32)
            self._times = (T0 + DT * np.arange(steps)).astype(np.float32)
            xs = np.linspace(*X_RANGE, hw, dtype=np.float32)
            ys = np.linspace(*Y_RANGE, hw, dtype=np.float32)
            self._grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)

        def __len__(self):
            return n_traj

        def __getitem__(self, idx):
            sample = {
                "input_fields": torch.from_numpy(self._fields[idx, :1]),
                "input_time_grid": torch.from_numpy(self._times[:1]),
                "space_grid": torch.from_numpy(self._grid),
            }
            if self._fields.shape[1] > 1:
                sample["output_fields"] = torch.from_numpy(self._fields[idx, 1:])
                sample["output_time_grid"] = torch.from_numpy(self._times[1:])
            return sample

    fake_data = types.ModuleType("the_well.data")
    fake_data.WellDataset = FakeWellDataset
    fake_root = types.ModuleType("the_well")
    fake_root.data = fake_data
    monkeypatch.setitem(sys.modules, "the_well", fake_root)
    monkeypatch.setitem(sys.modules, "the_well.data", fake_data)
    return calls


def test_resolve_path_defaults_to_huggingface():
    assert resolve_path() == "hf://datasets/polymathic-ai/"
    assert resolve_path("") == "hf://datasets/polymathic-ai/"


def test_resolve_path_passthrough_for_local_dir():
    assert resolve_path("/tmp/well") == "/tmp/well"


def test_load_well_slice_raises_when_dependency_missing(monkeypatch):
    # Pretend the dependency is absent — _load_well_dataset should re-raise.
    monkeypatch.setitem(sys.modules, "the_well", None)
    with pytest.raises(TheWellNotInstalledError):
        load_well_slice(
            "active_matter", n_traj=1, n_points=8, seed=0, device="cpu", use_cache=False
        )


def test_load_well_slice_returns_expected_shapes(monkeypatch, tmp_path):
    _install_fake_well_module(monkeypatch)
    monkeypatch.setenv("PINNRL_WELL_CACHE", str(tmp_path))

    out = load_well_slice(
        "active_matter",
        split="train",
        n_traj=2,
        n_points=8,
        seed=0,
        device="cpu",
        use_cache=False,
    )
    assert set(out) == {"x", "t", "u"}
    assert out["x"].shape == (8, 2)
    assert out["t"].shape == (8, 1)
    # active_matter has 5 default fields, fake stub has 2 — registry only
    # influences default model output_dim, not the loader's actual u shape.
    assert out["u"].shape == (8, 2)
    assert out["x"].dtype == torch.float32


def test_load_well_slice_round_trips_through_cache(monkeypatch, tmp_path):
    _install_fake_well_module(monkeypatch)
    monkeypatch.setenv("PINNRL_WELL_CACHE", str(tmp_path))

    first = load_well_slice("active_matter", n_traj=1, n_points=4, seed=42, device="cpu")

    # Now break the loader: any further call would crash if it bypassed the cache.
    monkeypatch.setitem(sys.modules, "the_well", None)
    second = load_well_slice("active_matter", n_traj=1, n_points=4, seed=42, device="cpu")
    assert torch.allclose(first["x"], second["x"])
    assert torch.allclose(first["t"], second["t"])
    assert torch.allclose(first["u"], second["u"])


def test_load_well_slice_unknown_dataset_raises_before_dependency(monkeypatch):
    # Even without the_well installed, an unknown dataset should fail fast
    # at the registry layer rather than at import time.
    with pytest.raises(KeyError):
        load_well_slice("totally_made_up", n_traj=1, n_points=4, seed=0, device="cpu")


def test_cache_key_partitions_by_base(monkeypatch, tmp_path):
    """Two different ``base`` paths must not share a cached ``.npz``.

    A bug here lets a run pointed at a stale local mirror reuse the slice
    that was flattened from the canonical HF dataset (or vice versa) and
    silently train on wrong data.
    """
    _install_fake_well_module(monkeypatch)
    monkeypatch.setenv("PINNRL_WELL_CACHE", str(tmp_path))

    hf_run = load_well_slice("active_matter", n_traj=1, n_points=4, seed=0, device="cpu", base=None)
    local_run = load_well_slice(
        "active_matter",
        n_traj=1,
        n_points=4,
        seed=0,
        device="cpu",
        base="/some/local/mirror",
    )

    # Two distinct cache files should have been written — one per base.
    cache_files = sorted(tmp_path.glob("active_matter__train__t1_p4_s0_*.npz"))
    assert len(cache_files) == 2, cache_files

    # Sanity: both runs returned populated tensors.
    assert hf_run["u"].shape == (4, 2)
    assert local_run["u"].shape == (4, 2)


def test_load_well_slice_uses_the_real_time_and_space_grids(monkeypatch, tmp_path):
    """Regression: placeholders put every point at t=0 on [0, 1]^2 whatever the data."""
    calls = _install_fake_well_module(monkeypatch, n_traj=1, n_steps=5, hw=6)
    out = load_well_slice("active_matter", n_traj=1, n_points=10_000, use_cache=False)
    t = out["t"].numpy().ravel()
    assert np.allclose(np.unique(t), T0 + DT * np.arange(5))
    assert np.isclose(out["x"][:, 0].min(), X_RANGE[0]) and np.isclose(
        out["x"][:, 0].max(), X_RANGE[1]
    )
    assert np.isclose(out["x"][:, 1].min(), Y_RANGE[0]) and np.isclose(
        out["x"][:, 1].max(), Y_RANGE[1]
    )
    assert calls[0]["full_trajectory_mode"] is True


def test_load_well_slice_keeps_x_t_u_aligned(monkeypatch):
    """Each u value is the field at its own (t, x, y) grid node."""
    _install_fake_well_module(monkeypatch, n_traj=1, n_steps=3, hw=4, n_fields=1)
    from the_well.data import WellDataset

    raw = WellDataset("p", "active_matter", "train", full_trajectory_mode=True)._fields[0]
    out = load_well_slice("active_matter", n_traj=1, n_points=10_000, use_cache=False)
    xs = np.linspace(*X_RANGE, 4)
    ys = np.linspace(*Y_RANGE, 4)
    for (x, y), t, u in zip(
        out["x"].numpy(), out["t"].numpy().ravel(), out["u"].numpy().ravel(), strict=True
    ):
        k, i, j = round((t - T0) / DT), int(np.argmin(abs(xs - x))), int(np.argmin(abs(ys - y)))
        assert np.isclose(raw[k, i, j, 0], u)


def test_n_steps_caps_the_trajectory_and_the_cache_key(monkeypatch, tmp_path):
    calls = _install_fake_well_module(monkeypatch, n_traj=1, n_steps=5, hw=4)
    monkeypatch.setenv("PINNRL_WELL_CACHE", str(tmp_path))
    out = load_well_slice("active_matter", n_traj=1, n_points=10_000, n_steps=2)
    assert calls[-1]["max_rollout_steps"] == 1
    assert np.allclose(np.unique(out["t"].numpy()), T0 + DT * np.arange(2))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 1 and "_n2__" in names[0] and names[0].endswith("__v2.npz")


def test_samples_without_grids_raise_a_clear_error():
    from pinnrl.datasets.well_loader import _extract_arrays

    with pytest.raises(KeyError, match="return_grid"):
        _extract_arrays({"input_fields": torch.zeros(1, 4, 4, 2)}, n_spatial_dims=2)
