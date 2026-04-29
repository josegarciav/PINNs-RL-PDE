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


def _install_fake_well_module(monkeypatch, n_traj=2, n_steps=3, hw=4, n_fields=2):
    """Install a fake ``the_well.data`` module exposing a tiny ``WellDataset``."""

    class FakeWellDataset:
        def __init__(self, well_base_path, well_dataset_name, well_split_name):
            self.path = well_base_path
            self.name = well_dataset_name
            self.split = well_split_name
            rng = np.random.default_rng(0)
            self._fields = rng.standard_normal(
                (n_traj, n_steps, hw, hw, n_fields)
            ).astype(np.float32)

        def __len__(self):
            return n_traj

        def __getitem__(self, idx):
            return {
                "input_fields": torch.from_numpy(self._fields[idx]),
                "time": np.linspace(0.0, 1.0, n_steps, dtype=np.float32),
                "space/x": np.linspace(0.0, 1.0, hw, dtype=np.float32),
                "space/y": np.linspace(0.0, 1.0, hw, dtype=np.float32),
            }

    fake_data = types.ModuleType("the_well.data")
    fake_data.WellDataset = FakeWellDataset
    fake_root = types.ModuleType("the_well")
    fake_root.data = fake_data
    monkeypatch.setitem(sys.modules, "the_well", fake_root)
    monkeypatch.setitem(sys.modules, "the_well.data", fake_data)


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

    first = load_well_slice(
        "active_matter", n_traj=1, n_points=4, seed=42, device="cpu"
    )

    # Now break the loader: any further call would crash if it bypassed the cache.
    monkeypatch.setitem(sys.modules, "the_well", None)
    second = load_well_slice(
        "active_matter", n_traj=1, n_points=4, seed=42, device="cpu"
    )
    assert torch.allclose(first["x"], second["x"])
    assert torch.allclose(first["t"], second["t"])
    assert torch.allclose(first["u"], second["u"])


def test_load_well_slice_unknown_dataset_raises_before_dependency(monkeypatch):
    # Even without the_well installed, an unknown dataset should fail fast
    # at the registry layer rather than at import time.
    with pytest.raises(KeyError):
        load_well_slice("totally_made_up", n_traj=1, n_points=4, seed=0, device="cpu")
