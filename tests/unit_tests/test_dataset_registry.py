"""Smoke tests for the Well dataset registry."""

import os

import pytest

from pinnrl.datasets import WELL_REGISTRY, get_entry, list_dataset_names
from pinnrl.datasets.registry import DEFAULT_LOADED_STEPS, DEFAULT_LOADED_STEPS_3D
from pinnrl.pdes.registry import PDE_SPECS


def test_registry_is_non_empty():
    assert len(WELL_REGISTRY) > 0


def test_list_dataset_names_returns_sorted_unique_keys():
    names = list_dataset_names()
    assert names == sorted(set(WELL_REGISTRY))


def test_get_entry_unknown_raises_keyerror_with_helpful_message():
    with pytest.raises(KeyError) as exc:
        get_entry("not_a_dataset")
    assert "not_a_dataset" in str(exc.value)
    # The error names a valid dataset so the user has a starting point.
    sample = next(iter(WELL_REGISTRY))
    assert sample in str(exc.value)


@pytest.mark.parametrize("name", list(WELL_REGISTRY))
def test_entry_invariants(name):
    entry = WELL_REGISTRY[name]
    assert entry.name == name
    assert entry.n_spatial_dims in (2, 3)
    assert len(entry.domain) == entry.n_spatial_dims
    assert entry.default_input_dim == entry.n_spatial_dims + 1
    assert entry.default_output_dim == len(entry.fields)
    assert entry.recommended_mode in ("data_only", "data_augmented")


@pytest.mark.parametrize("name", list(WELL_REGISTRY))
def test_matched_pde_keys_resolve_in_pde_registry(name):
    entry = WELL_REGISTRY[name]
    if entry.default_pde_key is None:
        # Unmatched datasets must use ``data_only``.
        assert entry.recommended_mode == "data_only"
        return
    valid_keys = {spec.key for spec in PDE_SPECS}
    assert entry.default_pde_key in valid_keys, (
        f"{name}.default_pde_key={entry.default_pde_key!r} not in PDE_SPECS"
    )
    # Matched entries should default to data_augmented.
    assert entry.recommended_mode == "data_augmented"


@pytest.mark.parametrize("name", list(WELL_REGISTRY))
def test_entry_grid_and_time_invariants(name):
    entry = WELL_REGISTRY[name]
    assert len(entry.fields) > 0 and len(set(entry.fields)) == len(entry.fields)
    assert all(lo < hi for lo, hi in entry.domain)
    assert entry.dt > 0 and entry.n_steps >= 2
    assert entry.time_domain == (0.0, entry.dt * (entry.n_steps - 1))
    assert 1 <= entry.default_loaded_steps <= DEFAULT_LOADED_STEPS
    for n in (1, 2, 7, entry.n_steps, entry.n_steps + 5, None):
        steps = min(entry.default_loaded_steps if n is None else n, entry.n_steps)
        assert entry.loaded_steps(n) == steps
        assert entry.loaded_time_domain(n) == pytest.approx((0.0, entry.dt * (steps - 1)))


def test_3d_entries_read_a_short_window_by_default():
    for entry in WELL_REGISTRY.values():
        expected = DEFAULT_LOADED_STEPS_3D if entry.n_spatial_dims == 3 else DEFAULT_LOADED_STEPS
        assert entry.default_loaded_steps == expected, entry.name


def test_loaded_steps_rejects_nonpositive_counts():
    with pytest.raises(ValueError, match="at least 1"):
        WELL_REGISTRY["active_matter"].loaded_steps(0)


@pytest.mark.skipif(
    not os.environ.get("PINNRL_NETWORK_TESTS"),
    reason="reads every dataset from the Hugging Face mirror; set PINNRL_NETWORK_TESTS=1",
)
@pytest.mark.parametrize("name", list(WELL_REGISTRY))
def test_registry_matches_the_hugging_face_files(name):
    """Network: the entry agrees with the dataset's metadata and its first two steps."""
    import numpy as np

    from pinnrl.datasets.well_loader import _load_well_dataset

    entry = WELL_REGISTRY[name]
    dataset = _load_well_dataset(name, "train", None, n_steps=2)
    meta = dataset.metadata
    assert meta.n_spatial_dims == entry.n_spatial_dims
    assert meta.n_fields == len(entry.fields)
    assert meta.n_steps_per_trajectory[0] == entry.n_steps
    sample = dataset[0]
    grid = sample["space_grid"].numpy().reshape(-1, entry.n_spatial_dims)
    assert np.allclose(grid.min(axis=0), [lo for lo, _ in entry.domain], rtol=1e-4, atol=1e-6)
    assert np.allclose(grid.max(axis=0), [hi for _, hi in entry.domain], rtol=1e-4, atol=1e-6)
    times = np.concatenate([sample["input_time_grid"], sample["output_time_grid"]])
    assert np.diff(times)[0] == pytest.approx(entry.dt, rel=1e-4)


def test_loaded_window_follows_the_step_cap():
    entry = WELL_REGISTRY["gray_scott_reaction_diffusion"]  # dt = 10, 1001 steps
    assert entry.loaded_time_domain() == (0.0, 1000.0)  # the_well default: 101 steps
    assert entry.loaded_time_domain(3) == (0.0, 20.0)
    assert entry.loaded_time_domain(10_000) == entry.time_domain


def test_dataset_defaults_set_the_pde_window():
    import yaml

    from pinnrl.config import DEFAULT_CONFIG_PATH
    from pinnrl.training.train import build_config_dict

    with open(DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    dataset = {"name": "rayleigh_benard", "n_steps": 5}
    config = build_config_dict(raw, "heat", "fno", dataset=dataset)
    assert config["pde"]["domain"] == [[0.0, 4.0], [0.0, 1.0]]
    assert config["pde"]["time_domain"] == [0.0, 1.0]  # 4 steps of dt = 0.25
    assert config["pde"]["observation_data"]["n_steps"] == 5


def test_3d_dataset_window_matches_what_the_loader_reads():
    import yaml

    from pinnrl.config import DEFAULT_CONFIG_PATH
    from pinnrl.training.train import build_config_dict

    with open(DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    config = build_config_dict(raw, "heat", "feedforward", dataset={"name": "MHD_64"})
    assert config["pde"]["time_domain"] == pytest.approx(
        [0.0, 0.01 * (DEFAULT_LOADED_STEPS_3D - 1)]
    )
