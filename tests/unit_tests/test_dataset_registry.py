"""Smoke tests for the Well dataset registry."""

import pytest

from pinnrl.datasets import WELL_REGISTRY, get_entry, list_dataset_names
from pinnrl.training.train import PDE_REGISTRY


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
    valid_keys = {meta[2] for meta in PDE_REGISTRY.values()}
    assert entry.default_pde_key in valid_keys, (
        f"{name}.default_pde_key={entry.default_pde_key!r} not in PDE_REGISTRY"
    )
    # Matched entries should default to data_augmented.
    assert entry.recommended_mode == "data_augmented"
