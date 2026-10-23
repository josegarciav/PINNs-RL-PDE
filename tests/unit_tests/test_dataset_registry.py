"""Smoke tests for the Well dataset registry."""

import pytest

from pinnrl.datasets import WELL_REGISTRY, get_entry, list_dataset_names
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


# Measured from each dataset's first training file on the HF mirror
# (dimensions/time, dimensions/<axis>, t{0,1,2}_fields); see registry.py.
@pytest.mark.parametrize(
    "name, domain, t_end, channels",
    [
        ("active_matter", ((0.0, 10.0), (0.0, 10.0)), 20.0, 11),
        ("rayleigh_benard", ((0.0, 4.0), (0.0, 1.0)), 49.75, 4),
        ("gray_scott_reaction_diffusion", ((-1.0, 1.0), (-1.0, 1.0)), 10000.0, 2),
        ("rayleigh_taylor_instability", ((0.0, 1.0),) * 3, 118.004, 4),
        ("viscoelastic_instability", ((0.0, 6.2831855), (-1.0, 1.0)), 19.0, 8),
    ],
)
def test_registry_matches_the_dataset_files(name, domain, t_end, channels):
    entry = WELL_REGISTRY[name]
    assert entry.domain == domain
    assert entry.time_domain[1] == pytest.approx(t_end, rel=1e-4)
    assert entry.default_output_dim == channels


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
