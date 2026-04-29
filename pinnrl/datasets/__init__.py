"""Dataset adapters for pinnrl.

Currently exposes loaders for `The Well` (Polymathic AI) — see
:mod:`pinnrl.datasets.well_loader` for the entry point. The optional
``the_well`` dependency is imported lazily so that simply listing
available datasets in the dashboard works without a heavy install.
"""

from .registry import (
    WELL_REGISTRY,
    WellEntry,
    get_entry,
    list_dataset_names,
)
from .well_loader import (
    TheWellNotInstalledError,
    load_well_slice,
    resolve_path,
)

__all__ = [
    "WELL_REGISTRY",
    "WellEntry",
    "get_entry",
    "list_dataset_names",
    "TheWellNotInstalledError",
    "load_well_slice",
    "resolve_path",
]
