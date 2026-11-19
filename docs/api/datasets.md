# Datasets

Loaders for the simulation datasets of [The Well](https://github.com/PolymathicAI/the_well) (`pinnrl.datasets`); see the [datasets guide](../datasets.md). The registry needs nothing beyond the package; loading a slice needs the `well` extra (`pip install 'pinnrl[well]'`).

## The registry

`WELL_REGISTRY` maps a dataset name to its entry: where it lives, its fields, its grid and the equation it comes from.

::: datasets.registry.WELL_REGISTRY

::: datasets.registry.WellEntry

::: datasets.registry.list_dataset_names

::: datasets.registry.get_entry

## Loading

::: datasets.well_loader.load_well_slice

::: datasets.well_loader.resolve_path

::: datasets.well_loader.TheWellNotInstalledError
