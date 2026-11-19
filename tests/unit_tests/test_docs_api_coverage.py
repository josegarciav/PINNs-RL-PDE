"""Every public name of pinnrl has an entry in the API reference (docs/api/)."""

import importlib
import pkgutil
import re
from pathlib import Path

import pinnrl

API_PAGES = Path(__file__).resolve().parents[2] / "docs" / "api"

# Public names that deliberately have no entry, each with its reason.
EXCLUDED = {
    "pinnrl.__version__",  # the version string, stated on the package's PyPI page
}


def _directives() -> set[str]:
    """The ``::: path`` targets of every API page, as mkdocstrings reads them (relative to pinnrl)."""
    found: set[str] = set()
    for page in API_PAGES.glob("*.md"):
        found.update(re.findall(r"^::: ([\w.]+)", page.read_text(), flags=re.MULTILINE))
    return found


def _public_names() -> list[tuple[str, str]]:
    """``(module, name)`` for every name in an ``__all__`` of the package."""
    modules = ["pinnrl"] + [m.name for m in pkgutil.walk_packages(pinnrl.__path__, "pinnrl.")]
    names = []
    for module_name in modules:
        module = importlib.import_module(module_name)
        names += [(module_name, name) for name in getattr(module, "__all__", ())]
    return names


def _paths(module_name: str, name: str) -> set[str]:
    """The paths a directive may use for a name: where it is exported and where it is defined."""
    paths = {f"{module_name}.{name}"}
    obj = getattr(importlib.import_module(module_name), name, None)
    home, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if isinstance(home, str) and isinstance(qualname, str) and home.startswith("pinnrl"):
        paths.add(f"{home}.{qualname}")
    for sub in pkgutil.iter_modules(getattr(importlib.import_module(module_name), "__path__", [])):
        submodule = importlib.import_module(f"{module_name}.{sub.name}")  # a constant's home
        if getattr(submodule, name, None) is obj and obj is not None:
            paths.add(f"{module_name}.{sub.name}.{name}")
    return {p.removeprefix("pinnrl.") for p in paths}


def test_every_public_name_has_an_api_entry():
    directives = _directives()
    assert directives, f"no ':::' directives found under {API_PAGES}"
    missing = []
    for module_name, name in _public_names():
        if f"{module_name}.{name}" in EXCLUDED:
            continue
        documented = any(
            path == d or path.startswith(d + ".")
            for path in _paths(module_name, name)
            for d in directives
        )
        if not documented:
            missing.append(f"{module_name}.{name}")
    assert not missing, (
        "public names without an entry in docs/api/ (add a ':::' directive, or list the name in "
        f"EXCLUDED with its reason): {sorted(set(missing))}"
    )


def test_every_api_directive_points_at_something_that_exists():
    broken = []
    for directive in sorted(_directives()):
        parts = directive.split(".")
        for cut in range(len(parts), 0, -1):  # the longest importable module, then attributes
            try:
                obj = importlib.import_module("pinnrl." + ".".join(parts[:cut]))
            except ImportError:
                continue
            try:
                for attribute in parts[cut:]:
                    obj = getattr(obj, attribute)
            except AttributeError:
                broken.append(directive)
            break
        else:
            broken.append(directive)
    assert not broken, f"':::' directives in docs/api/ that name nothing in pinnrl: {broken}"
