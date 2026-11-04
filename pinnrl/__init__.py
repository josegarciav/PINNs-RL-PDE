"""pinnrl: physics-informed neural networks for PDEs, with an experimental RL sampler."""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pinnrl")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

# Libraries log to their own logger and leave handler setup to the application.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["CustomPDE", "Fields", "TimeSeries", "UnknownFunction", "__version__"]


def __getattr__(name):
    """Lazy top-level access to the user-defined-equation API (keeps ``import pinnrl`` light)."""
    if name in ("CustomPDE", "Fields"):
        from pinnrl.pdes import custom

        return getattr(custom, name)
    if name == "UnknownFunction":
        from pinnrl.pdes.unknowns import UnknownFunction

        return UnknownFunction
    if name == "TimeSeries":
        from pinnrl.pdes.inputs import TimeSeries

        return TimeSeries
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
