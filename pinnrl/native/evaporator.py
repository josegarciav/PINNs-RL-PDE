"""The falling-film evaporator's reference model integrated in C++ (``evaporator.cpp``, I16).

:func:`evaporator_solve` runs the method-of-lines model of
:class:`~pinnrl.process.FallingFilmEvaporator` with an L-stable fourth-order SDIRK method whose
implicit stages are solved exactly by one downstream sweep (the upwind Jacobian is block
lower-bidiagonal), so a solve costs microseconds per cell per step instead of SciPy's dense
factorisations. The inputs are piecewise-linear tables; the output is the state at requested times.
"""

from __future__ import annotations

import ctypes

import numpy as np


def evaporator_available() -> bool:
    """Whether the compiled evaporator integrator can be used here (compiles it on the first call)."""
    from pinnrl.native import _load_library

    return _load_library("evaporator") is not None


def evaporator_unavailable_reason() -> str | None:
    """Why :func:`evaporator_available` is false, or ``None``."""
    from pinnrl.native import _ERRORS, _load_library

    _load_library("evaporator")
    return _ERRORS.get("evaporator")


def evaporator_solve(
    nz: int,
    geometry: tuple[float, float, float, float, float, float, float],
    signal_times: np.ndarray,
    signal_values: np.ndarray,
    s0: np.ndarray,
    t_out: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-11,
    h0: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """States ``(nt, 2 nz)`` at ``t_out`` from ``s0`` at ``t_out[0]``.

    Args:
        nz: Grid points including the inlet.
        geometry: ``(length, velocity, perimeter, U, latent_heat, T_sat, k_bpe)``.
        signal_times: ``(n,)`` increasing times of the input tables.
        signal_values: ``(n, 4)`` columns ``W_in``, ``X_in``, ``T_s``, ``R_f`` at those times,
            interpolated linearly in between and held constant outside.
        s0: ``(2 nz,)`` state ``[W_0..W_{nz-1}, X_0..X_{nz-1}]`` (the inlet entries are ignored).
        t_out: ``(nt,)`` increasing output times; the first is the initial time.
        rtol: Relative tolerance of the step-size control (RMS norm, as SciPy's).
        atol: Absolute tolerance.
        h0: Initial step, or 0 for a thousandth of the span.

    Returns:
        The states and ``{"steps", "rejected", "newton"}``.

    Raises:
        RuntimeError: When the integrator is not available or the integration fails.
    """
    from pinnrl.native import _load_library

    lib = _load_library("evaporator")
    if lib is None:
        raise RuntimeError(evaporator_unavailable_reason())
    st = np.ascontiguousarray(signal_times, dtype=np.float64).ravel()
    sv = np.ascontiguousarray(signal_values, dtype=np.float64).reshape(st.size, 4)
    y0 = np.ascontiguousarray(s0, dtype=np.float64).ravel()
    tt = np.ascontiguousarray(t_out, dtype=np.float64).ravel()
    if y0.size != 2 * nz:
        raise ValueError(f"s0 must have 2 nz = {2 * nz} entries, got {y0.size}")
    geom = np.asarray(geometry, dtype=np.float64)
    out = np.empty((tt.size, 2 * nz))
    stats = (ctypes.c_long * 3)()
    ptr = ctypes.c_void_p
    status = lib.evap_solve(
        int(nz), ptr(geom.ctypes.data), int(st.size), ptr(st.ctypes.data), ptr(sv.ctypes.data),
        ptr(y0.ctypes.data), int(tt.size), ptr(tt.ctypes.data), ctypes.c_double(rtol),
        ctypes.c_double(atol), ctypes.c_double(h0), ptr(out.ctypes.data), stats,
    )  # fmt: skip
    info = {"steps": int(stats[0]), "rejected": int(stats[1]), "newton": int(stats[2])}
    if status != 0:
        what = (
            "the step size underflowed" if status == 1 else "the stage iteration did not converge"
        )
        raise RuntimeError(f"evaporator integration failed: {what} ({info})")
    return out, info


__all__ = ["evaporator_available", "evaporator_solve", "evaporator_unavailable_reason"]
