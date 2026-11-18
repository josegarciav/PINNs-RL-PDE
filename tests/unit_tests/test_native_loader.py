"""The compiled kernels' loader: where it cannot build, it says why and falls back cleanly."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import pinnrl.native as native
from pinnrl.native import evaporator as ev


@pytest.fixture
def fresh_loader(monkeypatch, tmp_path):
    """No kernel loaded or failed yet, and a cache directory of this test's own."""
    monkeypatch.setattr(native, "_LOADED", {})
    monkeypatch.setattr(native, "_ERRORS", {})
    monkeypatch.setenv("PINNRL_NATIVE_CACHE", str(tmp_path))
    monkeypatch.setenv("CXX", "c++")
    return tmp_path


def test_build_command_has_no_build_elsewhere(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "win32")
    assert native._build_command("c++", Path("a.cpp"), Path("a.so"), blas=True) is None


def test_build_command_takes_extra_flags_from_the_environment(monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "linux")
    monkeypatch.setenv("PINNRL_NATIVE_CXXFLAGS", "-I/opt/blas/include")
    monkeypatch.setenv("PINNRL_NATIVE_LDFLAGS", "-L/opt/blas/lib")
    cmd = native._build_command("g++", Path("jets.cpp"), Path("jets.so"), blas=True)
    assert cmd is not None
    assert (
        "-I/opt/blas/include" in cmd
        and "-lopenblas" in cmd
        and cmd[-3:] == ["-L/opt/blas/lib", "-o", "jets.so"]
    )


def test_no_compiler_is_reported_once(fresh_loader, monkeypatch):
    monkeypatch.delenv("CXX")
    monkeypatch.setattr(native.shutil, "which", lambda name: None)
    assert native._load_library("evaporator") is None
    assert "no C++ compiler" in native._ERRORS["evaporator"]
    assert native._load_library("evaporator") is None  # the failure is remembered
    assert ev.evaporator_available() is False
    assert ev.evaporator_unavailable_reason() == native._ERRORS["evaporator"]
    assert not list(fresh_loader.iterdir())


def test_unsupported_platform_is_reported(fresh_loader, monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "win32")
    assert native._load_library("jets") is None
    assert "macOS" in native._ERRORS["jets"]
    assert native.available() is False
    assert native.unavailable_reason() == native._ERRORS["jets"]


def test_compile_failure_keeps_the_compiler_message(fresh_loader, monkeypatch, caplog):
    def failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fatal error: no such header")

    monkeypatch.setattr(native.subprocess, "run", failing)
    assert native._load_library("evaporator") is None
    reason = native._ERRORS["evaporator"]
    assert reason.startswith("compiling evaporator.cpp failed") and "no such header" in reason
    assert "OpenBLAS" not in reason  # the hint is for the BLAS kernel on Linux only
    assert reason in caplog.text
    with pytest.raises(RuntimeError, match="compiling evaporator.cpp failed"):
        ev.evaporator_solve(3, (1.0,) * 7, [0.0, 1.0], np.zeros((2, 4)), np.zeros(6), [0.0, 1.0])


def test_compile_failure_on_linux_points_at_openblas(fresh_loader, monkeypatch):
    monkeypatch.setattr(native.sys, "platform", "linux")
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="cblas.h missing"),
    )
    assert native._load_library("jets") is None
    assert "libopenblas-dev" in native._ERRORS["jets"]


@pytest.mark.skipif(not ev.evaporator_available(), reason="no compiled integrator here")
def test_evaporator_solve_checks_the_state_size():
    assert ev.evaporator_unavailable_reason() is None
    with pytest.raises(ValueError, match="2 nz = 6"):
        ev.evaporator_solve(3, (1.0,) * 7, [0.0, 1.0], np.zeros((2, 4)), np.zeros(5), [0.0, 1.0])
