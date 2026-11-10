"""An optional compiled kernel for the derivatives a PINN residual needs.

A residual such as ``u_t - alpha u_xx`` is usually built by reverse-mode autograd applied
twice, and the weight gradient by a third pass through that graph. For a plain ``tanh``
multilayer perceptron the same numbers come from one forward pass that carries the first and
second derivatives next to the value (Taylor mode) and one hand-written reverse pass.
:func:`mlp_jets` does that in a small C++ kernel (``jets.cpp``, compiled on first use and
cached), wrapped as a :class:`torch.autograd.Function` so the rest of the loss stays ordinary
PyTorch. Experiment I11 measures the step at roughly half the autograd time on Apple silicon.

It is opt-in (``pde.build_trainer(..., derivatives="native")``) and narrow by design:

* macOS with the Accelerate framework and a C++ compiler (``clang++``);
* a feed-forward network of ``Linear`` layers and ``tanh`` only, float32, on the CPU;
* first derivatives and pure second derivatives (``u_x``, ``u_t``, ``u_xx``; not ``u_xt`` or
  ``u_xxx``);
* gradients with respect to the weights only, once (no Hessian-vector products).

Anything else falls back to autograd; :func:`available` says whether the kernel can be used here.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("jets.cpp")
_LIB: ctypes.CDLL | None = None
_FAILED: str | None = None


def _cache_dir() -> Path:
    root = os.environ.get("PINNRL_NATIVE_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "pinnrl" / "native"


def _load() -> ctypes.CDLL | None:
    """Compile (once per source version) and load the kernel; ``None`` when that is not possible."""
    global _LIB, _FAILED
    if _LIB is not None or _FAILED is not None:
        return _LIB
    if sys.platform != "darwin":
        _FAILED = "the native kernel needs macOS (Accelerate)"
        return None
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        _FAILED = "no C++ compiler found (install the Xcode command line tools)"
        return None
    source = _SOURCE.read_bytes()
    tag = hashlib.sha256(source).hexdigest()[:16]
    lib_path = _cache_dir() / f"jets-{tag}.dylib"
    if not lib_path.exists():
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib_path.with_suffix(f".{os.getpid()}.tmp")
        cmd = [
            compiler, "-O3", "-mcpu=native", "-std=c++17", "-shared", "-fPIC",
            "-DACCELERATE_NEW_LAPACK", str(_SOURCE), "-framework", "Accelerate", "-o", str(tmp),
        ]  # fmt: skip
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _FAILED = f"compiling the native kernel failed: {result.stderr.strip()[-400:]}"
            logger.warning(_FAILED)
            return None
        os.replace(tmp, lib_path)
    lib = ctypes.CDLL(str(lib_path))
    lib.pinn_jets_work_size.restype = ctypes.c_long
    lib.pinn_jets_forward.restype = None
    lib.pinn_jets_backward.restype = None
    _LIB = lib
    return lib


def available() -> bool:
    """Whether the compiled kernel can be used on this machine (compiles it on the first call)."""
    return _load() is not None


def unavailable_reason() -> str | None:
    """Why :func:`available` is false, or ``None`` when the kernel is usable."""
    _load()
    return _FAILED


def _int_array(values: Sequence[int]):
    return (ctypes.c_int * max(len(values), 1))(*values)


def _pointer_array(tensors: Sequence[torch.Tensor]):
    return (ctypes.c_void_p * len(tensors))(*[t.data_ptr() for t in tensors])


# Work buffers are tens of megabytes a step; fresh ones cost page faults every time. A buffer is
# taken from this pool in the forward pass and handed back at the end of the backward pass.
_POOL: dict[int, list[torch.Tensor]] = {}
_POOL_LIMIT = 4  # buffers kept per size


def _take(size: int) -> torch.Tensor:
    stack = _POOL.get(size)
    return stack.pop() if stack else torch.empty(size)


def _give(buffer: torch.Tensor) -> None:
    stack = _POOL.setdefault(buffer.numel(), [])
    if len(stack) < _POOL_LIMIT:
        stack.append(buffer)


class _Jets(torch.autograd.Function):
    """``(S N, k)`` output jets of the MLP with weights ``params = (W0, b0, W1, b1, ...)``."""

    @staticmethod
    def forward(ctx, x, first, parent, *params):  # noqa: D102
        lib = _load()
        assert lib is not None
        weights = [p.detach().contiguous() for p in params[0::2]]
        biases = [p.detach().contiguous() for p in params[1::2]]
        n, n_layers = x.shape[0], len(weights)
        sizes = [x.shape[1]] + [w.shape[0] for w in weights]
        streams = 1 + len(first) + len(parent)
        c_sizes = _int_array(sizes)
        work = _take(lib.pinn_jets_work_size(n, n_layers, c_sizes, streams))
        out = torch.empty(streams * n, sizes[-1])
        x = x.detach().contiguous()
        lib.pinn_jets_forward(
            ctypes.c_void_p(x.data_ptr()), n, n_layers, c_sizes, _pointer_array(weights),
            _pointer_array(biases), len(first), _int_array(first), len(parent), _int_array(parent),
            ctypes.c_void_p(work.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        )  # fmt: skip
        ctx.jet_state = (work, weights, sizes, n, tuple(first), tuple(parent))
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):  # noqa: D102
        lib = _load()
        assert lib is not None
        work, weights, sizes, n, first, parent = ctx.jet_state
        streams = 1 + len(first) + len(parent)
        grad_w = [torch.empty_like(w) for w in weights]
        grad_b = [torch.empty(w.shape[0]) for w in weights]
        scratch = _take(2 * streams * n * max(sizes))
        grad_out = grad_out.contiguous()
        lib.pinn_jets_backward(
            n, len(weights), _int_array(sizes), _pointer_array(weights), len(first), len(parent),
            _int_array(parent), ctypes.c_void_p(work.data_ptr()),
            ctypes.c_void_p(grad_out.data_ptr()), _pointer_array(grad_w), _pointer_array(grad_b),
            ctypes.c_void_p(scratch.data_ptr()),
        )  # fmt: skip
        _give(scratch)
        _give(work)
        ctx.jet_state = None
        grads: list[torch.Tensor | None] = [None, None, None]
        for gw, gb in zip(grad_w, grad_b, strict=True):
            grads += [gw, gb]
        return tuple(grads)


def supported_layers(layers: Sequence[torch.nn.Module]) -> list[torch.nn.Linear] | None:
    """The ``Linear`` layers of a ``Linear, Tanh, ..., Linear`` stack, or ``None`` for anything else.

    Dropout with probability zero and identity modules are ignored.
    """
    linears: list[torch.nn.Linear] = []
    expect_linear = True
    for layer in layers:
        if isinstance(layer, torch.nn.Identity) or (
            isinstance(layer, torch.nn.Dropout) and layer.p == 0.0
        ):
            continue
        if expect_linear and isinstance(layer, torch.nn.Linear) and layer.bias is not None:
            linears.append(layer)
        elif not expect_linear and isinstance(layer, torch.nn.Tanh):
            pass
        else:
            return None
        expect_linear = not expect_linear
    if not linears or expect_linear:  # must end on a Linear
        return None
    return linears


def mlp_jets(
    linears: Sequence[torch.nn.Linear],
    x: torch.Tensor,
    first: Sequence[int] = (),
    second: Sequence[int] = (),
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Value and derivatives of a tanh MLP at ``x`` through the compiled kernel.

    Args:
        linears: The ``Linear`` layers in order (``tanh`` between them).
        x: Inputs, ``(N, d)``, float32 on the CPU; treated as data (no gradient flows to it).
        first: Input columns to differentiate once.
        second: Input columns to differentiate twice (each is added to ``first`` if missing).

    Returns:
        ``(u, du, d2u)``: the output ``(N, k)``, ``{column: du/dx_column}`` and
        ``{column: d2u/dx_column^2}``, all differentiable with respect to the weights.

    Raises:
        RuntimeError: When the kernel is not available (see :func:`unavailable_reason`) or the
            input is not float32 on the CPU.
    """
    if not available():
        raise RuntimeError(unavailable_reason())
    if x.dtype != torch.float32 or x.device.type != "cpu":
        raise RuntimeError("the native kernel needs float32 inputs on the CPU")
    first_cols = list(dict.fromkeys([*first, *second]))
    parent = [first_cols.index(c) for c in second]
    params: list[torch.Tensor] = []
    for lin in linears:
        params += [lin.weight, lin.bias]
    out = _Jets.apply(x, tuple(first_cols), tuple(parent), *params)
    n = x.shape[0]
    u = out[:n]
    du = {c: out[(1 + i) * n : (2 + i) * n] for i, c in enumerate(first_cols)}
    d2u = {
        c: out[(1 + len(first_cols) + j) * n : (2 + len(first_cols) + j) * n]
        for j, c in enumerate(second)
    }
    return u, du, d2u


__all__ = ["available", "mlp_jets", "supported_layers", "unavailable_reason"]
