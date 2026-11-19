"""An optional compiled kernel for the derivatives a PINN residual needs.

A residual such as ``u_t - alpha u_xx`` is usually built by reverse-mode autograd applied
twice, and the weight gradient by a third pass through that graph. For a plain ``tanh``
multilayer perceptron the same numbers come from one forward pass that carries the derivatives
next to the value as Taylor coefficients (Taylor mode) and one hand-written reverse pass.
:func:`mlp_jets` does that in a small C++ kernel (``jets.cpp``, compiled on first use and
cached), wrapped as a :class:`torch.autograd.Function` so the rest of the loss stays ordinary
PyTorch. Experiments I11–I14 measure the step at a half to a seventh of the autograd time.

The trainer uses it wherever it applies (``derivatives="auto"``, the default of
``pde.build_trainer`` and ``pinnrl-train``; ``"autograd"`` switches it off). It is narrow by design:

* macOS (the Accelerate framework, ``clang++``) or Linux (OpenBLAS, ``libopenblas-dev``, and ``g++``);
* a feed-forward network of ``Linear`` layers and ``tanh``, with or without ``LayerNorm``
  before each ``tanh``, float32, on the CPU;
* derivatives of any order with respect to the inputs, mixed ones included (``u_x``, ``u_xx``,
  ``u_xt``, ``u_xxx``, ``u_xxxx``), each as a tuple of input columns;
* gradients with respect to the weights only; a gradient that is itself differentiated
  (``create_graph=True``: Hessian-vector products) is taken by autograd on the same function.

Anything else falls back to autograd; :func:`available` says whether the kernel can be used here.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import logging
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("jets.cpp")
_LIB: ctypes.CDLL | None = None
_FAILED: str | None = None


def _cache_dir() -> Path:
    root = os.environ.get("PINNRL_NATIVE_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "pinnrl" / "native"


def _build_command(compiler: str, source: Path, out: Path, blas: bool) -> list[str] | None:
    """The compiler command for this platform, or ``None`` where the kernel has no build."""
    common = [
        "-O3", "-std=c++17", "-shared", "-fPIC",
        # sums may be reordered (so the reductions vectorise); NaN and infinity keep their meaning
        "-fassociative-math", "-fno-signed-zeros", "-fno-trapping-math", "-fno-math-errno",
        "-freciprocal-math",
        *os.environ.get("PINNRL_NATIVE_CXXFLAGS", "").split(),
    ]  # fmt: skip
    if sys.platform == "darwin":
        link = ["-framework", "Accelerate"] if blas else []
        arch = ["-mcpu=native", "-DACCELERATE_NEW_LAPACK"]
    elif sys.platform.startswith("linux"):
        link = ["-lopenblas"] if blas else []
        arch = ["-march=native"]
    else:
        return None
    link += os.environ.get("PINNRL_NATIVE_LDFLAGS", "").split()
    return [compiler, *common, *arch, str(source), *link, "-o", str(out)]


# name -> (main source, every file whose change rebuilds it, needs a BLAS)
_LIBRARIES: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "jets": ("jets.cpp", ("jets.cpp", "fast_tanh.h"), True),
    "evaporator": ("evaporator.cpp", ("evaporator.cpp",), False),
}
_LOADED: dict[str, ctypes.CDLL] = {}
_ERRORS: dict[str, str] = {}


def _compiler() -> str | None:
    return (
        os.environ.get("CXX")
        or shutil.which("clang++")
        or shutil.which("g++")
        or shutil.which("c++")
    )


def _load_library(name: str) -> ctypes.CDLL | None:
    """Compile (once per source version) and load one of the kernels; ``None`` when that fails."""
    if name in _LOADED:
        return _LOADED[name]
    if name in _ERRORS:
        return None
    main, sources, blas = _LIBRARIES[name]
    compiler = _compiler()
    if compiler is None:
        _ERRORS[name] = (
            "no C++ compiler found (install the Xcode command line tools on macOS, g++ on Linux)"
        )
        return None
    here = _SOURCE.parent
    digest = hashlib.sha256(b"flags-4")
    for f in sources:
        digest.update((here / f).read_bytes())
    lib_path = _cache_dir() / f"{name}-{digest.hexdigest()[:16]}.so"
    if not lib_path.exists():
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = lib_path.with_suffix(f".{os.getpid()}.tmp")
        cmd = _build_command(compiler, here / main, tmp, blas)
        if cmd is None:
            _ERRORS[name] = (
                "the native kernels build on macOS (Accelerate) and Linux (OpenBLAS) only"
            )
            return None
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            hint = (
                " (on Linux it needs OpenBLAS: apt install libopenblas-dev, or point "
                "PINNRL_NATIVE_CXXFLAGS / PINNRL_NATIVE_LDFLAGS at its headers and library)"
                if blas and sys.platform.startswith("linux")
                else ""
            )
            _ERRORS[name] = f"compiling {main} failed{hint}: {result.stderr.strip()[-400:]}"
            logger.warning(_ERRORS[name])
            return None
        os.replace(tmp, lib_path)
    lib = ctypes.CDLL(str(lib_path))
    if name == "jets":
        lib.pinn_jets_work_size.restype = ctypes.c_long
        lib.pinn_jets_forward.restype = None
        lib.pinn_jets_backward.restype = None
    else:
        lib.evap_solve.restype = ctypes.c_int
    _LOADED[name] = lib
    return lib


def _load() -> ctypes.CDLL | None:
    """The jets kernel (the network's derivatives), or ``None`` when it cannot be built."""
    global _LIB, _FAILED
    _LIB = _load_library("jets")
    _FAILED = _ERRORS.get("jets")
    return _LIB


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
    """``(S N, k)`` output streams of the MLP: the value and the Taylor coefficients.

    ``params`` are ``(W0, b0, W1, b1, ...)`` followed, with layer normalization,
    by ``(gamma0, beta0, gamma1, beta1, ...)`` for the hidden layers.
    """

    @staticmethod
    def forward(ctx, x, cols, alphas, n_layers, eps, *params):  # noqa: D102
        lib = _load()
        assert lib is not None
        linear, norm = params[: 2 * n_layers], params[2 * n_layers :]
        weights = [p.detach().contiguous() for p in linear[0::2]]
        biases = [p.detach().contiguous() for p in linear[1::2]]
        gammas = [p.detach().contiguous() for p in norm[0::2]]
        betas = [p.detach().contiguous() for p in norm[1::2]]
        n = x.shape[0]
        sizes = [x.shape[1]] + [w.shape[0] for w in weights]
        streams, dims = len(alphas), len(cols)
        flat_alpha = [a for alpha in alphas for a in alpha]
        c_sizes = _int_array(sizes)
        work = _take(lib.pinn_jets_work_size(n, n_layers, c_sizes, streams, int(bool(gammas))))
        out = torch.empty(streams * n, sizes[-1])
        points = x.detach().contiguous()
        lib.pinn_jets_forward(
            ctypes.c_void_p(points.data_ptr()), n, n_layers, c_sizes, _pointer_array(weights),
            _pointer_array(biases), streams, dims, _int_array(cols), _int_array(flat_alpha),
            _pointer_array(gammas) if gammas else None, _pointer_array(betas) if gammas else None,
            ctypes.c_float(eps), ctypes.c_void_p(work.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        )  # fmt: skip
        ctx.jet_state = (work, weights, gammas, eps, sizes, n, dims, tuple(flat_alpha))
        ctx.jet_call = (tuple(cols), tuple(alphas), n_layers, eps)
        ctx.save_for_backward(x, *params)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        # The compiled reverse pass gives numbers, not a graph, and it consumes the work buffer.
        # A gradient that must itself be differentiated (``create_graph=True``: Hessian-vector
        # products, the loss spectrum, sensitivities) or a second pass through a retained graph
        # is taken by autograd on the same function instead.
        if torch.is_grad_enabled() or ctx.jet_state is None:
            return _Jets._autograd_backward(ctx, grad_out)
        lib = _load()
        assert lib is not None
        work, weights, gammas, eps, sizes, n, dims, flat_alpha = ctx.jet_state
        streams = len(flat_alpha) // dims if dims else 1
        grad_w = [torch.empty_like(w) for w in weights]
        grad_b = [torch.empty(w.shape[0]) for w in weights]
        grad_gamma = [torch.empty_like(g) for g in gammas]
        grad_beta = [torch.empty_like(g) for g in gammas]
        scratch = _take(2 * streams * n * max(sizes))
        grad_out = grad_out.contiguous()
        lib.pinn_jets_backward(
            n, len(weights), _int_array(sizes), _pointer_array(weights), streams, dims,
            _int_array(flat_alpha), _pointer_array(gammas) if gammas else None, ctypes.c_float(eps),
            ctypes.c_void_p(work.data_ptr()), ctypes.c_void_p(grad_out.data_ptr()),
            _pointer_array(grad_w), _pointer_array(grad_b),
            _pointer_array(grad_gamma) if gammas else None,
            _pointer_array(grad_beta) if gammas else None, ctypes.c_void_p(scratch.data_ptr()),
        )  # fmt: skip
        _give(scratch)
        _give(work)
        ctx.jet_state = None
        grads: list[torch.Tensor | None] = [None, None, None, None, None]
        for gw, gb in zip(grad_w, grad_b, strict=True):
            grads += [gw, gb]
        for gg, gb in zip(grad_gamma, grad_beta, strict=True):
            grads += [gg, gb]
        return tuple(grads)

    @staticmethod
    def _autograd_backward(ctx, grad_out):
        x, *params = ctx.saved_tensors
        cols, alphas, n_layers, eps = ctx.jet_call
        if ctx.jet_state is not None:
            _give(ctx.jet_state[0])
            ctx.jet_state = None
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            out = autograd_streams(x, cols, alphas, n_layers, eps, params)
            wanted = [p for p in params if p.requires_grad]
            got = iter(
                torch.autograd.grad(
                    out, wanted, grad_outputs=grad_out, create_graph=create_graph, allow_unused=True
                )
                if wanted
                else ()
            )
        grads = [next(got) if p.requires_grad else None for p in params]
        return (None, None, None, None, None, *grads)


def autograd_streams(
    x: torch.Tensor,
    cols: Sequence[int],
    alphas: Sequence[Sequence[int]],
    n_layers: int,
    eps: float,
    params: Sequence[torch.Tensor],
) -> torch.Tensor:
    """The kernel's output streams by autograd: the same function, differentiable to any order.

    ``(S N, k)``: for each multi-index of ``alphas`` (over the input columns
    ``cols``) the Taylor coefficient ``D^alpha u / alpha!`` of the network
    with the weights ``params`` (the layout of :func:`mlp_jets`). The kernel
    uses it where a gradient has to carry a graph, and the tests as the
    reference.
    """
    linear, norm = params[: 2 * n_layers], params[2 * n_layers :]
    x = x.detach().requires_grad_(any(sum(a) for a in alphas))
    h = x
    for k in range(n_layers):
        h = torch.nn.functional.linear(h, linear[2 * k], linear[2 * k + 1])
        if k < n_layers - 1:
            if norm:  # written out: nn.LayerNorm's third-order derivative is wrong
                c = h - h.mean(dim=-1, keepdim=True)
                h = c * torch.rsqrt(c.square().mean(dim=-1, keepdim=True) + eps)
                h = norm[2 * k + 1] + norm[2 * k] * h
            h = torch.tanh(h)
    done: dict[tuple[int, ...], list[torch.Tensor]] = {
        tuple([0] * len(cols)): [h[:, j] for j in range(h.shape[1])]
    }
    streams = []
    for alpha in alphas:  # sorted by total degree, so one lower index is always there
        alpha = tuple(alpha)
        if alpha not in done:
            j = next(i for i, a in enumerate(alpha) if a)
            lower = alpha[:j] + (alpha[j] - 1,) + alpha[j + 1 :]
            done[alpha] = [
                torch.autograd.grad(f.sum(), x, create_graph=True)[0][:, cols[j]]
                for f in done[lower]
            ]
        factorial = math.prod(math.factorial(a) for a in alpha)
        streams.append(torch.stack(done[alpha], dim=1) / factorial)
    return torch.cat(streams, dim=0)


def supported_stack(
    layers: Sequence[torch.nn.Module],
) -> tuple[list[torch.nn.Linear], list[torch.nn.LayerNorm]] | None:
    """The layers of a ``Linear, [LayerNorm,] Tanh, ..., Linear`` stack, or ``None`` for anything else.

    Returns the ``Linear`` layers and the ``LayerNorm`` layers (one per hidden
    layer, or none at all; all with affine parameters and the same ``eps``).
    Dropout with probability zero and identity modules are ignored.
    """
    linears: list[torch.nn.Linear] = []
    norms: list[torch.nn.LayerNorm] = []
    state = "linear"  # what must come next: linear, then (norm or tanh)
    for layer in layers:
        if isinstance(layer, torch.nn.Identity) or (
            isinstance(layer, torch.nn.Dropout) and layer.p == 0.0
        ):
            continue
        if state == "linear" and isinstance(layer, torch.nn.Linear) and layer.bias is not None:
            linears.append(layer)
            state = "after-linear"
        elif (
            state == "after-linear"
            and isinstance(layer, torch.nn.LayerNorm)
            and layer.elementwise_affine
            and layer.bias is not None
            and tuple(layer.normalized_shape) == (linears[-1].out_features,)
        ):
            norms.append(layer)
            state = "after-norm"
        elif state in ("after-linear", "after-norm") and isinstance(layer, torch.nn.Tanh):
            state = "linear"
        else:
            return None
    if not linears or state != "after-linear":  # must end on a Linear
        return None
    if norms and (len(norms) != len(linears) - 1 or len({m.eps for m in norms}) != 1):
        return None
    return linears, norms


def supported_layers(layers: Sequence[torch.nn.Module]) -> list[torch.nn.Linear] | None:
    """The ``Linear`` layers of a stack the kernel covers (see :func:`supported_stack`), or ``None``."""
    stack = supported_stack(layers)
    return None if stack is None else stack[0]


def taylor_streams(
    derivatives: Sequence[Sequence[int]],
) -> tuple[list[int], list[tuple[int, ...]]]:
    """The input columns and the multi-indices the kernel carries for ``derivatives``.

    Each derivative is a tuple of input columns (``(0, 0, 1)`` for
    ``d^3u / dx_0^2 dx_1``). The multi-indices (over the sorted distinct
    columns) are the downward closure of the requested ones, the zero index
    first and then by total degree, which is the order the kernel's
    recurrences need.
    """
    cols = sorted({c for d in derivatives for c in d})
    index = {c: j for j, c in enumerate(cols)}
    alphas = {tuple([0] * len(cols))}
    for d in derivatives:
        top = [0] * len(cols)
        for c in d:
            top[index[c]] += 1
        alphas.update(itertools.product(*(range(k + 1) for k in top)))
    return cols, sorted(alphas, key=lambda a: (sum(a), a))


def _columns(cols: Sequence[int], alpha: Sequence[int]) -> tuple[int, ...]:
    return tuple(c for c, k in zip(cols, alpha, strict=True) for _ in range(k))


def mlp_jets(
    linears: Sequence[torch.nn.Linear],
    x: torch.Tensor,
    derivatives: Sequence[Sequence[int]] = (),
    norms: Sequence[torch.nn.LayerNorm] = (),
) -> tuple[torch.Tensor, dict[tuple[int, ...], torch.Tensor]]:
    """Value and derivatives of a tanh MLP at ``x`` through the compiled kernel.

    Args:
        linears: The ``Linear`` layers in order (``tanh`` between them).
        x: Inputs ``(N, d)``, float32 on the CPU.
        derivatives: The derivatives wanted, each a tuple of input columns:
            ``(0,)`` for ``du/dx_0``, ``(1, 1)`` for ``d2u/dx_1^2``, ``(0, 1)``
            for ``d2u/dx_0 dx_1``, ``(0, 0, 0)`` for the third derivative along
            ``x_0``. Any order; a derivative brings every lower one it is built
            from (``(0, 0, 0)`` gives ``(0,)`` and ``(0, 0)`` too).
        norms: The ``LayerNorm`` after each hidden ``Linear`` (before its ``tanh``), or none.

    Returns:
        ``(u, du)``: the output ``(N, k)`` and ``{columns: derivative}`` with
        the columns sorted, for the requested derivatives and those they are
        built from, all differentiable with respect to the weights.

    Raises:
        RuntimeError: When the kernel is not available (see :func:`unavailable_reason`) or the
            input is not float32 on the CPU.
        ValueError: When ``norms`` does not have one entry per hidden layer, or a derivative
            names a column ``x`` does not have.
    """
    if not available():
        raise RuntimeError(unavailable_reason())
    if x.dtype != torch.float32 or x.device.type != "cpu":
        raise RuntimeError("the native kernel needs float32 inputs on the CPU")
    if norms and len(norms) != len(linears) - 1:
        raise ValueError("norms needs one LayerNorm per hidden layer")
    cols, alphas = taylor_streams(derivatives)
    if cols and (cols[0] < 0 or cols[-1] >= x.shape[1]):
        raise ValueError(f"derivative columns must be in 0..{x.shape[1] - 1}, got {cols}")
    params: list[torch.Tensor] = []
    for lin in linears:
        params += [lin.weight, lin.bias]
    for norm in norms:
        params += [norm.weight, norm.bias]
    eps = float(norms[0].eps) if norms else 0.0
    out = _Jets.apply(x, tuple(cols), tuple(alphas), len(linears), eps, *params)
    n = x.shape[0]
    du: dict[tuple[int, ...], torch.Tensor] = {}
    for s, alpha in enumerate(alphas):
        if s == 0:
            continue
        factorial = math.prod(math.factorial(k) for k in alpha)
        block = out[s * n : (s + 1) * n]
        du[_columns(cols, alpha)] = block * factorial if factorial != 1 else block
    return out[:n], du


__all__ = [
    "autograd_streams",
    "available",
    "evaporator_available",
    "evaporator_solve",
    "mlp_jets",
    "supported_layers",
    "supported_stack",
    "taylor_streams",
    "unavailable_reason",
]

from pinnrl.native.evaporator import evaporator_available, evaporator_solve  # noqa: E402
