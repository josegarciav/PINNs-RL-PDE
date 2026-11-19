"""A compiled residual step, specialised for one equation.

The general kernel (:func:`pinnrl.native.mlp_jets`) returns the network's derivatives and
leaves the residual, the loss and the start of the reverse pass to PyTorch. This module is the
step a tracing compiler would produce instead: for one set of equations it writes C++ in which

* the recurrences of the derivative streams through ``tanh`` are one loop each way, with a
  straight-line body for exactly the streams the equations use;
* the residual, its square and its adjoint are inlined at the output of the forward pass
  (SymPy differentiates the expressions and prints them as C);
* the batch runs in blocks of points that stay in the cache from the first layer of the
  forward pass to the last of the reverse pass,

and compiles it once per equation (cached next to the general kernel). One call returns the
mean squared residual and its gradient with respect to every weight. Experiment I17 measures
the step against the general kernel and against JAX.

:func:`residual_loss` is what :class:`~pinnrl.CustomPDE` calls; it returns ``None`` wherever
the specialised step does not apply, and the caller goes on with the general kernel.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import os
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy
import torch
from sympy.codegen.ast import float32, real
from sympy.printing.c import C99CodePrinter

from pinnrl import native

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_VERSION = b"specialize-1"


# ----------------------------------------------------------------------------- tanh streams
def tanh_program(alphas: Sequence[Sequence[int]]) -> tuple[list[str], list[str]]:
    """Straight-line C for the Taylor streams through ``tanh``, forward and reverse.

    ``alphas`` are the multi-indices of the streams (downward closed, sorted by degree, the
    zero index first), as :func:`pinnrl.native.taylor_streams` returns them. With
    ``s = tanh(a)`` and ``q = 1 - s^2`` the coefficients follow

        alpha_i s_alpha = sum_{beta <= alpha, beta_i >= 1} beta_i a_beta q_{alpha - beta},
        q_gamma = delta_{gamma 0} - sum_{beta <= gamma} s_beta s_{gamma - beta}.

    The forward body reads ``a0`` (already ``tanh`` of the value stream) and ``a1..`` and
    defines ``h0, h1, ...``. The reverse body reads ``a*``, ``h*`` and the adjoints ``g*`` of
    the ``h`` streams and defines ``r0, r1, ...``, the adjoints of the ``a`` streams.
    """
    alphas = [tuple(a) for a in alphas]
    index = {a: s for s, a in enumerate(alphas)}
    n = len(alphas)
    terms: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]  # (coef, a index, q index)
    pairs: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]  # (factor, b, c), b <= c
    q_needed = [False] * n
    q_needed[0] = n > 1
    for s, al in enumerate(alphas):
        i = next((j for j, a in enumerate(al) if a), 0)
        for t, be in enumerate(alphas):
            if any(b > a for a, b in zip(al, be, strict=True)):
                continue
            g = index[tuple(a - b for a, b in zip(al, be, strict=True))]
            if t <= g:
                pairs[s].append((1.0 if t == g else 2.0, t, g))
            if s > 0 and be[i] >= 1:
                terms[s].append((be[i] / al[i], t, g))
                q_needed[g] = True

    def lit(value: float) -> str:
        return f"{value!r}f"

    forward = ["const float h0 = a0;", "const float q0 = 1.0f - h0 * h0;"]
    for s in range(1, n):
        parts = [
            (f"a{t} * q{g}" if c == 1.0 else f"{lit(c)} * a{t} * q{g}") for c, t, g in terms[s]
        ]
        forward.append(f"const float h{s} = {' + '.join(parts)};")
        if q_needed[s]:
            parts = [
                f"{lit(f)} * h{b} * h{c}" if f != 1.0 else f"h{b} * h{c}" for f, b, c in pairs[s]
            ]
            forward.append(f"const float q{s} = -({' + '.join(parts)});")

    # the reverse pass recomputes q from the saved h streams, then walks the program backwards
    reverse = ["const float q0 = 1.0f - h0 * h0;"]
    for s in range(1, n):
        if q_needed[s]:
            parts = [
                f"{lit(f)} * h{b} * h{c}" if f != 1.0 else f"h{b} * h{c}" for f, b, c in pairs[s]
            ]
            reverse.append(f"const float q{s} = -({' + '.join(parts)});")
    hbar: list[list[str]] = [[f"g{s}"] for s in range(n)]
    abar: list[list[str]] = [[] for _ in range(n)]
    qbar: list[list[str]] = [[] for _ in range(n)]
    for s in range(n - 1, 0, -1):
        if q_needed[s] and qbar[s]:
            reverse.append(f"const float qb{s} = {' + '.join(qbar[s])};")
            for _, b, c in pairs[s]:
                hbar[b].append(f"-2.0f * qb{s} * h{c}")
                if b != c:
                    hbar[c].append(f"-2.0f * qb{s} * h{b}")
        reverse.append(f"const float hb{s} = {' + '.join(hbar[s])};".replace("+ -", "- "))
        for coef, t, g in terms[s]:
            scaled = f"hb{s}" if coef == 1.0 else f"{lit(coef)} * hb{s}"
            abar[t].append(f"{scaled} * q{g}")
            qbar[g].append(f"{scaled} * a{t}")
    if qbar[0]:
        reverse.append(f"const float qb0 = {' + '.join(qbar[0])};")
        hbar[0].append("-2.0f * qb0 * h0")
    reverse.append(f"const float r0 = ({' + '.join(hbar[0])}) * q0;".replace("+ -", "- "))
    for s in range(1, n):
        reverse.append(f"const float r{s} = {' + '.join(abar[s]) if abar[s] else '0.0f'};")
    return forward, reverse


# --------------------------------------------------------------------------------- residual
class _Printer(C99CodePrinter):
    """Single-precision C with small integer powers written as products."""

    def __init__(self):
        super().__init__(settings={"type_aliases": {real: float32}, "strict": True})

    def _print_Pow(self, expr):
        base, exponent = expr.as_base_exp()
        if exponent.is_Integer and 1 < abs(int(exponent)) <= 8:
            product = "*".join([f"({self._print(base)})"] * abs(int(exponent)))
            return f"({product})" if exponent > 0 else f"(1.0F/({product}))"
        return super()._print_Pow(expr)


@dataclass(frozen=True)
class ResidualSpec:
    """What the generated code depends on: the equations and the layout of their inputs."""

    equations: tuple[str, ...]  # SymPy source of each residual, in the C symbols below
    alphas: tuple[tuple[int, ...], ...]  # the streams
    n_inputs: int
    n_outputs: int
    code: str  # the residual loop
    parameters: tuple[str, ...] = field(default=())


def residual_code(
    expressions: Sequence[sympy.Expr],
    streams: Mapping[str, tuple[int, int, float]],
    coordinates: Mapping[str, int],
    parameters: Sequence[str],
    n_inputs: int,
    n_outputs: int,
) -> str | None:
    """The loop over the points of a block: sum of squared residuals and the output adjoints.

    ``streams`` maps a symbol of the expressions to ``(stream, output column, factor)``: the
    symbol is ``factor`` times that Taylor coefficient. ``coordinates`` maps a symbol to an
    input column and ``parameters`` lists the remaining symbols in the order of the ``p``
    array. Returns ``None`` when an expression has no C form (the caller falls back).
    """
    printer = _Printer()
    pi = sympy.Float(math.pi)
    rename = {sympy.Symbol(name): sympy.Symbol(f"s_{name}") for name in streams}
    rename.update({sympy.Symbol(name): sympy.Symbol(f"c_{name}") for name in coordinates})
    rename.update({sympy.Symbol(name): sympy.Symbol(f"p_{name}") for name in parameters})
    exprs = [e.xreplace({sympy.pi: pi}).xreplace(rename) for e in expressions]
    known = set(rename.values())
    if any(e.free_symbols - known for e in exprs):
        return None
    used = sorted({s.name[2:] for e in exprs for s in e.free_symbols if s.name.startswith("s_")})
    partials = [[sympy.diff(e, sympy.Symbol(f"s_{name}")) for name in used] for e in exprs]
    flat = list(exprs) + [p for row in partials for p in row]
    try:
        temporaries, reduced = sympy.cse(flat, symbols=sympy.numbered_symbols("w"))
        lines = []
        for name in used:
            s, j, factor = streams[name]
            scale = "" if factor == 1.0 else f"{factor!r}f * "
            lines.append(
                f"const float s_{name} = {scale}out[((size_t){s} * nb + n) * {n_outputs} + {j}];"
            )
        for e in exprs:
            for sym in sorted(e.free_symbols, key=lambda v: v.name):
                line = None
                if sym.name.startswith("c_"):
                    line = f"const float {sym.name} = X[(size_t)n * {n_inputs} + {coordinates[sym.name[2:]]}];"
                elif sym.name.startswith("p_"):
                    line = f"const float {sym.name} = p[{list(parameters).index(sym.name[2:])}];"
                if line and line not in lines:
                    lines.append(line)
        for sym, value in temporaries:
            lines.append(f"const float {sym} = {printer.doprint(value)};")
        n_eq = len(exprs)
        for q in range(n_eq):
            lines.append(f"const float r{q} = m[{q}] * ({printer.doprint(reduced[q])});")
            lines.append(f"const float e{q} = 2.0f * m[{q}] * r{q};")
        lines.append("acc += " + " + ".join(f"r{q} * r{q}" for q in range(n_eq)) + ";")
        for i, name in enumerate(used):
            s, j, factor = streams[name]
            parts = []
            for q in range(n_eq):
                partial = reduced[n_eq + q * len(used) + i]
                if partial != 0:
                    parts.append(f"e{q} * ({printer.doprint(partial)})")
            if parts:
                scale = "" if factor == 1.0 else f"{factor!r}f * "
                lines.append(
                    f"G[((size_t){s} * nb + n) * {n_outputs} + {j}] = {scale}({' + '.join(parts)});"
                )
    except Exception as error:  # noqa: BLE001 - a function SymPy cannot print as C
        logger.debug("no C form for the residual: %s", error)
        return None
    return "\n".join("        " + line for line in lines)


def _source(spec: ResidualSpec) -> str:
    forward, reverse = tanh_program(spec.alphas)
    n = len(spec.alphas)
    body_f = "\n".join("        " + line for line in forward)
    body_r = "\n".join("        " + line for line in reverse)
    load_a = "\n".join(f"    const float* __restrict A{s} = A + {s} * M;" for s in range(n))
    load_h = "\n".join(f"    const float* __restrict H{s} = H + {s} * M;" for s in range(n))
    load_o = "\n".join(f"    float* __restrict O{s} = Hn + {s} * M;" for s in range(n))
    load_g = "\n".join(f"    float* __restrict G{s} = G + {s} * M;" for s in range(n))
    get_a = " ".join(f"const float a{s} = A{s}[e];" for s in range(n))
    get_h = " ".join(f"const float h{s} = H{s}[e];" for s in range(n))
    get_g = " ".join(f"const float g{s} = G{s}[e];" for s in range(n))
    put_o = " ".join(f"O{s}[e] = h{s};" for s in range(n))
    put_g = " ".join(f"G{s}[e] = r{s};" for s in range(n))
    return f"""// Generated by pinnrl.native.specialize for: {" ; ".join(spec.equations)}
#include "{_HERE / "jets.cpp"}"
#include "{_HERE / "step.h"}"

namespace {{

inline void gen_tanh_fwd(const float* A, float* Hn, size_t M) {{
{load_a}
{load_o}
    for (size_t e = 0; e < M; ++e) {{
        {get_a}
{body_f}
        {put_o}
    }}
}}

inline void gen_tanh_bwd(const float* A, const float* H, float* G, size_t M) {{
{load_a}
{load_h}
{load_g}
    for (size_t e = 0; e < M; ++e) {{
        {get_a} (void)a0;
        {get_h}
        {get_g}
{body_r}
        {put_g}
    }}
}}

inline double gen_residual(const float* __restrict out, float* __restrict G, const float* __restrict X,
                           int nb, const float* p, const float* m) {{
    std::memset(G, 0, sizeof(float) * (size_t){n} * nb * {spec.n_outputs});
    float acc = 0.0f;
    (void)X; (void)p;
    for (int n = 0; n < nb; ++n) {{
{spec.code}
    }}
    return (double)acc;
}}

}}  // namespace

extern "C" double pinn_residual_step(const float* X, int N, int L, const int* sizes,
                                     const float* const* W, const float* const* b, const int* cols,
                                     const int* alpha, const float* const* gamma,
                                     const float* const* beta, float eps, const float* params,
                                     const float* mult, float scale, float* const* gW,
                                     float* const* gb, float* const* ggamma, float* const* gbeta,
                                     int block) {{
    return blocked_residual(X, N, L, sizes, W, b, {n}, {len(spec.alphas[0])}, cols, alpha, gamma, beta,
                            eps, params, mult, scale, gW, gb, ggamma, gbeta, block, gen_tanh_fwd,
                            gen_tanh_bwd, gen_residual);
}}
"""


# -------------------------------------------------------------------------------- building
_LIBS: dict[str, ctypes.CDLL | None] = {}
_PENDING: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def _digest(source: str) -> str:
    digest = hashlib.sha256(_VERSION)
    for name in ("jets.cpp", "step.h", "fast_tanh.h"):
        digest.update((_HERE / name).read_bytes())
    digest.update(source.encode())
    return digest.hexdigest()[:16]


def _build(source: str, key: str) -> None:
    """Compile ``source`` into the cache and load it; records ``None`` when that fails."""
    lib_path = native._cache_dir() / f"step-{key}.so"
    lib: ctypes.CDLL | None = None
    try:
        if not lib_path.exists():
            compiler = native._compiler()
            lib_path.parent.mkdir(parents=True, exist_ok=True)
            cpp = lib_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.cpp")
            tmp = cpp.with_suffix(".tmp")
            cpp.write_text(source)
            cmd = native._build_command(compiler, cpp, tmp, True) if compiler else None
            if cmd is None:
                raise RuntimeError("no compiler for this platform")
            result = subprocess.run(cmd, capture_output=True, text=True)
            cpp.unlink(missing_ok=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip()[-600:])
            os.replace(tmp, lib_path)
        lib = ctypes.CDLL(str(lib_path))
        lib.pinn_residual_step.restype = ctypes.c_double
    except Exception as error:  # noqa: BLE001 - the general kernel stays in charge
        logger.warning("the specialised step could not be built: %s", error)
        lib = None
    with _LOCK:
        _LIBS[key] = lib


def step_library(spec: ResidualSpec, wait: bool) -> ctypes.CDLL | None:
    """The compiled step of ``spec``; builds in the background and returns ``None`` until ready.

    With ``wait`` the call blocks until the build has finished (or failed).
    """
    source = _source(spec)
    key = _digest(source)
    with _LOCK:
        if key in _LIBS:
            return _LIBS[key]
        thread = _PENDING.get(key)
        if thread is None:
            thread = threading.Thread(target=_build, args=(source, key), daemon=True)
            _PENDING[key] = thread
            thread.start()
    if wait:
        thread.join()
        with _LOCK:
            return _LIBS.get(key)
    return None


def block_size() -> int:
    """Points per block of the specialised step (``PINNRL_NATIVE_BLOCK`` overrides it).

    1,024 measured best on an Apple M4 for 4 to 7 streams and layers of 64 (I17): smaller
    blocks pay the matrix product's cost per call, larger ones leave the cache.
    """
    return max(1, int(os.environ.get("PINNRL_NATIVE_BLOCK", "1024")))


def specialize_after() -> int | None:
    """Calls of a residual before its specialised step takes over (``None``: never).

    ``PINNRL_NATIVE_SPECIALIZE``: ``0`` or ``off`` disables the specialised step, any other
    integer is the number of calls served by the general kernel first (default 32, so a short
    run never pays for a compilation). The switch happens at exactly that call, waiting for
    the compiler if it has to, so a run does not depend on how long the build took.
    """
    raw = os.environ.get("PINNRL_NATIVE_SPECIALIZE", "32").strip().lower()
    if raw in ("0", "off", "no", "false"):
        return None
    return max(1, int(raw))


class _ResidualStep(torch.autograd.Function):
    """The mean squared residual, with the weight gradient computed in the same call."""

    @staticmethod
    def forward(ctx, x, call, fallback, *params):  # noqa: D102
        loss, grads = call(x, params)
        ctx.step_grads = grads
        ctx.step_params = params
        ctx.fallback = fallback
        return loss

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        if torch.is_grad_enabled():  # a gradient that carries a graph: autograd's business
            with torch.enable_grad():
                loss = ctx.fallback()
                params = [p for p in ctx.step_params if p.requires_grad]
                got = iter(
                    torch.autograd.grad(
                        loss, params, grad_outputs=grad_out, create_graph=True, allow_unused=True
                    )
                )
            grads = [next(got) if p.requires_grad else None for p in ctx.step_params]
            return (None, None, None, *grads)
        return (None, None, None, *(g * grad_out for g in ctx.step_grads))


def run_step(
    lib: ctypes.CDLL,
    spec: ResidualSpec,
    cols: Sequence[int],
    linears: Sequence[torch.nn.Linear],
    norms: Sequence[torch.nn.LayerNorm],
    x: torch.Tensor,
    parameters: Sequence[float],
    multipliers: Sequence[float],
    fallback: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Mean squared residual at ``x`` as a scalar that backpropagates to the weights."""
    params: list[torch.Tensor] = []
    for lin in linears:
        params += [lin.weight, lin.bias]
    for norm in norms:
        params += [norm.weight, norm.bias]
    n_layers = len(linears)
    eps = float(norms[0].eps) if norms else 0.0
    flat_alpha = [a for alpha in spec.alphas for a in alpha]
    n_eq = len(spec.equations)

    def call(points: torch.Tensor, tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, Any]:
        linear, norm = tensors[: 2 * n_layers], tensors[2 * n_layers :]
        weights = [p.detach().contiguous() for p in linear[0::2]]
        biases = [p.detach().contiguous() for p in linear[1::2]]
        gammas = [p.detach().contiguous() for p in norm[0::2]]
        betas = [p.detach().contiguous() for p in norm[1::2]]
        sizes = [points.shape[1]] + [w.shape[0] for w in weights]
        grad_w = [torch.empty_like(w) for w in weights]
        grad_b = [torch.empty_like(v) for v in biases]
        grad_gamma = [torch.empty_like(g) for g in gammas]
        grad_beta = [torch.empty_like(g) for g in gammas]
        points = points.detach().contiguous()
        n = points.shape[0]
        ptr = native._pointer_array
        loss = lib.pinn_residual_step(
            ctypes.c_void_p(points.data_ptr()), n, n_layers, native._int_array(sizes),
            ptr(weights), ptr(biases), native._int_array(list(cols)), native._int_array(flat_alpha),
            ptr(gammas) if gammas else None, ptr(betas) if gammas else None, ctypes.c_float(eps),
            (ctypes.c_float * max(len(parameters), 1))(*parameters),
            (ctypes.c_float * n_eq)(*multipliers[:n_eq]), ctypes.c_float(1.0 / (n * n_eq)),
            ptr(grad_w), ptr(grad_b), ptr(grad_gamma) if gammas else None,
            ptr(grad_beta) if gammas else None, block_size(),
        )  # fmt: skip
        grads: list[torch.Tensor] = []
        for gw, gb in zip(grad_w, grad_b, strict=True):
            grads += [gw, gb]
        for gg, gb in zip(grad_gamma, grad_beta, strict=True):
            grads += [gg, gb]
        return torch.tensor(loss, dtype=torch.float32), grads

    return _ResidualStep.apply(x, call, fallback, *params)
