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
_VERSION = b"specialize-4"


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


# ---------------------------------------------------------------------- layer normalization
def norm_program(alphas: Sequence[Sequence[int]]) -> str:
    """C++ for the Taylor streams through layer normalization, forward and reverse.

    Per row of a layer, with ``c_s = a_s - mean(a_s)``, ``v_s = mean(sum c_b c_c)`` (plus
    ``eps`` at 0), ``r = v^(-1/2)`` by its recurrence and ``z_s = sum c_b r_c``, the output is
    ``y = gamma z + beta``. The general kernel walks index tables with one loop over the
    features per term; here every row is three loops over the features each way, each with a
    straight-line body for the streams in ``alphas``. The state saved for the reverse pass has
    the layout of ``NormState`` in ``jets.cpp``.
    """
    alphas = [tuple(a) for a in alphas]
    index = {a: s for s, a in enumerate(alphas)}
    n = len(alphas)
    ordered: list[list[tuple[int, int]]] = [[] for _ in range(n)]  # (b, c) with b + c = s
    unordered: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]
    rterms: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]  # (coef, r index, v index)
    rdiv = [1] * n
    for s, al in enumerate(alphas):
        i = next((j for j, a in enumerate(al) if a), 0)
        if s > 0:
            rdiv[s] = al[i]
        for t, be in enumerate(alphas):
            if any(b > a for a, b in zip(al, be, strict=True)):
                continue
            diff = tuple(a - b for a, b in zip(al, be, strict=True))
            g = index[diff]
            ordered[s].append((t, g))
            if t <= g:
                unordered[s].append((1.0 if t == g else 2.0, t, g))
            if s > 0 and diff[i] >= 1:
                rterms[s].append((-0.5 * diff[i], t, g))
                if t != 0:
                    rterms[s].append((-float(diff[i]), g, t))
    dots = sorted({(b, c) for s in range(n) for _, b, c in unordered[s]})
    rng = range(n)

    def per_stream(template: str, sep: str = " ") -> str:
        return sep.join(template.format(s=s) for s in rng)

    def lit(value: float) -> str:
        return f"{value!r}f"

    # ---- forward
    fwd = []
    fwd.append(per_stream("const float* __restrict a{s} = A + {s} * step + (size_t)n * w;"))
    fwd.append(per_stream("float* __restrict y{s} = Y + {s} * step + (size_t)n * w;"))
    fwd.append(per_stream("float* __restrict c{s} = st.c + ((size_t)n * S + {s}) * w;"))
    fwd.append(per_stream("float* __restrict z{s} = st.z + ((size_t)n * S + {s}) * w;"))
    fwd.append(per_stream("float m{s} = 0.0f;"))
    fwd.append("PINN_SIMD for (int k = 0; k < w; ++k) { " + per_stream("m{s} += a{s}[k];") + " }")
    fwd.append(per_stream("m{s} *= inv;"))
    fwd.append(" ".join(f"float d{b}_{c} = 0.0f;" for b, c in dots))
    fwd.append("PINN_SIMD for (int k = 0; k < w; ++k) {")
    fwd.append("    " + per_stream("const float x{s} = a{s}[k] - m{s}; c{s}[k] = x{s};"))
    fwd.append("    " + " ".join(f"d{b}_{c} += x{b} * x{c};" for b, c in dots))
    fwd.append("}")
    for s in rng:
        parts = [f"d{b}_{c}" if f == 1.0 else f"{lit(f)} * d{b}_{c}" for f, b, c in unordered[s]]
        extra = " + eps" if s == 0 else ""
        fwd.append(f"const float v{s} = inv * ({' + '.join(parts)}){extra};")
    fwd.append("const float r0 = 1.0f / std::sqrt(v0);")
    for s in range(1, n):
        parts = [f"{lit(c)} * r{rr} * v{vv}" for c, rr, vv in rterms[s]]
        fwd.append(f"const float r{s} = ({' + '.join(parts)}) / (v0 * {lit(float(rdiv[s]))});")
    fwd.append("PINN_SIMD for (int k = 0; k < w; ++k) {")
    for s in rng:
        parts = [f"c{b}[k] * r{c}" for b, c in ordered[s]]
        fwd.append(f"    const float t{s} = {' + '.join(parts)};")
        tail = " + beta[k]" if s == 0 else ""
        fwd.append(f"    z{s}[k] = t{s}; y{s}[k] = gamma[k] * t{s}{tail};")
    fwd.append("}")
    fwd.append("float* vs = st.v + (size_t)n * S; float* rs = st.r + (size_t)n * S;")
    fwd.append(per_stream("vs[{s}] = v{s}; rs[{s}] = r{s};"))

    # ---- reverse
    bwd = []
    bwd.append(per_stream("float* __restrict g{s} = G + {s} * step + (size_t)n * w;"))
    bwd.append(per_stream("const float* __restrict c{s} = st.c + ((size_t)n * S + {s}) * w;"))
    bwd.append(per_stream("const float* __restrict z{s} = st.z + ((size_t)n * S + {s}) * w;"))
    bwd.append(per_stream("float* __restrict ch{s} = chat + {s} * w;"))
    bwd.append("const float* vs = st.v + (size_t)n * S; const float* rs = st.r + (size_t)n * S;")
    bwd.append(per_stream("const float v{s} = vs[{s}]; const float r{s} = rs[{s}];"))
    bwd.append(per_stream("(void)v{s};"))
    bwd.append(per_stream("float rh{s} = 0.0f;"))
    bwd.append("PINN_SIMD for (int k = 0; k < w; ++k) {")
    bwd.append("    gbeta[k] += g0[k];")
    bwd.append("    ggamma[k] += " + per_stream("g{s}[k] * z{s}[k]", " + ") + ";")
    bwd.append("    " + per_stream("const float zh{s} = gamma[k] * g{s}[k];"))
    chat_terms: list[list[str]] = [[] for _ in rng]
    rhat_terms: list[list[str]] = [[] for _ in rng]
    for s in rng:
        for b, c in ordered[s]:
            chat_terms[b].append(f"zh{s} * r{c}")
            rhat_terms[c].append(f"zh{s} * c{b}[k]")
    for s in rng:
        bwd.append(f"    ch{s}[k] = {' + '.join(chat_terms[s]) if chat_terms[s] else '0.0f'};")
        if rhat_terms[s]:
            bwd.append(f"    rh{s} += {' + '.join(rhat_terms[s])};")
    bwd.append("}")
    bwd.append(per_stream("float vh{s} = 0.0f;"))
    for s in range(n - 1, 0, -1):
        bwd.append(
            f"{{ const float sc = rh{s} / (v0 * {lit(float(rdiv[s]))}); vh0 -= rh{s} * r{s} / v0;"
        )
        for coef, rr, vv in rterms[s]:
            bwd.append(f"  rh{rr} += sc * {lit(coef)} * v{vv}; vh{vv} += sc * {lit(coef)} * r{rr};")
        bwd.append("}")
    bwd.append("vh0 += rh0 * (-0.5f * r0 * r0 * r0);")
    bwd.append(per_stream("const float gv{s} = 2.0f * vh{s} * inv;"))
    bwd.append(per_stream("(void)gv{s};"))
    add_terms: list[list[str]] = [[] for _ in rng]
    for s in rng:
        for _f, b, c in unordered[s]:
            add_terms[b].append(f"gv{s} * c{c}[k]")
            if b != c:
                add_terms[c].append(f"gv{s} * c{b}[k]")
    bwd.append(per_stream("float m{s} = 0.0f;"))
    bwd.append("PINN_SIMD for (int k = 0; k < w; ++k) {")
    for s in rng:
        extra = (" + " + " + ".join(add_terms[s])) if add_terms[s] else ""
        bwd.append(f"    const float u{s} = ch{s}[k]{extra}; ch{s}[k] = u{s}; m{s} += u{s};")
    bwd.append("}")
    bwd.append(per_stream("m{s} *= inv;"))
    bwd.append(
        "PINN_SIMD for (int k = 0; k < w; ++k) { " + per_stream("g{s}[k] = ch{s}[k] - m{s};") + " }"
    )

    body_f = "\n".join("        " + line for line in fwd)
    body_b = "\n".join("        " + line for line in bwd)
    return f"""inline void gen_norm_fwd(const float* A, float* Y, int N, int w, const float* __restrict gamma,
                         const float* __restrict beta, float eps, float* state) {{
    const int S = {n};
    const size_t step = (size_t)N * w;
    const float inv = 1.0f / w;
    NormState st(state, N, S, w);
    for (int n = 0; n < N; ++n) {{
{body_f}
    }}
}}

// ggamma and gbeta are accumulated into.
inline void gen_norm_bwd(const float* state, float* G, int N, int w, const float* __restrict gamma,
                         float* __restrict ggamma, float* __restrict gbeta) {{
    const int S = {n};
    const size_t step = (size_t)N * w;
    const float inv = 1.0f / w;
    NormState st(state, N, S, w);
    static thread_local std::vector<float> scratch;
    if (scratch.size() < (size_t)S * w) scratch.resize((size_t)S * w);
    float* chat = scratch.data();
    for (int n = 0; n < N; ++n) {{
{body_b}
    }}
}}
"""


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

    equations: tuple[str, ...]  # the source text of each residual
    alphas: tuple[tuple[int, ...], ...]  # the streams
    n_inputs: int
    n_outputs: int
    code: str  # the residual loop
    parameters: tuple[str, ...] = field(default=())  # the ``p`` array
    trainable: tuple[str, ...] = field(default=())  # the parameters whose gradient is returned
    externals: tuple[str, ...] = field(default=())  # values given per point (``E``)


def residual_code(
    expressions: Sequence[sympy.Expr],
    streams: Mapping[str, tuple[int, int]],
    coordinates: Mapping[str, int],
    parameters: Sequence[str],
    n_inputs: int,
    n_outputs: int,
    trainable: Sequence[str] = (),
    externals: Sequence[str] = (),
) -> str | None:
    """The loop over the points of a block: sum of squared residuals and its adjoints.

    ``streams`` maps a symbol of the expressions to ``(stream, output column)``: at run time
    the symbol is ``fac[stream * k + column]`` times that Taylor coefficient (the factorial of
    the multi-index and the network's output and input scaling), plus ``off[column]`` for a
    value. ``coordinates`` maps a symbol to an input column, ``parameters`` lists the symbols
    of the ``p`` array (those in ``trainable`` get their gradient accumulated in ``gp``) and
    ``externals`` the symbols whose value is given per point in ``E`` (an unknown function, a
    measured input), with their adjoints written to ``GE``. Returns ``None`` when an
    expression has no C form (the caller falls back).
    """
    printer = _Printer()
    pi = sympy.Float(math.pi)
    rename = {sympy.Symbol(name): sympy.Symbol(f"s_{name}") for name in streams}
    rename.update({sympy.Symbol(name): sympy.Symbol(f"c_{name}") for name in coordinates})
    rename.update({sympy.Symbol(name): sympy.Symbol(f"p_{name}") for name in parameters})
    rename.update({sympy.Symbol(name): sympy.Symbol(f"x_{name}") for name in externals})
    exprs = [e.xreplace({sympy.pi: pi}).xreplace(rename) for e in expressions]
    known = set(rename.values())
    if any(e.free_symbols - known for e in exprs):
        return None
    free = {s.name for e in exprs for s in e.free_symbols}
    used = sorted(name for name in streams if f"s_{name}" in free)
    wrt = (
        [f"s_{name}" for name in used]
        + [f"x_{name}" for name in externals]
        + [f"p_{name}" for name in trainable]
    )
    n_eq, k = len(exprs), n_outputs
    partials = [sympy.diff(e, sympy.Symbol(name)) for e in exprs for name in wrt]
    try:
        temporaries, reduced = sympy.cse(
            list(exprs) + partials, symbols=sympy.numbered_symbols("w")
        )
        lines = []
        for name in used:
            s, j = streams[name]
            offset = f" + off[{j}]" if s == 0 else ""
            lines.append(
                f"const float s_{name} = fac[{s * k + j}] * out[((size_t){s} * nb + n) * {k} + {j}]{offset};"
            )
        for name, col in coordinates.items():
            if f"c_{name}" in free:
                lines.append(f"const float c_{name} = X[(size_t)n * {n_inputs} + {col}];")
        for i, name in enumerate(parameters):
            if f"p_{name}" in free:
                lines.append(f"const float p_{name} = p[{i}];")
        for i, name in enumerate(externals):
            if f"x_{name}" in free:
                lines.append(f"const float x_{name} = E[(size_t)n * {len(externals)} + {i}];")
        for sym, value in temporaries:
            lines.append(f"const float {sym} = {printer.doprint(value)};")
        for q in range(n_eq):
            lines.append(f"const float r{q} = m[{q}] * ({printer.doprint(reduced[q])});")
            lines.append(f"const float e{q} = 2.0f * m[{q}] * r{q};")
        lines.append("acc += " + " + ".join(f"r{q} * r{q}" for q in range(n_eq)) + ";")

        def adjoint(position: int) -> str | None:
            parts = []
            for q in range(n_eq):
                partial = reduced[n_eq + q * len(wrt) + position]
                if partial != 0:
                    parts.append(f"e{q} * ({printer.doprint(partial)})")
            return " + ".join(parts) if parts else None

        for i, name in enumerate(used):
            s, j = streams[name]
            value = adjoint(i)
            if value:
                lines.append(
                    f"G[((size_t){s} * nb + n) * {k} + {j}] = fac[{s * k + j}] * ({value});"
                )
        for i, _name in enumerate(externals):
            value = adjoint(len(used) + i)
            lines.append(f"GE[(size_t)n * {len(externals)} + {i}] = {value or '0.0f'};")
        for i, _name in enumerate(trainable):
            value = adjoint(len(used) + len(externals) + i)
            if value:
                lines.append(f"gp{i} += {value};")
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
    declare_gp = "\n".join(f"    float gp{i} = 0.0f;" for i in range(len(spec.trainable)))
    store_gp = "\n".join(f"    gparam[{i}] += (double)gp{i};" for i in range(len(spec.trainable)))
    return f"""// Generated by pinnrl.native.specialize for: {" ; ".join(spec.equations)}
#include "{_HERE / "jets.cpp"}"
#include "{_HERE / "step.h"}"

namespace {{

inline void gen_tanh_fwd(const float* A, float* Hn, size_t M) {{
{load_a}
{load_o}
    PINN_SIMD for (size_t e = 0; e < M; ++e) {{
        {get_a}
{body_f}
        {put_o}
    }}
}}

inline void gen_tanh_bwd(const float* A, const float* H, float* G, size_t M) {{
{load_a}
{load_h}
{load_g}
    PINN_SIMD for (size_t e = 0; e < M; ++e) {{
        {get_a} (void)a0;
        {get_h}
        {get_g}
{body_r}
        {put_g}
    }}
}}

{norm_program(spec.alphas)}
inline double gen_residual(const float* __restrict out, float* __restrict G, const float* __restrict X,
                           int nb, const float* p, const float* m, const float* fac,
                           const float* off, const float* __restrict E, float* __restrict GE,
                           double* gparam) {{
    std::memset(G, 0, sizeof(float) * (size_t){n} * nb * {spec.n_outputs});
    float acc = 0.0f;
{declare_gp}
    (void)X; (void)p; (void)fac; (void)off; (void)E; (void)GE; (void)gparam;
    for (int n = 0; n < nb; ++n) {{
{spec.code}
    }}
{store_gp}
    return (double)acc;
}}

}}  // namespace

extern "C" double pinn_residual_step(const float* X, const float* XP, int N, int L, const int* sizes,
                                     const float* const* W, const float* const* b, const int* cols,
                                     const int* alpha, const float* const* gamma,
                                     const float* const* beta, float eps, const float* params,
                                     const float* mult, const float* fac, const float* off,
                                     const float* E, float* GE, double* gparam, float scale,
                                     float* const* gW, float* const* gb, float* const* ggamma,
                                     float* const* gbeta, int block) {{
    return blocked_residual(X, XP, N, L, sizes, W, b, {n}, {len(spec.alphas[0])}, cols, alpha, gamma,
                            beta, eps, params, mult, fac, off, E, GE, {len(spec.externals)}, gparam,
                            {len(spec.trainable)}, scale, gW, gb, ggamma, gbeta, block, gen_tanh_fwd,
                            gen_tanh_bwd, gen_norm_fwd, gen_norm_bwd, gen_residual);
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
    """The mean squared residual, with every gradient computed in the same call.

    The differentiable inputs are leaves: the trainable parameters of the equations, the
    network's weights and the parameters of the unknown functions.
    """

    @staticmethod
    def forward(ctx, call, fallback, *tensors):  # noqa: D102
        loss, grads = call(tensors)
        ctx.step_grads = grads
        ctx.step_tensors = tensors
        ctx.fallback = fallback
        return loss

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        tensors = ctx.step_tensors
        if torch.is_grad_enabled():  # a gradient that carries a graph: autograd's business
            with torch.enable_grad():
                loss = ctx.fallback()
                wanted = [p for p in tensors if p.requires_grad]
                got = iter(
                    torch.autograd.grad(
                        loss, wanted, grad_outputs=grad_out, create_graph=True, allow_unused=True
                    )
                )
            return (None, None, *(next(got) if p.requires_grad else None for p in tensors))
        return (None, None, *(None if g is None else g * grad_out for g in ctx.step_grads))


def run_step(
    lib: ctypes.CDLL,
    spec: ResidualSpec,
    cols: Sequence[int],
    linears: Sequence[torch.nn.Linear],
    norms: Sequence[torch.nn.LayerNorm],
    x: torch.Tensor,
    parameters: Sequence[float | torch.Tensor],
    multipliers: Sequence[float],
    fallback: Callable[[], torch.Tensor],
    scaling: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]
    | None = None,
    externals: Callable[[], tuple[torch.Tensor, Sequence[torch.Tensor]]] | None = None,
) -> torch.Tensor:
    """Mean squared residual at ``x`` as a scalar that backpropagates to every unknown.

    ``parameters`` are the values of ``spec.parameters``, tensors for those in
    ``spec.trainable``. ``scaling`` is the network's ``(input lower bounds, input spans,
    output scale, output offset)`` when it is non-dimensionalized. ``externals`` returns the
    per-point values ``(N, len(spec.externals))`` of the unknown functions and inputs, with
    their graph, and the parameters that graph leads to.
    """
    weights_and_biases: list[torch.Tensor] = []
    for lin in linears:
        weights_and_biases += [lin.weight, lin.bias]
    for norm in norms:
        weights_and_biases += [norm.weight, norm.bias]
    n_layers = len(linears)
    eps = float(norms[0].eps) if norms else 0.0
    flat_alpha = [a for alpha in spec.alphas for a in alpha]
    n_eq, k = len(spec.equations), spec.n_outputs
    thetas = [
        v
        for name, v in zip(spec.parameters, parameters, strict=True)
        if name in spec.trainable and isinstance(v, torch.Tensor)
    ]
    values = [float(v) for v in parameters]
    points = x.detach().contiguous()
    n, d = points.shape
    slope = [1.0] * d
    out_scale, out_offset = [1.0] * k, [0.0] * k
    inputs = points
    if scaling is not None:
        lo, span, scale_t, offset_t = scaling
        if lo is not None and span is not None:
            inputs = (2.0 * (points - lo) / span - 1.0).contiguous()
            slope = [2.0 / float(v) for v in span.reshape(-1)]
        out_scale = [float(v) for v in scale_t.reshape(-1)]
        out_offset = [float(v) for v in offset_t.reshape(-1)]
    fac = []
    for alpha in spec.alphas:
        base = float(math.prod(math.factorial(a) for a in alpha))
        base *= math.prod(slope[c] ** a for c, a in zip(cols, alpha, strict=True))
        fac += [base * out_scale[j] for j in range(k)]
    extra: list[torch.Tensor] = []
    ext = None
    if externals is not None:
        with torch.enable_grad():
            ext, leaves = externals()
        extra = list(leaves)
    n_ext = len(spec.externals)

    def call(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, Any]:
        net = tensors[len(thetas) : len(thetas) + len(weights_and_biases)]
        linear, norm = net[: 2 * n_layers], net[2 * n_layers :]
        weights = [p.detach().contiguous() for p in linear[0::2]]
        biases = [p.detach().contiguous() for p in linear[1::2]]
        gammas = [p.detach().contiguous() for p in norm[0::2]]
        betas = [p.detach().contiguous() for p in norm[1::2]]
        sizes = [d] + [w.shape[0] for w in weights]
        grad_w = [torch.empty_like(w) for w in weights]
        grad_b = [torch.empty_like(v) for v in biases]
        grad_gamma = [torch.empty_like(g) for g in gammas]
        grad_beta = [torch.empty_like(g) for g in gammas]
        e_values = ext.detach().to(torch.float32).contiguous() if ext is not None else None
        e_grads = torch.empty_like(e_values) if e_values is not None else None
        g_param = (ctypes.c_double * max(len(thetas), 1))()
        ptr = native._pointer_array
        loss = lib.pinn_residual_step(
            ctypes.c_void_p(inputs.data_ptr()), ctypes.c_void_p(points.data_ptr()), n, n_layers,
            native._int_array(sizes), ptr(weights), ptr(biases), native._int_array(list(cols)),
            native._int_array(flat_alpha), ptr(gammas) if gammas else None,
            ptr(betas) if gammas else None, ctypes.c_float(eps),
            (ctypes.c_float * max(len(values), 1))(*values),
            (ctypes.c_float * n_eq)(*multipliers[:n_eq]), (ctypes.c_float * len(fac))(*fac),
            (ctypes.c_float * k)(*out_offset),
            ctypes.c_void_p(e_values.data_ptr()) if e_values is not None else None,
            ctypes.c_void_p(e_grads.data_ptr()) if e_grads is not None else None,
            g_param, ctypes.c_float(1.0 / (n * n_eq)), ptr(grad_w), ptr(grad_b),
            ptr(grad_gamma) if gammas else None, ptr(grad_beta) if gammas else None, block_size(),
        )  # fmt: skip
        grads: list[torch.Tensor | None] = [
            torch.full_like(theta, g_param[i]) for i, theta in enumerate(thetas)
        ]
        for gw, gb in zip(grad_w, grad_b, strict=True):
            grads += [gw, gb]
        for gg, gb in zip(grad_gamma, grad_beta, strict=True):
            grads += [gg, gb]
        if extra:  # from the adjoints of the per-point values to the parameters behind them
            if ext is not None and ext.requires_grad:
                with torch.enable_grad():
                    grads += list(
                        torch.autograd.grad(ext, extra, grad_outputs=e_grads, allow_unused=True)
                    )
            else:
                grads += [None] * len(extra)
        return torch.tensor(loss, dtype=torch.float32), grads

    assert n_ext == 0 or ext is not None
    return _ResidualStep.apply(call, fallback, *thetas, *weights_and_biases, *extra)
