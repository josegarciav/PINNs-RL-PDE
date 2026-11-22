"""D10: a least-squares optimizer for physics-informed losses, as a prototype in plain PyTorch.

A PINN loss is a sum of squares, L = 1/2 ||r(theta)||^2 with r the stacked residuals of the
equation, the initial condition and the boundary condition. Adam and L-BFGS treat it as a
generic function. Levenberg-Marquardt uses the squares: with J = dr/dtheta,

    delta = -(J^T J + lambda I)^{-1} J^T r        (parameter space, P x P)
          = -J^T (J J^T + lambda I)^{-1} r        (sample space,    N x N)

and lambda adapted by the gain ratio (Nielsen's rule). This script measures what that buys in
error reached per second on one CPU core, before anything is built into the compiled engine:
the same network, points and loss for

    lm          Levenberg-Marquardt, dense J by one gradient per point (torch.func), float64
    adam        Adam at 1e-3
    lbfgs       L-BFGS with a strong-Wolfe line search
    adam_lbfgs  Adam for the first 30 % of the budget, then L-BFGS

and, as variants on the same linear algebra (the first moves of PLAN.md):

    flow        M1: the Gauss-Newton flow theta' = -J^+ r by Euler steps whose length is set
                from an error estimate (how far the residual left its linear model), with one
                correction that reuses the factorization
    blockB      M4: B rows drawn in proportion to their squared residual from a pool, and
                satisfied by a B x B solve (block Kaczmarz); block32, block128, ...
    varpro      M5: the last layer solved by linear least squares at every step, the hidden
                weights stepped on the Jacobian projected off the last layer's (Kaufman)

and the second group of moves:

    continueK   M6: Levenberg-Marquardt while the viscosity moves from an easy value to the
                target in K geometric stages (Burgers only)
    filterB     M3: an extended Kalman filter over a stream of fresh batches of B rows, never
                revisited, carrying the weights' covariance
    al          M8: initial and boundary rows as constraints with multipliers (augmented
                Lagrangian, penalty 1) inside Levenberg-Marquardt; lm_w1 is its control, the
                same penalty without multipliers

on heat, Burgers (nu = 0.01/pi) and the KdV soliton, each for a fixed wall-clock budget, with
the relative L2 error against the exact or reference solution recorded along the way.

Usage: python experiments/D10_least_squares_optimizer/run.py [--budget 60] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

HERE = Path(__file__).parent
W_IC, W_BC = 10.0, 10.0
N_COL, N_IC, N_BC = 2000, 256, 256
METHODS = ("lm", "adam", "lbfgs", "adam_lbfgs")
VARIANTS = ("flow", "block32", "block128", "varpro", "varpro60")
STAGE2 = ("continue8", "filter64", "al", "lm_w1")


# ------------------------------------------------------------------------------ problems
def _kdv_exact(x, t):
    return 0.5 / np.cosh(0.5 * (x - t)) ** 2  # c = 1, u_t + 6 u u_x + u_xxx = 0


PROBLEMS = {
    "heat": dict(
        x=(0.0, 1.0),
        t=(0.0, 1.0),
        order=2,
        linear=True,
        residual=lambda u, ux, uxx, uxxx, ut: ut - 0.1 * uxx,
        initial=lambda x: torch.sin(math.pi * x),
        boundary=lambda x, t: torch.zeros_like(x),
        exact=lambda x, t: np.exp(-0.1 * np.pi**2 * t) * np.sin(np.pi * x),
    ),
    "burgers": dict(
        x=(-1.0, 1.0),
        t=(0.0, 1.0),
        order=2,
        residual=lambda u, ux, uxx, uxxx, ut: ut + u * ux - (0.01 / math.pi) * uxx,
        # the same equation at another viscosity, for continuation
        family=lambda nu: lambda u, ux, uxx, uxxx, ut: ut + u * ux - nu * uxx,
        easy=0.1,
        target=0.01 / math.pi,
        initial=lambda x: -torch.sin(math.pi * x),
        boundary=lambda x, t: torch.zeros_like(x),
        exact=None,  # the Cole-Hopf reference of pinnrl.benchmarks
    ),
    "kdv": dict(
        x=(-10.0, 12.0),
        t=(0.0, 2.0),
        order=3,
        residual=lambda u, ux, uxx, uxxx, ut: ut + 6.0 * u * ux + uxxx,
        initial=lambda x: 0.5 / torch.cosh(0.5 * x) ** 2,
        boundary=lambda x, t: 0.5 / torch.cosh(0.5 * (x - t)) ** 2,
        exact=_kdv_exact,
    ),
}


def reference(problem):
    """``(points (M, 2), values (M,))`` to grade a network against."""
    spec = PROBLEMS[problem]
    if spec["exact"] is None:
        from pinnrl.benchmarks.reference import burgers_reference

        ref = burgers_reference(nx=256, nt=101)
        xx, tt = ref.flat_points()
        return np.hstack([xx, tt]), ref.u.reshape(-1)
    x = np.linspace(*spec["x"], 201)
    t = np.linspace(*spec["t"], 101)
    xx, tt = np.meshgrid(x, t, indexing="ij")
    return np.stack([xx.ravel(), tt.ravel()], 1), spec["exact"](xx, tt).ravel()


def make_net(width, depth, seed):
    torch.manual_seed(seed)
    layers, n_in = [], 2
    for _ in range(depth):
        layers += [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
        n_in = width
    layers.append(torch.nn.Linear(n_in, 1))
    return torch.nn.Sequential(*layers).double()


def sample(problem, generator, sizes=None):
    n_col, n_ic, n_bc = sizes or (N_COL, N_IC, N_BC)
    spec = PROBLEMS[problem]
    (x0, x1), (t0, t1) = spec["x"], spec["t"]
    rand = lambda n: torch.rand(n, dtype=torch.float64, generator=generator)  # noqa: E731
    col = torch.stack([x0 + (x1 - x0) * rand(n_col), t0 + (t1 - t0) * rand(n_col)], 1)
    ic = torch.stack([x0 + (x1 - x0) * rand(n_ic), torch.full((n_ic,), t0, dtype=torch.float64)], 1)
    side = torch.where(torch.arange(n_bc) % 2 == 0, x0, x1).double()
    bc = torch.stack([side, t0 + (t1 - t0) * rand(n_bc)], 1)
    return col, ic, bc


class Scaler(torch.nn.Module):
    """Inputs mapped to [-1, 1] so every problem's box looks the same to the network."""

    def __init__(self, net, problem):
        super().__init__()
        spec = PROBLEMS[problem]
        self.net = net
        lo = torch.tensor([spec["x"][0], spec["t"][0]], dtype=torch.float64)
        hi = torch.tensor([spec["x"][1], spec["t"][1]], dtype=torch.float64)
        self.register_buffer("lo", lo)
        self.register_buffer("span", hi - lo)

    def forward(self, z):
        return self.net(2.0 * (z - self.lo) / self.span - 1.0)


# ------------------------------------------------------- residual vector, two formulations
def residual_rows(problem, model, weights=(W_IC, W_BC)):
    """``r(params, points) -> (N,)`` by torch.func, differentiable row by row.

    ``rows.physics["residual"]`` can be replaced to change the equation (continuation)."""
    spec = PROBLEMS[problem]
    physics = {"residual": spec["residual"]}
    buffers = dict(model.named_buffers())

    def u(params, z):
        return functional_call(model, (params, buffers), (z.unsqueeze(0),)).squeeze()

    du = grad(u, argnums=1)

    def ux(params, z):
        return du(params, z)[0]

    dux = grad(ux, argnums=1)

    def uxx(params, z):
        return dux(params, z)[0]

    duxx = grad(uxx, argnums=1)

    def pde_row(params, z):
        g = du(params, z)
        third = duxx(params, z)[0] if spec["order"] >= 3 else 0.0
        return physics["residual"](u(params, z), g[0], uxx(params, z), third, g[1])

    def scales(counts):
        n_col, n_ic, n_bc = counts
        return (
            1.0 / math.sqrt(n_col),
            math.sqrt(weights[0] / n_ic),
            math.sqrt(weights[1] / n_bc),
        )

    def rows(params, col, ic, bc, counts=None, shift=None):
        """The scaled residual vector. ``counts`` fixes the scaling when the points given are
        a subset of a larger set, so a row has the same value alone as within the set."""
        s_r, s_ic, s_bc = scales(counts or (col.shape[0], ic.shape[0], bc.shape[0]))
        empty = col.new_zeros(0)
        r = vmap(pde_row, in_dims=(None, 0))(params, col) if len(col) else empty
        v_ic, v_bc = empty, empty
        if len(ic):
            v_ic = vmap(u, in_dims=(None, 0))(params, ic) - spec["initial"](ic[:, 0])
        if len(bc):
            v_bc = vmap(u, in_dims=(None, 0))(params, bc) - spec["boundary"](bc[:, 0], bc[:, 1])
        if shift is not None:  # multipliers over the penalty, added to the constraint rows
            v_ic, v_bc = v_ic + shift[0], v_bc + shift[1]
        return torch.cat([s_r * r, s_ic * v_ic, s_bc * v_bc])

    def jacobian(params, col, ic, bc, counts=None, only=None):
        """``dr/dtheta`` (N, P): each row depends on one point, so one gradient per point.
        ``only`` restricts the columns to the named parameters."""
        s = scales(counts or (col.shape[0], ic.shape[0], bc.shape[0]))
        names = list(params) if only is None else list(only)
        chosen = {n: params[n] for n in names}
        blocks = []
        for fn, pts, scale in ((pde_row, col, s[0]), (u, ic, s[1]), (u, bc, s[2])):
            if pts.shape[0] == 0:
                continue

            def of_chosen(sub, z, fn=fn):
                return fn({**params, **sub}, z)

            per_point = vmap(grad(of_chosen, argnums=0), in_dims=(None, 0))(chosen, pts)
            flat = torch.cat([per_point[n].reshape(pts.shape[0], -1) for n in names], dim=1)
            blocks.append(scale * flat)
        return torch.cat(blocks)

    rows.physics = physics
    return rows, jacobian


def loss_autograd(problem, model, col, ic, bc):
    """The same sum of squares by batched reverse mode, the textbook way (for the baselines)."""
    spec = PROBLEMS[problem]
    z = col.clone().requires_grad_(True)
    u = model(z)[:, 0]
    g = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0].sum(), z, create_graph=True)[0][:, 0]
    third = 0.0
    if spec["order"] >= 3:
        third = torch.autograd.grad(uxx.sum(), z, create_graph=True)[0][:, 0]
    r = spec["residual"](u, g[:, 0], uxx, third, g[:, 1])
    v_ic = model(ic)[:, 0] - spec["initial"](ic[:, 0])
    v_bc = model(bc)[:, 0] - spec["boundary"](bc[:, 0], bc[:, 1])
    return (r**2).mean() + W_IC * (v_ic**2).mean() + W_BC * (v_bc**2).mean()


# ------------------------------------------------------------------------------ optimizers
class Clock:
    """Wall-clock time spent optimizing: evaluation of the error is not charged."""

    def __init__(self, model, points, values, every):
        self.model, self.every = model, every
        self.points = torch.tensor(points)
        self.values = values
        self.norm = float(np.linalg.norm(values))
        self.spent, self.mark, self.next = 0.0, time.perf_counter(), 0.0
        self.curve: list[tuple[float, float]] = []

    def tick(self, force=False):
        now = time.perf_counter()
        self.spent += now - self.mark
        if force or self.spent >= self.next:
            with torch.no_grad():
                pred = self.model(self.points)[:, 0].numpy()
            self.curve.append((self.spent, float(np.linalg.norm(pred - self.values) / self.norm)))
            self.next = self.spent + self.every
        self.mark = time.perf_counter()
        return self.spent


class Weights:
    """The model's parameters as one flat vector."""

    def __init__(self, model):
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for _, p in model.named_parameters()]
        self.sizes = [math.prod(s) for s in self.shapes]

    def read(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def tree(self, vec):
        out, i = {}, 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes, strict=True):
            out[name] = vec[i : i + size].reshape(shape)
            i += size
        return out

    def write(self, vec):
        with torch.no_grad():
            for p, value in zip(self.model.parameters(), self.tree(vec).values(), strict=True):
                p.copy_(value)


class Linearization:
    """``-(J^T J + lam I)^{-1} J^T rhs`` through whichever Gram matrix is smaller."""

    def __init__(self, J):
        self.J = J
        self.small = J.shape[0] <= J.shape[1]
        self.gram = J @ J.T if self.small else J.T @ J
        self.eye = torch.eye(self.gram.shape[0], dtype=self.gram.dtype)
        self.lam, self.factor = None, None

    def step(self, rhs, lam):
        if lam != self.lam:  # one factorization per damping, shared by every right-hand side
            shift = lam
            while True:  # a Gram matrix of low rank can defeat a very small damping
                factor, info = torch.linalg.cholesky_ex(self.gram + shift * self.eye)
                if int(info) == 0:
                    break
                shift = max(10.0 * shift, 1e-14 * float(self.gram.diagonal().mean()))
            self.lam, self.factor = lam, factor
        if self.small:
            return -self.J.T @ torch.cholesky_solve(rhs[:, None], self.factor)[:, 0]
        return -torch.cholesky_solve((self.J.T @ rhs)[:, None], self.factor)[:, 0]


def nielsen(lam, rho):
    """The damping after an accepted step of gain ratio ``rho``."""
    return max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-12)


def run_lm(problem, model, clock, budget, generator, resample_every=5):
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model)

    def residual(vec, pts):
        return rows(weights.tree(vec), *pts)

    lam, nu, it, stats = 1e-3, 2.0, 0, {"iterations": 0, "rejected": 0}
    pts = sample(problem, generator)
    while clock.tick() < budget:
        if it % resample_every == 0:
            pts = sample(problem, generator)
        r = residual(theta, pts)
        J = jacobian(weights.tree(theta), *pts)
        g = J.T @ r
        cost = 0.5 * float(r @ r)
        lin = Linearization(J)
        for _ in range(8):  # damping retries on the same linearization
            delta = lin.step(r, lam)
            r_new = residual(theta + delta, pts)
            new_cost = 0.5 * float(r_new @ r_new)
            predicted = 0.5 * float(delta @ (lam * delta - g))
            rho = (cost - new_cost) / max(predicted, 1e-300)
            if rho > 0 and math.isfinite(new_cost):
                theta = theta + delta
                lam, nu = nielsen(lam, rho), 2.0
                break
            lam, nu = lam * nu, 2.0 * nu
            stats["rejected"] += 1
        weights.write(theta)
        it += 1
    stats["iterations"] = it
    stats["rows"], stats["parameters"] = int(J.shape[0]), int(J.shape[1])
    return stats


def run_flow(problem, model, clock, budget, generator, resample_every=5, tol=0.2):
    """M1. Levenberg-Marquardt read as an ODE method. One damped step

        (J^T J + lam I) delta = -J^T r

    is one linearly implicit Euler step of the gradient flow ``theta' = -J^T r`` with step
    1/lam. So lam is set as an ODE code sets its step: from the local error, here how far the
    true residual lands from its linear model r + J delta, held near ``tol`` (relative to
    |r|). A second stage solves for that discrepancy with the factorization already at hand.

    (The flow ``theta' = -J^+ r`` with a fixed cut-off of the pseudo-inverse was tried first
    and does not move: the Jacobian has numerical rank near 100 of 2241, see the README.)"""
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model)

    def residual(vec, pts):
        return rows(weights.tree(vec), *pts)

    lam, it, stats = 1e-3, 0, {"iterations": 0, "rejected": 0, "corrected": 0}
    pts = sample(problem, generator)
    while clock.tick() < budget:
        if it % resample_every == 0:
            pts = sample(problem, generator)
        r = residual(theta, pts)
        J = jacobian(weights.tree(theta), *pts)
        lin = Linearization(J)
        norm = float(r.norm())
        for _ in range(8):
            delta = lin.step(r, lam)
            model_r = r + J @ delta
            r1 = residual(theta + delta, pts)
            n1 = float(r1.norm())
            err = float((r1 - model_r).norm()) / norm
            if math.isfinite(n1) and n1 < norm:
                cand = theta + delta
                cand2 = cand + lin.step(r1 - model_r, lam)
                n2 = float(residual(cand2, pts).norm())
                if math.isfinite(n2) and n2 < n1:
                    cand, stats["corrected"] = cand2, stats["corrected"] + 1
                theta = cand
                # the error falls as 1/lam^2, so the step-size rule of an order-1 method
                lam = max(lam * min(4.0, max(0.2, math.sqrt(err / tol))), 1e-12)
                break
            lam *= 4.0 if not math.isfinite(err) else min(10.0, max(2.0, math.sqrt(err / tol)))
            stats["rejected"] += 1
        weights.write(theta)
        it += 1
    stats["iterations"] = it
    return stats


def run_block(problem, model, clock, budget, generator, block=64, damping=1e-2, weighted=True):
    """M4. Block Kaczmarz with the block chosen by the residual. A pool of points is scored by
    one residual evaluation and cut into blocks of ``block`` rows, drawn without replacement
    in proportion to their squared residual (uniformly if not ``weighted``). Each block is
    satisfied in turn by a ``block x block`` solve at the current weights, damped by
    ``damping`` times the mean of its Gram matrix's diagonal. The merit is checked once per
    pool: if the fresh pool's cost rose, the weights go back and the damping triples."""
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model)
    counts = (N_COL, N_IC, N_BC)
    total = sum(counts)
    per_pool = max(1, total // (4 * block))  # a quarter of the pool is used, then a new one

    def split(pts, idx):
        a, b = N_COL, N_COL + N_IC
        return (
            pts[0][idx[idx < a]],
            pts[1][idx[(idx >= a) & (idx < b)] - a],
            pts[2][idx[idx >= b] - b],
        )

    it, stats = 0, {"iterations": 0, "rejected": 0, "block": block}
    kept, kept_cost = theta, math.inf
    while clock.tick() < budget:
        pts = sample(problem, generator)
        r = rows(weights.tree(theta), *pts)
        cost = float(r @ r)
        if not math.isfinite(cost) or cost > 2.0 * kept_cost:
            theta, damping = kept, damping * 3.0
            stats["rejected"] += 1
            r = rows(weights.tree(theta), *pts)
        else:
            if cost < kept_cost:
                damping = max(damping / 1.5, 1e-8)
            kept, kept_cost = theta, min(cost, kept_cost) if cost < kept_cost else kept_cost
        score = r**2 + 1e-300 if weighted else torch.ones_like(r)
        order = torch.multinomial(score, per_pool * block, replacement=False, generator=generator)
        for k in range(per_pool):
            if clock.tick() >= budget:
                break
            sub = split(pts, order[k * block : (k + 1) * block].sort().values)
            tree = weights.tree(theta)
            r_b = rows(tree, *sub, counts=counts)
            lin = Linearization(jacobian(tree, *sub, counts=counts))
            theta = theta + lin.step(r_b, damping * float(lin.gram.diagonal().mean()))
            weights.write(theta)
            it += 1
    stats["iterations"], stats["damping"] = it, damping
    return stats


def run_varpro(problem, model, clock, budget, generator, resample_every=5, project=True, warm=0):
    """M5. Variable projection. The output layer c enters u linearly, so it is solved, not
    trained: exactly for a linear equation, by Gauss-Newton sweeps on c alone otherwise. The
    hidden weights take a Levenberg-Marquardt step on the Jacobian with the range of the last
    layer's Jacobian projected out (Kaufman's approximation), and c is solved again."""
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model)
    last = weights.names[-2:]
    n_c = sum(weights.sizes[-2:])
    linear = bool(PROBLEMS[problem].get("linear"))
    sweeps = 1 if linear else 2

    def residual(vec, pts):
        return rows(weights.tree(vec), *pts)

    def solve_last(vec, pts):
        """c by least squares: one exact solve for a linear equation, damped Gauss-Newton
        sweeps on c alone otherwise (a 33 x 33 system, the damping raised until the cost
        falls)."""
        for _ in range(sweeps):
            r = residual(vec, pts)
            Jc = jacobian(weights.tree(vec), *pts, only=last)
            if linear:
                dc = torch.linalg.lstsq(Jc, -r[:, None], driver="gelsd").solution[:, 0]
                vec = vec.clone()
                vec[-n_c:] += dc
                continue
            lin, cost = Linearization(Jc), float(r @ r)
            scale = float(lin.gram.diagonal().mean())
            for damp in (1e-8, 1e-4, 1e-2, 1.0, 100.0):
                cand = vec.clone()
                cand[-n_c:] += lin.step(r, damp * scale)
                r_new = residual(cand, pts)
                if float(r_new @ r_new) < cost:
                    vec = cand
                    break
        return vec

    lam, nu, it, stats = 1e-3, 2.0, 0, {"iterations": 0, "rejected": 0}
    pts = sample(problem, generator)
    while clock.tick() < budget:
        plain = it < warm  # ordinary steps on every weight first
        if it % resample_every == 0:
            pts = sample(problem, generator)
            if not plain:
                theta = solve_last(theta, pts)
        r = residual(theta, pts)
        J = jacobian(weights.tree(theta), *pts)
        if plain:
            Jh, rh = J, r
        elif project:
            Q = torch.linalg.qr(J[:, -n_c:]).Q
            Jh = J[:, :-n_c] - Q @ (Q.T @ J[:, :-n_c])
            rh = r - Q @ (Q.T @ r)
        else:
            Jh, rh = J[:, :-n_c], r
        g = Jh.T @ rh
        cost = 0.5 * float(r @ r)
        lin = Linearization(Jh)
        for _ in range(8):
            delta = lin.step(rh, lam)
            cand = theta.clone()
            cand[: len(delta)] += delta
            if not plain:
                cand = solve_last(cand, pts)
            r_new = residual(cand, pts)
            new_cost = 0.5 * float(r_new @ r_new)
            predicted = 0.5 * float(delta @ (lam * delta - g))
            rho = (cost - new_cost) / max(predicted, 1e-300)
            if rho > 0 and math.isfinite(new_cost):
                theta = cand
                lam, nu = nielsen(lam, rho), 2.0
                break
            lam, nu = lam * nu, 2.0 * nu
            stats["rejected"] += 1
        weights.write(theta)
        it += 1
    stats["iterations"] = it
    return stats


def run_continue(problem, model, clock, budget, generator, stages=8, resample_every=5):
    """M6. Levenberg-Marquardt on a moving equation: the viscosity starts at an easy value and
    reaches the target in ``stages`` geometric steps. A stage ends when the cost falls under
    1e-5 or after 15 steps, and the target is forced at half the budget. (The tangent
    predictor of the plan is the first Gauss-Newton step at the new viscosity, so it needs no
    code of its own.)"""
    spec = PROBLEMS[problem]
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model)
    ratio = (spec["target"] / spec["easy"]) ** (1.0 / stages)
    mu, at_stage = spec["easy"], 0
    rows.physics["residual"] = spec["family"](mu)

    def residual(vec, pts):
        return rows(weights.tree(vec), *pts)

    lam, nu, it = 1e-3, 2.0, 0
    stats = {"iterations": 0, "rejected": 0, "reached_target_s": None}
    pts = sample(problem, generator)
    while (spent := clock.tick()) < budget:
        if it % resample_every == 0:
            pts = sample(problem, generator)
        r = residual(theta, pts)
        cost = 0.5 * float(r @ r)
        if mu > spec["target"] and (cost < 1e-5 or at_stage >= 15 or spent > 0.5 * budget):
            mu = spec["target"] if spent > 0.5 * budget else max(mu * ratio, spec["target"])
            if mu <= spec["target"] * (1 + 1e-12):
                mu, stats["reached_target_s"] = spec["target"], spent
                rows.physics["residual"] = spec["residual"]
            else:
                rows.physics["residual"] = spec["family"](mu)
            at_stage = 0
            r = residual(theta, pts)
            cost = 0.5 * float(r @ r)
        J = jacobian(weights.tree(theta), *pts)
        g = J.T @ r
        lin = Linearization(J)
        for _ in range(8):
            delta = lin.step(r, lam)
            r_new = residual(theta + delta, pts)
            new_cost = 0.5 * float(r_new @ r_new)
            predicted = 0.5 * float(delta @ (lam * delta - g))
            rho = (cost - new_cost) / max(predicted, 1e-300)
            if rho > 0 and math.isfinite(new_cost):
                theta = theta + delta
                lam, nu = nielsen(lam, rho), 2.0
                break
            lam, nu = lam * nu, 2.0 * nu
            stats["rejected"] += 1
        weights.write(theta)
        it, at_stage = it + 1, at_stage + 1
    stats["iterations"] = it
    return stats


def run_filter(
    problem, model, clock, budget, generator, batch=64, prior=1.0, forget=0.995, cap=1e12
):
    """M3. Training as filtering. Each step draws ``batch`` fresh rows (three quarters in the
    domain, the rest on the initial line and the boundary), never seen again, and applies an
    extended Kalman update with unit row noise:

        K = P J^T (J P J^T + I)^{-1},   theta -= K r,   P = (P - K J P) / forget

    P is the full covariance of the weights (the plan's low-rank form is the next step if
    this one works), with every variance capped at ``cap`` times the prior."""
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model)
    sizes = (batch - 2 * (batch // 8), batch // 8, batch // 8)
    unit = (1, 1, 1)  # rows unscaled: a stream has no total count
    P = prior * torch.eye(theta.numel(), dtype=theta.dtype)
    eye = torch.eye(batch, dtype=theta.dtype)
    it, stats = 0, {"iterations": 0, "rejected": 0, "batch": batch, "capped": 0}
    while clock.tick() < budget:
        pts = sample(problem, generator, sizes)
        tree = weights.tree(theta)
        r = rows(tree, *pts, counts=unit)
        J = jacobian(tree, *pts, counts=unit)
        PJt = P @ J.T
        gain = torch.linalg.solve(J @ PJt + eye, PJt.T).T
        cand = theta - gain @ r
        # The update should lower the residual of its own batch. One that more than doubles
        # it is the sign of a covariance gone bad: the step is refused and P starts again.
        after = float(rows(weights.tree(cand), *pts, counts=unit).norm())
        if math.isfinite(after) and after <= 2.0 * float(r.norm()):
            theta = cand
            P = (P - gain @ PJt.T) / forget
            P = 0.5 * (P + P.T)
            # Forgetting inflates directions the stream does not excite without bound, and
            # the filter then diverges (seen after 100 to 150 s, when P's range passes what
            # double precision holds). Rows and columns are scaled so no variance exceeds
            # ``cap`` times the prior. The growth itself is needed: a cap of 1 stalls it.
            shrink = torch.clamp(cap * prior / P.diagonal().clamp_min(1e-300), max=1.0).sqrt()
            if bool((shrink < 1.0).any()):
                P = shrink[:, None] * P * shrink[None, :]
                stats["capped"] += 1
        else:
            P = prior * torch.eye(theta.numel(), dtype=theta.dtype)
            stats["rejected"] += 1
        weights.write(theta)
        it += 1
    stats["iterations"] = it
    return stats


def run_al(problem, model, clock, budget, generator, penalty=1.0, multipliers=True, every=5):
    """M8. The initial and boundary conditions as constraints c(theta) = 0 on fixed points:

        1/2 |r_pde|^2 + lambda . c + penalty/2 |c|^2  =  1/2 |r_pde|^2 + penalty/2 |c + lambda/penalty|^2 + const

    so each inner problem is least squares with shifted constraint rows, solved by the same
    Levenberg-Marquardt steps, and ``lambda += penalty * c`` every ``every`` steps. Without
    ``multipliers`` it is Levenberg-Marquardt at loss weights equal to ``penalty``."""
    weights = Weights(model)
    theta = weights.read()
    rows, jacobian = residual_rows(problem, model, weights=(penalty, penalty))
    col, ic, bc = sample(problem, generator)  # the constraint points stay; col is redrawn
    shift = (torch.zeros(len(ic), dtype=theta.dtype), torch.zeros(len(bc), dtype=theta.dtype))

    def residual(vec, pts):
        return rows(weights.tree(vec), *pts, shift=shift)

    lam, nu, it, stats = 1e-3, 2.0, 0, {"iterations": 0, "rejected": 0}
    while clock.tick() < budget:
        if it % every == 0 and it:
            col = sample(problem, generator)[0]
            if multipliers:  # lambda/penalty += c, with c the unshifted constraint values
                empty = col[:0]
                c_ic = rows(weights.tree(theta), empty, ic, bc[:0], counts=(1, 1, 1))
                c_bc = rows(weights.tree(theta), empty, ic[:0], bc, counts=(1, 1, 1))
                root = math.sqrt(penalty)
                shift = (shift[0] + c_ic / root, shift[1] + c_bc / root)
        pts = (col, ic, bc)
        r = residual(theta, pts)
        J = jacobian(weights.tree(theta), *pts)
        g = J.T @ r
        cost = 0.5 * float(r @ r)
        lin = Linearization(J)
        for _ in range(8):
            delta = lin.step(r, lam)
            r_new = residual(theta + delta, pts)
            new_cost = 0.5 * float(r_new @ r_new)
            predicted = 0.5 * float(delta @ (lam * delta - g))
            rho = (cost - new_cost) / max(predicted, 1e-300)
            if rho > 0 and math.isfinite(new_cost):
                theta = theta + delta
                lam, nu = nielsen(lam, rho), 2.0
                break
            lam, nu = lam * nu, 2.0 * nu
            stats["rejected"] += 1
        weights.write(theta)
        it += 1
    stats["iterations"] = it
    stats["multiplier_rms"] = float(torch.cat(shift).pow(2).mean().sqrt()) * penalty
    return stats


def run_adam(problem, model, clock, budget, generator, resample_every=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    it, pts = 0, sample(problem, generator)
    while clock.tick() < budget:
        if it % resample_every == 0:
            pts = sample(problem, generator)
        optimizer.zero_grad()
        loss_autograd(problem, model, *pts).backward()
        optimizer.step()
        it += 1
    return {"iterations": it}


def run_lbfgs(problem, model, clock, budget, generator, resample_every=10):
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe"
    )
    it, pts = 0, sample(problem, generator)

    def closure():
        optimizer.zero_grad()
        loss = loss_autograd(problem, model, *pts)
        loss.backward()
        return loss

    while clock.tick() < budget:
        if it % resample_every == 0:
            pts = sample(problem, generator)
        optimizer.step(closure)
        it += 1
    return {"iterations": it}


def run_one(problem, method, seed, budget, width, depth, every):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1000 + seed)
    model = Scaler(make_net(width, depth, seed), problem)
    points, values = reference(problem)
    clock = Clock(model, points, values, every)
    clock.tick(force=True)
    if method == "lm":
        stats = run_lm(problem, model, clock, budget, generator)
    elif method == "adam":
        stats = run_adam(problem, model, clock, budget, generator)
    elif method == "lbfgs":
        stats = run_lbfgs(problem, model, clock, budget, generator)
    elif method == "flow":
        stats = run_flow(problem, model, clock, budget, generator)
    elif method.startswith("block"):
        stats = run_block(problem, model, clock, budget, generator, block=int(method[5:]))
    elif method.startswith("varpro") or method == "alternate":
        stats = run_varpro(
            problem,
            model,
            clock,
            budget,
            generator,
            project=method != "alternate",
            warm=int(method[6:] or 0) if method.startswith("varpro") else 0,
        )
    elif method.startswith("continue"):
        stats = run_continue(problem, model, clock, budget, generator, stages=int(method[8:]))
    elif method.startswith("filter"):
        size, _, exponent = method[6:].partition("c")  # filter64c4: variances capped at 1e4
        stats = run_filter(
            problem,
            model,
            clock,
            budget,
            generator,
            batch=int(size),
            cap=10.0 ** int(exponent or 12),
        )
    elif method in ("al", "lm_w1"):
        stats = run_al(problem, model, clock, budget, generator, multipliers=method == "al")
    else:
        assert method == "adam_lbfgs", method
        stats = run_adam(problem, model, clock, 0.3 * budget, generator)
        second = run_lbfgs(problem, model, clock, budget, generator)
        stats = {"iterations": stats["iterations"], "lbfgs_iterations": second["iterations"]}
    clock.tick(force=True)
    return {
        "problem": problem,
        "method": method,
        "seed": seed,
        "width": width,
        "depth": depth,
        "curve": clock.curve,
        "final": clock.curve[-1][1],
        "best": min(e for _, e in clock.curve),
        **stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--budget", type=float, default=60.0, help="seconds of optimization per run"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--every", type=float, default=2.0, help="seconds between error readings")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--fresh", action="store_true", help="discard saved runs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "budget_s": args.budget,
        "machine": platform.machine(),
        "system": platform.platform(),
        "torch": torch.__version__,
        "threads": 1,
        "cpus": os.cpu_count(),
        "points": [N_COL, N_IC, N_BC],
    }
    path = args.out / f"runs_{args.depth}x{args.width}.json"
    # A run is saved as soon as it ends, and an interrupted sweep resumes where it stopped.
    runs = json.loads(path.read_text())["runs"] if path.exists() and not args.fresh else []
    runs = [r for r in runs if r.get("budget_s") == args.budget]
    done = {(r["problem"], r["method"], r["seed"]) for r in runs}
    for problem in args.problems:
        for method in args.methods:
            for seed in range(args.seeds):
                if (problem, method, seed) in done:
                    continue
                if method.startswith("continue") and "family" not in PROBLEMS[problem]:
                    continue
                run = run_one(
                    problem, method, seed, args.budget, args.width, args.depth, args.every
                )
                run["budget_s"] = args.budget
                runs.append(run)
                path.write_text(json.dumps({"meta": meta, "runs": runs}))
                print(
                    f"{problem:<8} {method:<11} seed {seed}  final {run['final']:.2e}  "
                    f"best {run['best']:.2e}  iterations {run['iterations']}",
                    flush=True,
                )
    print("saved", path)


if __name__ == "__main__":
    main()
