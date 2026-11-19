"""I17: one PINN training step in pinnrl and in the alternatives, on the same problems.

Is the package's compiled step the fastest way to take a PINN step on a CPU? The same step is
written in every framework that could claim it:

    network d -> 64 -> 64 -> 64 -> 64 -> k, tanh, float32
    loss = mean(r^2) + 10 mean(ic^2) + 10 mean(bc^2)
    4096 collocation + 512 initial + 512 boundary points (Dirichlet), Adam at 1e-3

on nine problems (see PROBLEMS): second order (heat, Burgers, Allen-Cahn), third (KdV, and BBM
with its mixed u_xxt), fourth (Cahn-Hilliard, Kuramoto-Sivashinsky), a two-field system (the
nonlinear Schrodinger equation as real and imaginary parts) and two space dimensions (heat).

Arms (each in its own process, so one framework's threads and compiler do not touch another's):

    pinnrl            the package's trainer, default settings: the step compiled per equation
    pinnrl-general    the same with PINNRL_NATIVE_SPECIALIZE=off: the general kernel
    pinnrl-autograd   the package's trainer, derivatives="autograd"
    torch-eager       a minimal PyTorch loop, nested torch.autograd.grad (the textbook step)
    torch-compile     the same loop under torch.compile
    torch-func        torch.func: nested forward-mode jvp for the input derivatives, one reverse pass
    jax-reverse       JAX, jit: vmap of nested grad (reverse over reverse)
    jax-forward       JAX, jit: nested jvp (forward mode) for the input derivatives, one reverse pass
    jax-jet           JAX, jit: jax.experimental.jet (Taylor mode), one reverse pass
    deepxde           DeepXDE with its PyTorch backend, dde.grad.jacobian

Every arm is timed at one thread: the best of ``--rounds`` rounds of ``--steps`` steps, after a
warm-up that includes any compilation. The passes over the arms are repeated ``--passes`` times
and each arm keeps its best.

Usage:
    uv run --with jax --with deepxde python experiments/I17_step_head_to_head/run.py
An arm whose framework is not installed, or that fails, is recorded with its error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
WIDTH, DEPTH = 64, 4
N_COL, N_IC, N_BC = 4096, 512, 512
W_IC, W_BC, LR = 10.0, 10.0, 1e-3

# Each problem: the equations as pinnrl reads them, the fields, the domain (space, then t), the
# derivative chains every arm follows (a chain differentiates along its coordinates in turn and
# yields every prefix), and the residuals from a dict of arrays keyed "u", "u_x", "u_txx" (the
# coordinates of a derivative sorted, as pinnrl spells them).
PROBLEMS = {
    "heat": dict(
        label="heat",
        equations=["u_t - 0.1*u_xx"],
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        chains=[("x", "x"), ("t",)],
        residual=lambda d: [d["u_t"] - 0.1 * d["u_xx"]],
    ),
    "burgers": dict(
        label="Burgers",
        equations=["u_t + u*u_x - 0.01*u_xx"],
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        chains=[("x", "x"), ("t",)],
        residual=lambda d: [d["u_t"] + d["u"] * d["u_x"] - 0.01 * d["u_xx"]],
    ),
    "allen_cahn": dict(
        label="Allen–Cahn",
        equations=["u_t - 0.0001*u_xx + 5*u^3 - 5*u"],
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        chains=[("x", "x"), ("t",)],
        residual=lambda d: [d["u_t"] - 0.0001 * d["u_xx"] + 5.0 * d["u"] ** 3 - 5.0 * d["u"]],
    ),
    "kdv": dict(
        label="KdV",
        equations=["u_t + 6*u*u_x + u_xxx"],
        domain={"x": (-15.0, 15.0), "t": (0.0, 2.0)},
        chains=[("x", "x", "x"), ("t",)],
        residual=lambda d: [d["u_t"] + 6.0 * d["u"] * d["u_x"] + d["u_xxx"]],
    ),
    "bbm": dict(
        label="BBM",
        equations=["u_t + u_x + u*u_x - u_xxt"],
        domain={"x": (-15.0, 15.0), "t": (0.0, 2.0)},
        chains=[("x", "x", "t"), ("t",)],
        forward_chains=[("x", "x", "t")],  # forward mode gets u_t on the way to u_xxt
        residual=lambda d: [d["u_t"] + d["u_x"] + d["u"] * d["u_x"] - d["u_txx"]],
    ),
    "cahn_hilliard": dict(
        label="Cahn–Hilliard",
        equations=["u_t - (6*u*u_x^2 + (3*u^2 - 1)*u_xx - 0.01*u_xxxx)"],
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        chains=[("x", "x", "x", "x"), ("t",)],
        residual=lambda d: [
            d["u_t"]
            - (
                6.0 * d["u"] * d["u_x"] ** 2
                + (3.0 * d["u"] ** 2 - 1.0) * d["u_xx"]
                - 0.01 * d["u_xxxx"]
            )
        ],
    ),
    "kuramoto_sivashinsky": dict(
        label="Kuramoto–Sivashinsky",
        equations=["u_t + u*u_x + u_xx + u_xxxx"],
        domain={"x": (-10.0, 10.0), "t": (0.0, 1.0)},
        chains=[("x", "x", "x", "x"), ("t",)],
        residual=lambda d: [d["u_t"] + d["u"] * d["u_x"] + d["u_xx"] + d["u_xxxx"]],
    ),
    "schrodinger": dict(
        label="Schrödinger (2 fields)",
        equations=["u_t + 0.5*v_xx + (u^2 + v^2)*v", "v_t - 0.5*u_xx - (u^2 + v^2)*u"],
        fields=("u", "v"),
        domain={"x": (-5.0, 5.0), "t": (0.0, 1.5)},
        chains=[("x", "x"), ("t",)],
        residual=lambda d: [
            d["u_t"] + 0.5 * d["v_xx"] + (d["u"] ** 2 + d["v"] ** 2) * d["v"],
            d["v_t"] - 0.5 * d["u_xx"] - (d["u"] ** 2 + d["v"] ** 2) * d["u"],
        ],
    ),
    "heat_2d": dict(
        label="heat, 2-D",
        equations=["u_t - 0.1*(u_xx + u_yy)"],
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)},
        chains=[("x", "x"), ("y", "y"), ("t",)],
        residual=lambda d: [d["u_t"] - 0.1 * (d["u_xx"] + d["u_yy"])],
    ),
}
ARMS = [
    "pinnrl",
    "pinnrl-general",
    "pinnrl-autograd",
    "torch-eager",
    "torch-compile",
    "torch-func",
    "jax-reverse",
    "jax-forward",
    "jax-jet",
    "deepxde",
]


def layout(problem):
    """Fields, coordinate names, their columns and the box ``(lo, hi)`` per column."""
    spec = PROBLEMS[problem]
    names = list(spec["domain"])
    return spec.get("fields", ("u",)), names, [spec["domain"][c] for c in names]


def as_dict(problem, derivs):
    """``{sorted coordinates: (N, k) array}`` to the named columns the residuals read."""
    fields = layout(problem)[0]
    out = {}
    for key, value in derivs.items():
        for j, name in enumerate(fields):
            out[name + ("_" + "".join(sorted(key)) if key else "")] = value[:, j]
    return out


def forward_chains(value_fn, chains, direction, jvp):
    """Forward mode nested along each chain: ``z -> {sorted coordinates: (N, k) array}``.

    Each level is one ``jvp`` of the level below along the next coordinate; its primal output
    carries the lower derivatives and its tangent the new ones, so a chain of length ``m``
    costs ``m`` nested jvps and yields every prefix (and, for a mixed chain, the mixed lower
    ones as well).
    """

    def build(chain):
        keys, fn = [()], (lambda z: (value_fn(z),))
        for c in chain:
            new = [tuple(sorted(k + (c,))) for k in keys]
            keep = [i for i, k in enumerate(new) if k not in keys and new.index(k) == i]

            def level(z, below=fn, c=c, keep=keep):
                primal, tangent = jvp(below, z, direction(c))
                return (*primal, *(tangent[i] for i in keep))

            keys, fn = keys + [new[i] for i in keep], level
        return keys, fn

    built = [build(chain) for chain in chains]

    def derivatives(z):
        out = {}
        for keys, fn in built:
            for key, value in zip(keys, fn(z), strict=True):
                out.setdefault(key, value)
        return out

    return derivatives


def best_of(step, steps, rounds, warmup=30):
    for _ in range(warmup):
        step()
    best = float("inf")
    for _ in range(rounds):
        tic = time.perf_counter()
        for _ in range(steps):
            step()
        best = min(best, 1e3 * (time.perf_counter() - tic) / steps)
    return best


# ----------------------------------------------------------------------------- pinnrl
def arm_pinnrl(problem, mode, steps, rounds):
    import logging

    import torch

    from pinnrl import CustomPDE

    logging.disable(logging.WARNING)
    spec = PROBLEMS[problem]
    fields, names, _ = layout(problem)
    n_space = len(names) - 1
    torch.manual_seed(0)
    pde = CustomPDE(
        spec["equations"] if len(fields) > 1 else spec["equations"][0],
        fields=list(fields),
        domain=spec["domain"],
        initial_condition=dict.fromkeys(fields, "exp(-x^2)"),
        boundary_conditions={"dirichlet": {"value": 0.0}},
    )
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [WIDTH] * DEPTH, "layer_norm": False},
        derivatives=mode,
        num_collocation_points=N_COL,
        batch_size=N_COL,
        num_boundary_points=N_BC // (2 * n_space),  # per face: N_BC boundary points in all
        num_initial_points=N_IC,
        learning_rate=LR,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": W_BC, "initial": W_IC},
    )
    active = pde.native_active(trainer.model)
    assert active == (mode == "auto"), "the kernel must run in the kernel arms and only there"
    trainer.train_steps(num_steps=100, resample_every=1)  # past the switch to the compiled step
    best = float("inf")
    for _ in range(rounds):
        tic = time.perf_counter()
        trainer.train_steps(num_steps=steps, resample_every=1)
        best = min(best, 1e3 * (time.perf_counter() - tic) / steps)
    return best, {"native_active": active, "specialised": "_step_lib" in pde.__dict__}


# ------------------------------------------------------------------------------ torch
def _torch_net(problem):
    import torch

    fields, names, _ = layout(problem)
    torch.manual_seed(0)
    layers, n_in = [], len(names)
    for _ in range(DEPTH):
        layers += [torch.nn.Linear(n_in, WIDTH), torch.nn.Tanh()]
        n_in = WIDTH
    layers.append(torch.nn.Linear(n_in, len(fields)))
    return torch.nn.Sequential(*layers)


def _torch_points(problem):
    import torch

    _, names, box = layout(problem)
    d, n_space = len(names), len(names) - 1
    lo = torch.tensor([b[0] for b in box])
    span = torch.tensor([b[1] - b[0] for b in box])
    xt = lo + span * torch.rand(N_COL, d)
    ic = lo + span * torch.rand(N_IC, d)
    ic[:, -1] = lo[-1]
    bc = lo + span * torch.rand(N_BC, d)
    index = torch.arange(N_BC)
    axis, side = index % n_space, (index // n_space) % 2
    bc[index, axis] = lo[axis] + span[axis] * side.float()
    return xt, ic, bc


def _residual_torch_eager(problem, net, xt):
    import torch

    spec = PROBLEMS[problem]
    fields, names, _ = layout(problem)
    if not xt.requires_grad:
        xt = xt.requires_grad_(True)
    derivs = {(): net(xt)}
    gradients = {}  # the full input gradient of each derivative, one backward pass per field

    def along(key, c):
        if key not in gradients:
            gradients[key] = [
                torch.autograd.grad(derivs[key][:, j].sum(), xt, create_graph=True)[0]
                for j in range(len(fields))
            ]
        return torch.stack([g[:, names.index(c)] for g in gradients[key]], dim=1)

    for chain in spec["chains"]:
        key = ()
        for c in chain:
            new = tuple(sorted(key + (c,)))
            if new not in derivs:
                derivs[new] = along(key, c)
            key = new
    return torch.stack(spec["residual"](as_dict(problem, derivs)), dim=1)


def _residual_torch_func(problem, net, xt):
    import torch
    from torch.func import jvp

    spec = PROBLEMS[problem]
    names = layout(problem)[1]

    def direction(c):
        v = torch.zeros_like(xt)
        v[:, names.index(c)] = 1.0
        return v

    derivatives = forward_chains(
        net,
        spec.get("forward_chains", spec["chains"]),
        direction,
        lambda f, z, v: jvp(f, (z,), (v,)),
    )
    return torch.stack(spec["residual"](as_dict(problem, derivatives(xt))), dim=1)


def arm_torch(problem, kind, steps, rounds):
    import torch

    net = _torch_net(problem)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    # the two formulations must give the same residual before either is timed
    xt = _torch_points(problem)[0]
    r_eager = _residual_torch_eager(problem, net, xt.clone())
    r_func = _residual_torch_func(problem, net, xt.clone())
    assert (r_eager - r_func).norm() < 5e-3 * r_eager.norm() + 1e-4, "torch arms disagree"
    res = _residual_torch_func if kind == "func" else _residual_torch_eager

    def loss_fn(xt, ic, bc):
        r = res(problem, net, xt)
        u_ic = net(ic) - torch.exp(-(ic[:, 0:1] ** 2))
        return (r**2).mean() + W_IC * (u_ic**2).mean() + W_BC * (net(bc) ** 2).mean()

    if kind == "compile":
        loss_fn = torch.compile(loss_fn, fullgraph=True)  # no silent fall back to eager

    def step():
        xt, ic, bc = _torch_points(problem)
        if kind != "func":
            xt.requires_grad_(True)  # outside the compiled function
        optimizer.zero_grad()
        loss_fn(xt, ic, bc).backward()
        optimizer.step()

    return best_of(step, steps, rounds), {}


# -------------------------------------------------------------------------------- jax
def arm_jax(problem, kind, steps, rounds):
    import jax
    import jax.numpy as jnp
    from jax.experimental import jet

    spec = PROBLEMS[problem]
    fields, names, box = layout(problem)
    d, k, n_space = len(names), len(fields), len(names) - 1
    lo = jnp.array([b[0] for b in box])
    span = jnp.array([b[1] - b[0] for b in box])
    key = jax.random.PRNGKey(0)
    sizes = [d] + [WIDTH] * DEPTH + [k]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        key, sub = jax.random.split(key)
        limit = (6.0 / (a + b)) ** 0.5
        params.append((jax.random.uniform(sub, (b, a), minval=-limit, maxval=limit), jnp.zeros(b)))

    def net(params, z):  # z (N, d) -> (N, k)
        h = z
        for w, b in params[:-1]:
            h = jnp.tanh(h @ w.T + b)
        w, b = params[-1]
        return h @ w.T + b

    def direction(c):
        return jnp.zeros((N_COL, d)).at[:, names.index(c)].set(1.0)

    def along(g, col):
        return lambda z: jax.grad(g)(z)[col]

    def derivs_reverse(params, xt):
        def component(j):
            return lambda z: net(params, z[None, :])[0, j]

        out = {(): net(params, xt)}
        for chain in spec["chains"]:
            fns, key_ = [component(j) for j in range(k)], ()
            for c in chain:
                col = names.index(c)
                fns = [along(g, col) for g in fns]
                key_ = tuple(sorted(key_ + (c,)))
                if key_ not in out:
                    out[key_] = jnp.stack([jax.vmap(g)(xt) for g in fns], axis=1)
        return out

    def derivs_forward(params, xt):
        return forward_chains(
            lambda z: net(params, z),
            spec.get("forward_chains", spec["chains"]),
            direction,
            lambda f, z, v: jax.jvp(f, (z,), (v,)),
        )(xt)

    def derivs_jet(params, xt):
        out = {}
        for chain in spec.get("forward_chains", spec["chains"]):
            run = 1
            while run < len(chain) and chain[run] == chain[0]:
                run += 1
            if run < 2:  # nothing for Taylor mode to do: plain forward mode
                part = forward_chains(
                    lambda z: net(params, z),
                    [chain],
                    direction,
                    lambda f, z, v: jax.jvp(f, (z,), (v,)),
                )(xt)
            else:  # the run of one coordinate by jet, the rest by jvp over it
                head = chain[0]

                def taylor(z, head=head, run=run):
                    series = [direction(head)] + [jnp.zeros((N_COL, d))] * (run - 1)
                    value, higher = jet.jet(lambda y: net(params, y), (z,), (series,))
                    return (value, *higher)

                keys = [tuple([head] * m) for m in range(run + 1)]
                fn = taylor
                for c in chain[run:]:
                    new = [tuple(sorted(key_ + (c,))) for key_ in keys]

                    def level(z, below=fn, c=c):
                        primal, tangent = jax.jvp(below, (z,), (direction(c),))
                        return (*primal, *tangent)

                    keys, fn = keys + new, level
                part = dict(zip(keys, fn(xt), strict=True))
            for key_, value in part.items():
                out.setdefault(key_, value)
        return out

    derivs = {"reverse": derivs_reverse, "forward": derivs_forward, "jet": derivs_jet}[kind]

    def res(params, xt, how=derivs):
        return jnp.stack(spec["residual"](as_dict(problem, how(params, xt))), axis=1)

    xt0 = lo + span * jax.random.uniform(key, (N_COL, d))
    reference = res(params, xt0, derivs_reverse)
    gap = jnp.linalg.norm(res(params, xt0) - reference)
    assert gap < 5e-3 * jnp.linalg.norm(reference) + 1e-4, "jax arms disagree"

    def loss_fn(params, xt, ic, bc):
        r = res(params, xt)
        u_ic = net(params, ic) - jnp.exp(-(ic[:, 0:1] ** 2))
        return (r**2).mean() + W_IC * (u_ic**2).mean() + W_BC * (net(params, bc) ** 2).mean()

    @jax.jit
    def step(params, m, v, i, key):
        key, k1, k2, k3 = jax.random.split(key, 4)
        xt = lo + span * jax.random.uniform(k1, (N_COL, d))
        ic = (lo + span * jax.random.uniform(k2, (N_IC, d))).at[:, -1].set(lo[-1])
        index = jnp.arange(N_BC)
        axis, side = index % n_space, (index // n_space) % 2
        bc = lo + span * jax.random.uniform(k3, (N_BC, d))
        bc = bc.at[index, axis].set(lo[axis] + span[axis] * side)
        g = jax.grad(loss_fn)(params, xt, ic, bc)
        m = jax.tree_util.tree_map(lambda m, g: 0.9 * m + 0.1 * g, m, g)
        v = jax.tree_util.tree_map(lambda v, g: 0.999 * v + 0.001 * g * g, v, g)
        scale = LR * jnp.sqrt(1 - 0.999**i) / (1 - 0.9**i)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - scale * m / (jnp.sqrt(v) + 1e-8), params, m, v
        )
        return params, m, v, key

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = [params, zeros, zeros, key, 0]

    def one_step():
        state[4] += 1
        p, m, v, k_ = step(state[0], state[1], state[2], float(state[4]), state[3])
        p[0][0].block_until_ready()
        state[:4] = [p, m, v, k_]

    return best_of(one_step, steps, rounds), {"jax": jax.__version__}


# ---------------------------------------------------------------------------- deepxde
def arm_deepxde(problem, steps, rounds):
    import deepxde as dde
    import numpy as np

    spec = PROBLEMS[problem]
    fields, names, box = layout(problem)
    n_space = len(names) - 1
    if n_space == 1:
        space = dde.geometry.Interval(*box[0])
    else:
        space = dde.geometry.Rectangle([b[0] for b in box[:-1]], [b[1] for b in box[:-1]])
    geometry = dde.geometry.GeometryXTime(space, dde.geometry.TimeDomain(*box[-1]))

    def pde(x, y):
        named = {}
        for j, name in enumerate(fields):
            named[name] = y[:, j : j + 1]
            for chain in spec["chains"]:
                current, key = named[name], ()
                for c in chain:
                    key = tuple(sorted(key + (c,)))
                    label = name + "_" + "".join(key)
                    if label not in named:
                        named[label] = dde.grad.jacobian(current, x, i=0, j=names.index(c))
                    current = named[label]
        return spec["residual"](named)

    conditions = [
        dde.icbc.DirichletBC(
            geometry, lambda x: 0.0, lambda _, on_boundary: on_boundary, component=j
        )
        for j in range(len(fields))
    ] + [
        dde.icbc.IC(
            geometry,
            lambda x: np.exp(-(x[:, 0:1] ** 2)),
            lambda _, on_initial: on_initial,
            component=j,
        )
        for j in range(len(fields))
    ]
    data = dde.data.TimePDE(
        geometry, pde, conditions, num_domain=N_COL, num_boundary=N_BC, num_initial=N_IC
    )
    net = dde.nn.FNN([len(names)] + [WIDTH] * DEPTH + [len(fields)], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    weights = [1.0] * len(spec["equations"]) + [W_BC] * len(fields) + [W_IC] * len(fields)
    model.compile("adam", lr=LR, loss_weights=weights)
    model.train(iterations=30, display_every=10**9, verbose=0)
    best = float("inf")
    for _ in range(rounds):  # the slope between two run lengths removes train()'s fixed cost
        tic = time.perf_counter()
        model.train(iterations=steps // 4, display_every=10**9, verbose=0)
        mid = time.perf_counter()
        model.train(iterations=steps + steps // 4, display_every=10**9, verbose=0)
        end = time.perf_counter()
        best = min(best, 1e3 * ((end - mid) - (mid - tic)) / steps)
    return best, {"deepxde": dde.__version__, "points": "fixed (no resampling)"}


def run_arm(arm, problem, threads, steps, rounds):
    if arm.startswith(("pinnrl", "torch", "deepxde")):
        import torch

        if threads:
            torch.set_num_threads(threads)
    if arm in ("pinnrl", "pinnrl-general"):
        return arm_pinnrl(problem, "auto", steps, rounds)
    if arm == "pinnrl-autograd":
        return arm_pinnrl(problem, "autograd", steps, rounds)
    if arm.startswith("torch-"):
        return arm_torch(problem, arm.split("-")[1], steps, rounds)
    if arm.startswith("jax-"):
        return arm_jax(problem, arm.split("-")[1], steps, rounds)
    return arm_deepxde(problem, steps, rounds)


def child_env(arm, threads):
    env = dict(os.environ, TQDM_DISABLE="1", DDE_BACKEND="pytorch", JAX_PLATFORMS="cpu")
    env.pop("PINNRL_NATIVE_SPECIALIZE", None)
    if arm == "pinnrl-general":
        env["PINNRL_NATIVE_SPECIALIZE"] = "off"
    if threads:
        for name in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(threads)
        if threads == 1 and arm.startswith("jax"):
            env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    parser.add_argument("--threads", nargs="+", type=int, default=[1], help="0: the default")
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--child", nargs=3, metavar=("ARM", "PROBLEM", "THREADS"))
    args = parser.parse_args()
    if args.child:
        arm, problem, threads = args.child
        try:
            ms, info = run_arm(arm, problem, int(threads), args.steps, args.rounds)
            print("RESULT " + json.dumps({"ms_per_step": ms, **info}))
        except Exception as error:  # noqa: BLE001 - an arm that cannot run is a result
            text = f"{type(error).__name__}: {error}".strip().splitlines()
            print("RESULT " + json.dumps({"error": " ".join(text[:2])[:400]}))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    rows: dict[tuple[str, str, int], dict] = {}
    for sweep in range(args.passes):
        for problem in args.problems:
            for threads in args.threads:  # 0: the framework's default threading
                for arm in args.arms:
                    key = (arm, problem, threads)
                    if "error" in rows.get(key, {}):
                        continue
                    command = [sys.executable, __file__, "--child", arm, problem, str(threads)]
                    command += ["--steps", str(args.steps), "--rounds", str(args.rounds)]
                    done = subprocess.run(
                        command, env=child_env(arm, threads), capture_output=True, text=True
                    )
                    lines = [x for x in done.stdout.splitlines() if x.startswith("RESULT ")]
                    got = (
                        json.loads(lines[-1][7:])
                        if lines
                        else {
                            "error": (done.stderr.strip().splitlines() or ["no output"])[-1][:400]
                        }
                    )
                    old = rows.get(key)
                    if old is None or got.get("ms_per_step", 1e9) < old.get("ms_per_step", 1e9):
                        rows[key] = got
                    shown = (
                        f"{got['ms_per_step']:.2f} ms"
                        if "ms_per_step" in got
                        else got["error"][:90]
                    )
                    label = "1 thread" if threads else "default threads"
                    print(f"pass {sweep} {problem:<14} {label:<16} {arm:<16} {shown}", flush=True)
    out = [
        {"arm": arm, "problem": problem, "threads": threads or "default", **row}
        for (arm, problem, threads), row in rows.items()
    ]
    meta = {
        "machine": platform.machine(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "steps": args.steps,
        "rounds": args.rounds,
        "passes": args.passes,
    }
    if sys.platform == "darwin":  # low power mode halves every timing
        power = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
        meta["low_power_mode"] = "lowpowermode         1" in power
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        pass
    (args.out / "steps.json").write_text(json.dumps({"meta": meta, "rows": out}, indent=1))
    print("saved", args.out / "steps.json")


if __name__ == "__main__":
    main()
