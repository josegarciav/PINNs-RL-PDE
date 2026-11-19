"""I17: one PINN training step in pinnrl and in the alternatives, on the same problems.

Is the compiled kernel the fastest way to take a PINN step on a CPU? The same step is written
in every framework that could claim it:

    network 2 -> 64 -> 64 -> 64 -> 64 -> 1, tanh, float32
    loss = mean(r^2) + 10 mean(ic^2) + 10 mean(bc^2)
    4096 collocation + 512 initial + 512 boundary points (Dirichlet), Adam at 1e-3

on three residuals of rising order:

    heat            u_t - 0.1 u_xx
    kdv             u_t + 6 u u_x + u_xxx
    cahn_hilliard   u_t - (6 u u_x^2 + (3 u^2 - 1) u_xx - 0.01 u_xxxx)

Arms (each in its own process, so one framework's threads and compiler do not touch another's):

    pinnrl            the package's trainer, derivatives="auto" (the compiled kernel)
    pinnrl-autograd   the package's trainer, derivatives="autograd"
    torch-eager       a minimal PyTorch loop, nested torch.autograd.grad (the textbook step)
    torch-compile     the same loop under torch.compile
    torch-func        torch.func: nested forward-mode jvp for the input derivatives, one reverse pass
    jax-reverse       JAX, jit: vmap of nested grad (reverse over reverse)
    jax-forward       JAX, jit: nested jvp (forward mode) for the input derivatives, one reverse pass
    jax-jet           JAX, jit: jax.experimental.jet (Taylor mode), one reverse pass
    deepxde           DeepXDE with its PyTorch backend, dde.grad.jacobian / hessian

Every arm is timed at one thread and at the framework's default threading: the best of
``--rounds`` rounds of ``--steps`` steps, after a warm-up that includes any compilation. The
passes over the arms are repeated ``--passes`` times and each arm keeps its best.

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

# name -> (pinnrl equation, x-domain, t-domain, highest x-derivative)
PROBLEMS = {
    "heat": ("u_t - 0.1*u_xx", (0.0, 1.0), (0.0, 1.0), 2),
    "kdv": ("u_t + 6*u*u_x + u_xxx", (-15.0, 15.0), (0.0, 2.0), 3),
    "cahn_hilliard": (
        "u_t - (6*u*u_x^2 + (3*u^2 - 1)*u_xx - 0.01*u_xxxx)",
        (-1.0, 1.0),
        (0.0, 1.0),
        4,
    ),
}
ARMS = [
    "pinnrl",
    "pinnrl-autograd",
    "torch-eager",
    "torch-compile",
    "torch-func",
    "jax-reverse",
    "jax-forward",
    "jax-jet",
    "deepxde",
]


def residual(problem, u, d, u_t):
    """The residual from the value, the x-derivatives ``d[1..]`` and ``u_t`` (any array type)."""
    if problem == "heat":
        return u_t - 0.1 * d[2]
    if problem == "kdv":
        return u_t + 6.0 * u * d[1] + d[3]
    return u_t - (6.0 * u * d[1] ** 2 + (3.0 * u**2 - 1.0) * d[2] - 0.01 * d[4])


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
    equation, xdom, tdom, _ = PROBLEMS[problem]
    torch.manual_seed(0)
    pde = CustomPDE(
        equation,
        domain={"x": xdom, "t": tdom},
        initial_condition="exp(-x^2)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
    )
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [WIDTH] * DEPTH, "layer_norm": False},
        derivatives=mode,
        num_collocation_points=N_COL,
        batch_size=N_COL,
        num_boundary_points=N_BC,
        num_initial_points=N_IC,
        learning_rate=LR,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": W_BC, "initial": W_IC},
    )
    active = pde.native_active(trainer.model)
    assert active == (mode == "auto"), "the kernel must run in the auto arm and only there"
    trainer.train_steps(num_steps=30, resample_every=1)
    best = float("inf")
    for _ in range(rounds):
        tic = time.perf_counter()
        trainer.train_steps(num_steps=steps, resample_every=1)
        best = min(best, 1e3 * (time.perf_counter() - tic) / steps)
    return best, {"native_active": active}


# ------------------------------------------------------------------------------ torch
def _torch_net():
    import torch

    torch.manual_seed(0)
    layers, n_in = [], 2
    for _ in range(DEPTH):
        layers += [torch.nn.Linear(n_in, WIDTH), torch.nn.Tanh()]
        n_in = WIDTH
    layers.append(torch.nn.Linear(n_in, 1))
    return torch.nn.Sequential(*layers)


def _torch_points(problem):
    import torch

    _, (x0, x1), (t0, t1), _ = PROBLEMS[problem]
    lo, span = torch.tensor([x0, t0]), torch.tensor([x1 - x0, t1 - t0])
    xt = lo + span * torch.rand(N_COL, 2)
    ic = torch.cat([x0 + (x1 - x0) * torch.rand(N_IC, 1), torch.full((N_IC, 1), t0)], 1)
    side = x0 + (x1 - x0) * (torch.arange(N_BC) % 2).float().reshape(-1, 1)
    bc = torch.cat([side, t0 + (t1 - t0) * torch.rand(N_BC, 1)], 1)
    return xt, ic, bc


def _residual_torch_eager(problem, net, xt):
    import torch

    order = PROBLEMS[problem][3]
    xt = xt.requires_grad_(True)
    u = net(xt)[:, 0]
    g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    d, u_t = {1: g[:, 0]}, g[:, 1]
    for k in range(2, order + 1):
        d[k] = torch.autograd.grad(d[k - 1].sum(), xt, create_graph=True)[0][:, 0]
    return residual(problem, u, d, u_t)


def _residual_torch_func(problem, net, xt):
    import torch
    from torch.func import jvp

    order = PROBLEMS[problem][3]
    ex = torch.zeros_like(xt)
    ex[:, 0] = 1.0
    et = torch.zeros_like(xt)
    et[:, 1] = 1.0

    def value(z):
        return net(z)[:, 0]

    # forward mode nested ``order`` times along x: each level returns the next derivative,
    # and its primal output carries the lower ones
    def level(k):
        if k == 0:
            return lambda z: (value(z),)
        below = level(k - 1)

        def f(z):
            primal, tangent = jvp(below, (z,), (ex,))
            return (*primal, tangent[-1])

        return f

    u, *derivs = level(order)(xt)
    _, u_t = jvp(value, (xt,), (et,))
    return residual(problem, u, dict(enumerate(derivs, start=1)), u_t)


def arm_torch(problem, kind, steps, rounds):
    import torch

    net = _torch_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    # the two formulations must give the same residual before either is timed
    xt = _torch_points(problem)[0]
    r_eager = _residual_torch_eager(problem, net, xt.clone())
    r_func = _residual_torch_func(problem, net, xt.clone())
    assert torch.allclose(r_eager, r_func, rtol=2e-3, atol=2e-4), "torch arms disagree"
    res = _residual_torch_func if kind == "func" else _residual_torch_eager

    def loss_fn(xt, ic, bc):
        r = res(problem, net, xt)
        u_ic = net(ic)[:, 0] - torch.exp(-(ic[:, 0] ** 2))
        return (r**2).mean() + W_IC * (u_ic**2).mean() + W_BC * (net(bc) ** 2).mean()

    if kind == "compile":
        loss_fn = torch.compile(loss_fn)

    def step():
        xt, ic, bc = _torch_points(problem)
        optimizer.zero_grad()
        loss_fn(xt, ic, bc).backward()
        optimizer.step()

    return best_of(step, steps, rounds), {}


# -------------------------------------------------------------------------------- jax
def arm_jax(problem, kind, steps, rounds):
    import jax
    import jax.numpy as jnp
    from jax.experimental import jet

    _, (x0, x1), (t0, t1), order = PROBLEMS[problem]
    key = jax.random.PRNGKey(0)
    sizes = [2] + [WIDTH] * DEPTH + [1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        key, sub = jax.random.split(key)
        limit = (6.0 / (a + b)) ** 0.5
        params.append((jax.random.uniform(sub, (b, a), minval=-limit, maxval=limit), jnp.zeros(b)))

    def net(params, z):  # z (N, 2) -> (N,)
        h = z
        for w, b in params[:-1]:
            h = jnp.tanh(h @ w.T + b)
        w, b = params[-1]
        return (h @ w.T + b)[:, 0]

    def point(params, x, t):
        return net(params, jnp.stack([x, t])[None, :])[0]

    def res_reverse(params, xt):
        def one(x, t):
            fs = [lambda x, t: point(params, x, t)]
            for _ in range(order):
                fs.append(jax.grad(fs[-1], argnums=0))
            d = {k: fs[k](x, t) for k in range(1, order + 1)}
            return residual(problem, fs[0](x, t), d, jax.grad(fs[0], argnums=1)(x, t))

        return jax.vmap(one)(xt[:, 0], xt[:, 1])

    ex = jnp.zeros((N_COL, 2)).at[:, 0].set(1.0)
    et = jnp.zeros((N_COL, 2)).at[:, 1].set(1.0)

    def res_forward(params, xt):
        def level(k):
            if k == 0:
                return lambda z: (net(params, z),)
            below = level(k - 1)

            def f(z):
                primal, tangent = jax.jvp(below, (z,), (ex,))
                return (*primal, tangent[-1])

            return f

        u, *derivs = level(order)(xt)
        _, u_t = jax.jvp(lambda z: net(params, z), (xt,), (et,))
        return residual(problem, u, dict(enumerate(derivs, start=1)), u_t)

    def res_jet(params, xt):
        series = [ex] + [jnp.zeros_like(ex)] * (order - 1)
        u, derivs = jet.jet(lambda z: net(params, z), (xt,), (series,))
        _, u_t = jax.jvp(lambda z: net(params, z), (xt,), (et,))
        return residual(problem, u, dict(enumerate(derivs, start=1)), u_t)

    res = {"reverse": res_reverse, "forward": res_forward, "jet": res_jet}[kind]
    xt0 = jnp.array([x0, t0]) + jnp.array([x1 - x0, t1 - t0]) * jax.random.uniform(key, (N_COL, 2))
    reference = res_reverse(params, xt0)
    assert jnp.allclose(res(params, xt0), reference, rtol=2e-3, atol=2e-4), "jax arms disagree"

    def loss_fn(params, xt, ic, bc):
        r = res(params, xt)
        u_ic = net(params, ic) - jnp.exp(-(ic[:, 0] ** 2))
        return (r**2).mean() + W_IC * (u_ic**2).mean() + W_BC * (net(params, bc) ** 2).mean()

    @jax.jit
    def step(params, m, v, i, key):
        key, k1, k2, k3 = jax.random.split(key, 4)
        lo, span = jnp.array([x0, t0]), jnp.array([x1 - x0, t1 - t0])
        xt = lo + span * jax.random.uniform(k1, (N_COL, 2))
        ic = jnp.stack([x0 + (x1 - x0) * jax.random.uniform(k2, (N_IC,)), jnp.full(N_IC, t0)], 1)
        side = x0 + (x1 - x0) * (jnp.arange(N_BC) % 2)
        bc = jnp.stack([side, t0 + (t1 - t0) * jax.random.uniform(k3, (N_BC,))], 1)
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
        p, m, v, k = step(state[0], state[1], state[2], float(state[4]), state[3])
        p[0][0].block_until_ready()
        state[:4] = [p, m, v, k]

    return best_of(one_step, steps, rounds), {"jax": jax.__version__}


# ---------------------------------------------------------------------------- deepxde
def arm_deepxde(problem, steps, rounds):
    import deepxde as dde
    import numpy as np

    _, (x0, x1), (t0, t1), order = PROBLEMS[problem]
    geometry = dde.geometry.GeometryXTime(
        dde.geometry.Interval(x0, x1), dde.geometry.TimeDomain(t0, t1)
    )

    def pde(x, y):
        d = {1: dde.grad.jacobian(y, x, i=0, j=0)}
        if order >= 2:
            d[2] = dde.grad.hessian(y, x, i=0, j=0)
        for k in range(3, order + 1):
            d[k] = dde.grad.jacobian(d[k - 1], x, i=0, j=0)
        return residual(problem, y, d, dde.grad.jacobian(y, x, i=0, j=1))

    conditions = [
        dde.icbc.DirichletBC(geometry, lambda x: 0.0, lambda _, on_boundary: on_boundary),
        dde.icbc.IC(
            geometry, lambda x: np.exp(-(x[:, 0:1] ** 2)), lambda _, on_initial: on_initial
        ),
    ]
    data = dde.data.TimePDE(
        geometry, pde, conditions, num_domain=N_COL, num_boundary=N_BC, num_initial=N_IC
    )
    net = dde.nn.FNN([2] + [WIDTH] * DEPTH + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=LR, loss_weights=[1.0, W_BC, W_IC])
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
    if arm == "pinnrl":
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
            for threads in (1, 0):  # 0: the framework's default threading
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
