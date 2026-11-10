"""I11 (Python side): where a PINN training step spends its time, and three ways to write it.

The problem every implementation of this experiment trains, in Python, C++ and Rust:

    u_t = alpha u_xx on [0, 1] x [0, 1], u(x, 0) = sin(pi x), u(0, t) = u(1, t) = 0, alpha = 0.1
    network 2 -> 64 -> 64 -> 64 -> 64 -> 1, tanh, Xavier-uniform weights, zero biases, float32
    loss = mean(r^2) + 10 mean(ic^2) + 10 mean(bc^2), 4096 collocation + 512 initial + 512 boundary
    points drawn uniformly at every step, Adam (1e-3, 0.9, 0.999, 1e-8)

Variants:

* ``pinnrl``: the package's trainer on the catalogue's heat problem with the same sizes;
* ``autograd``: a minimal PyTorch loop, u_t and u_xx by reverse-mode autograd (two
  ``torch.autograd.grad`` calls with ``create_graph``), the textbook PINN step;
* ``taylor``: the same loop with u, u_x, u_xx and u_t carried forward through the layers
  (Taylor mode, written out in tensor operations) and one backward pass for the weights.

Each is timed at 1 and 4 threads; ``autograd`` is also broken down by stage.

Usage: python experiments/I11_native_pinn/bench_python.py [--steps 3000] [--seeds 3]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)
HERE = Path(__file__).parent
ALPHA, WIDTH, DEPTH = 0.1, 64, 4
N_COL, N_IC, N_BC = 4096, 512, 512
W_IC, W_BC, LR = 10.0, 10.0, 1e-3


class SigmoidTanh(torch.nn.Module):
    """tanh(x) written as 2 sigmoid(2x) - 1: portable, and 2.5 times faster than torch.tanh here."""

    def forward(self, x):
        return 2.0 * torch.sigmoid(2.0 * x) - 1.0


_ACCELERATE = None


def _vvtanhf(out, x):
    global _ACCELERATE
    if _ACCELERATE is None:
        _ACCELERATE = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    n = ctypes.c_int(x.numel())
    _ACCELERATE.vvtanhf(
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(x.data_ptr()), ctypes.byref(n)
    )


class _AccelerateTanh(torch.autograd.Function):
    """tanh by Accelerate's vectorised vvtanhf (macOS), differentiable to any order."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        _vvtanhf(out, x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        (out,) = ctx.saved_tensors  # the output keeps its graph, so this differentiates again
        return grad * (1.0 - out * out)


class AccelerateTanh(torch.nn.Module):
    def forward(self, x):
        return _AccelerateTanh.apply(x)


ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "sigmoid-tanh": SigmoidTanh,
    "accelerate-tanh": AccelerateTanh,
}


def make_net(seed, activation="tanh"):
    torch.manual_seed(seed)
    layers, n_in = [], 2
    for _ in range(DEPTH):
        layers += [torch.nn.Linear(n_in, WIDTH), ACTIVATIONS[activation]()]
        n_in = WIDTH
    layers.append(torch.nn.Linear(n_in, 1))
    net = torch.nn.Sequential(*layers)
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return net


def rel_l2(net):
    g = torch.linspace(0, 1, 101)
    X, T = torch.meshgrid(g, g, indexing="ij")
    xt = torch.stack([X.reshape(-1), T.reshape(-1)], 1)
    exact = torch.exp(-ALPHA * math.pi**2 * xt[:, 1]) * torch.sin(math.pi * xt[:, 0])
    with torch.no_grad():
        pred = net(xt).reshape(-1)
    return float(torch.linalg.norm(pred - exact) / torch.linalg.norm(exact))


def sample():
    xt = torch.rand(N_COL, 2)
    x_ic = torch.rand(N_IC, 1)
    t_bc = torch.rand(N_BC, 1)
    side = (torch.arange(N_BC) % 2).float().reshape(-1, 1)
    return xt, x_ic, torch.cat([side, t_bc], 1)


def residual_autograd(net, xt):
    xt = xt.requires_grad_(True)
    u = net(xt)
    g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0].sum(), xt, create_graph=True)[0][:, 0]
    return g[:, 1] - ALPHA * u_xx


def residual_taylor(net, xt):
    """u_t - alpha u_xx with value, x-slope, x-curvature and t-slope propagated layer by layer."""
    n = xt.shape[0]
    h = xt
    hx = torch.zeros(n, 2)
    hx[:, 0] = 1.0
    ht = torch.zeros(n, 2)
    ht[:, 1] = 1.0
    hxx = torch.zeros(n, 2)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    for k, lin in enumerate(linears):
        W = lin.weight
        a, ax, axx, at = h @ W.T + lin.bias, hx @ W.T, hxx @ W.T, ht @ W.T
        if k == len(linears) - 1:
            return (at - ALPHA * axx).reshape(-1)
        s = torch.tanh(a)
        d1 = 1.0 - s * s
        d2 = -2.0 * s * d1
        h, hx, hxx, ht = s, d1 * ax, d2 * ax * ax + d1 * axx, d1 * at


def residual_stacked(net, xt):
    """Taylor mode with the four streams stacked into one (4N, width) matrix: one product a layer."""
    n = xt.shape[0]
    e = torch.zeros(3 * n, 2)
    e[:n, 0] = 1.0
    e[2 * n :, 1] = 1.0
    H = torch.cat([xt, e], 0)  # rows: value, d/dx, d2/dx2, d/dt
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    for k, lin in enumerate(linears):
        A = H @ lin.weight.T
        a, ax, axx, at = A[:n] + lin.bias, A[n : 2 * n], A[2 * n : 3 * n], A[3 * n :]
        if k == len(linears) - 1:
            return (at - ALPHA * axx).reshape(-1)
        s = torch.tanh(a)
        d1 = 1.0 - s * s
        H = torch.cat([s, d1 * ax, -2.0 * s * d1 * ax * ax + d1 * axx, d1 * at], 0)


RESIDUALS = {
    "autograd": residual_autograd,
    "taylor": residual_taylor,
    "taylor-stacked": residual_stacked,
}


def loss_fn(net, batch, residual):
    xt, x_ic, xt_bc = batch
    r = residual(net, xt)
    ic = net(torch.cat([x_ic, torch.zeros_like(x_ic)], 1)) - torch.sin(math.pi * x_ic)
    bc = net(xt_bc)
    return (r**2).mean() + W_IC * (ic**2).mean() + W_BC * (bc**2).mean()


def run_raw(variant, seed, steps, breakdown=False, activation="tanh"):
    net = make_net(seed, activation)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    residual = RESIDUALS[variant]
    stages = dict.fromkeys(("sample", "forward and loss", "backward", "optimizer"), 0.0)
    torch.manual_seed(1000 + seed)
    tic = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        batch = sample()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net, batch, residual)
        t2 = time.perf_counter()
        loss.backward()
        t3 = time.perf_counter()
        opt.step()
        t4 = time.perf_counter()
        if breakdown:
            for key, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3), strict=True):
                stages[key] += dt
    elapsed = time.perf_counter() - tic
    out = {
        "ms_per_step": 1e3 * elapsed / steps,
        "rel_l2": rel_l2(net),
        "loss": float(loss.detach()),
    }
    if breakdown:
        out["stages_ms"] = {k: 1e3 * v / steps for k, v in stages.items()}
    return out


def _tanh_accelerate(x):
    out = torch.empty_like(x)
    _vvtanhf(out, x.contiguous())
    return out


_MANUAL_TANH = torch.tanh


@torch.no_grad()
def manual_gradients(linears, batch):
    """The loss and its weight gradient with no autograd graph: the native programs' algorithm
    (Taylor-mode forward, hand-written reverse) in tensor operations. Fills ``.grad``."""
    xt, x_ic, xt_bc = batch
    n = xt.shape[0]
    h, hx, ht, hxx = xt, torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n, 2)
    hx[:, 0] = 1.0
    ht[:, 1] = 1.0
    tape = []
    for k, lin in enumerate(linears):
        W = lin.weight
        a, ax, axx, at = torch.addmm(lin.bias, h, W.T), hx @ W.T, hxx @ W.T, ht @ W.T
        if k == len(linears) - 1:
            tape.append((h, hx, hxx, ht, None, None, None, None))
            break
        s = _MANUAL_TANH(a)
        d1 = 1.0 - s * s
        d2 = -2.0 * s * d1
        tape.append((h, hx, hxx, ht, s, ax, axx, at))
        h, hx, hxx, ht = s, d1 * ax, d2 * ax * ax + d1 * axx, d1 * at
    r = at - ALPHA * axx
    loss = (r * r).mean()
    g_t = (2.0 / n) * r
    g, gx, gxx, gt = None, None, -ALPHA * g_t, g_t
    grads = []
    for k in range(len(linears) - 1, -1, -1):
        h, hx, hxx, ht = tape[k][:4]
        gW = gxx.T @ hxx + gt.T @ ht
        if gx is not None:
            gW += gx.T @ hx
        gb = None
        if g is not None:
            gW += g.T @ h
            gb = g.sum(0)
        grads.append((gW, gb))
        if k == 0:
            break
        W = linears[k].weight
        s, ax, axx, at = tape[k - 1][4:]
        d1 = 1.0 - s * s
        d2 = -2.0 * s * d1
        d3 = -2.0 * d1 * (1.0 - 3.0 * s * s)
        ghxx, ght = gxx @ W, gt @ W
        ga = ghxx * (d3 * ax * ax + d2 * axx) + ght * d2 * at
        gax = ghxx * (2.0 * d2 * ax)
        if gx is not None:
            ghx = gx @ W
            ga += ghx * d2 * ax
            gax += ghx * d1
        if g is not None:
            ga += (g @ W) * d1
        g, gx, gxx, gt = ga, gax, ghxx * d1, ght * d1
    grads.reverse()
    # initial and boundary points: value only
    v_in = torch.cat([torch.cat([x_ic, torch.zeros_like(x_ic)], 1), xt_bc], 0)
    target = torch.cat([torch.sin(math.pi * x_ic), torch.zeros(xt_bc.shape[0], 1)], 0)
    coef = torch.cat(
        [
            torch.full((x_ic.shape[0], 1), W_IC / x_ic.shape[0]),
            torch.full((xt_bc.shape[0], 1), W_BC / xt_bc.shape[0]),
        ],
        0,
    )
    acts = [v_in]
    for k, lin in enumerate(linears):
        a = torch.addmm(lin.bias, acts[-1], lin.weight.T)
        if k < len(linears) - 1:
            acts.append(_MANUAL_TANH(a))
    e = a - target
    loss = loss + (coef * e * e).sum()
    g = 2.0 * coef * e
    for k in range(len(linears) - 1, -1, -1):
        gW, gb = grads[k]
        gW += g.T @ acts[k]
        gb = g.sum(0) if gb is None else gb + g.sum(0)
        linears[k].weight.grad, linears[k].bias.grad = gW, gb
        if k > 0:
            g = (g @ linears[k].weight) * (1.0 - acts[k] * acts[k])
    return loss


def run_manual(seed, steps, activation="tanh"):
    global _MANUAL_TANH
    _MANUAL_TANH = torch.tanh if activation == "tanh" else _tanh_accelerate
    net = make_net(seed)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    torch.manual_seed(1000 + seed)
    tic = time.perf_counter()
    for _ in range(steps):
        loss = manual_gradients(linears, sample())
        opt.step()
    elapsed = time.perf_counter() - tic
    return {"ms_per_step": 1e3 * elapsed / steps, "rel_l2": rel_l2(net), "loss": float(loss)}


def run_pinnrl(seed, steps):
    from pinnrl.pdes.catalog import from_catalog

    torch.manual_seed(seed)
    pde = from_catalog("heat")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [WIDTH] * DEPTH, "layer_norm": False},
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
    tic = time.perf_counter()
    result = trainer.train_steps(num_steps=steps, resample_every=1)
    wall = time.perf_counter() - tic
    err = rel_l2(trainer.model)
    return {
        "ms_per_step": 1e3 * float(result["train_time_s"]) / steps,
        "ms_per_step_wall": 1e3 * wall / steps,
        "rel_l2": err,
        "loss": float(result["loss"][-1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--long-steps", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    variants = [
        ("autograd", "tanh"),
        ("autograd", "sigmoid-tanh"),
        ("autograd", "accelerate-tanh"),
        ("taylor", "tanh"),
        ("taylor-stacked", "tanh"),
        ("manual", "tanh"),
        ("manual", "accelerate-tanh"),
        ("pinnrl", "tanh"),
    ]
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for variant, activation in variants:
            for seed in range(args.seeds):
                if variant == "pinnrl":
                    r = run_pinnrl(seed, args.steps)
                elif variant == "manual":
                    r = run_manual(seed, args.steps, activation)
                else:
                    r = run_raw(variant, seed, args.steps, activation=activation)
                name = f"python-{variant}" + ("" if activation == "tanh" else f" + {activation}")
                rows.append(
                    {"impl": name, "threads": threads, "seed": seed, "steps": args.steps, **r}
                )
                print(
                    f"{name:<36} threads={threads} seed={seed}  {r['ms_per_step']:.2f} ms/step  rel L2 {r['rel_l2']:.2e}",
                    flush=True,
                )
        for activation in ("tanh", "accelerate-tanh"):
            b = run_raw("autograd", 0, min(args.steps, 1000), breakdown=True, activation=activation)
            rows.append({"impl": f"breakdown {activation}", "threads": threads, **b})
            print(
                f"breakdown {activation} threads={threads}:",
                {k: round(v, 3) for k, v in b["stages_ms"].items()},
                flush=True,
            )
    torch.set_num_threads(4)
    for activation in ("tanh", "accelerate-tanh"):
        r = run_raw("autograd", 0, args.long_steps, activation=activation)
        rows.append(
            {
                "impl": f"python-autograd + {activation} long",
                "threads": 4,
                "seed": 0,
                "steps": args.long_steps,
                **r,
            }
        )
        print(
            f"long {activation}: {r['ms_per_step']:.2f} ms/step  rel L2 {r['rel_l2']:.2e}",
            flush=True,
        )
    (args.out / "python.json").write_text(json.dumps(rows, indent=1))
    print("saved", args.out / "python.json")


if __name__ == "__main__":
    main()
