"""The filter run by steps, not time, with what each step does recorded (README findings 51-53).

Filter as ``filter64g25`` (batch 64, prior 1, memory max(20, k/25), cap 1e12), seed 0. Per
step: the step length |dtheta|; the largest predicted row variance J_i P J_i^T and where its
row sits; the batch residual before and after; trace and largest diagonal of P; the largest
and root-mean-square change of u on a fixed probe of 4,000 points. The error every 100 steps.

usage: uv run python trace.py STEPS OUT.npz [mode] [problem] [stream] [delta]
  mode:   plain (P / f) | process (P + (1/f - 1) I) | directional | trust (a step limit on
          the change in u of delta = 0.25 times its scale; P never reset)
  stream: uniform | rad (residual pool) | radw (the same, importance-weighted rows)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run as d10  # noqa: E402

steps, out = int(sys.argv[1]), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "plain"
problem = sys.argv[4] if len(sys.argv) > 4 else "burgers_sharp"
stream = sys.argv[5] if len(sys.argv) > 5 else "uniform"
seed = 0
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(1000 + seed)
model = d10.Scaler(d10.make_net(32, 3, seed), problem)
points, values = d10.reference(problem)
grid = torch.tensor(points)
norm = float(np.linalg.norm(values))

weights = d10.Weights(model)
theta = weights.read()
rows, jacobian = d10.residual_rows(problem, model, (d10.W_IC, d10.W_BC))
batch, prior, cap, grow = 64, 1.0, 1e12, (20, 1 / 25)
sizes = (batch - 2 * (batch // 8), batch // 8, batch // 8)
unit = (1, 1, 1)
n = theta.numel()
P = prior * torch.eye(n, dtype=theta.dtype)
eye = torch.eye(batch, dtype=theta.dtype)
log = {
    k: []
    for k in (
        "dtheta",
        "smax",
        "smax_x",
        "smax_t",
        "smax_kind",
        "front",
        "before",
        "after",
        "trace",
        "pmax",
        "accepted",
        "du_max",
        "du_rms",
    )
}
probe = d10.sample(problem, torch.Generator().manual_seed(7), (4000, 0, 0))[0]


def predict(vec):
    with torch.no_grad():
        return torch.func.functional_call(model, weights.tree(vec), (probe,))[:, 0]


errors = []
tries = 0
DELTA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.25
# The solution's scale from its initial line: the network's own output is near 0 at first.
scale0 = float(d10.PROBLEMS[problem]["initial"](probe[:, 0]).abs().max())
pool = d10.ResidualPool(problem, model, generator) if stream in ("rad", "radw") else None
scale = torch.ones(batch, dtype=theta.dtype)
spent = 0.0
for it in range(steps):
    start = time.perf_counter()
    pts = d10.sample(problem, generator, sizes)
    if pool is not None:
        col, importance = pool.draw(sizes[0], replacement=False)
        pts = (col, *pts[1:])
        if stream == "radw":
            scale[: sizes[0]] = importance.sqrt()
    tree = weights.tree(theta)
    r = scale * rows(tree, *pts, counts=unit)
    J = scale[:, None] * jacobian(tree, *pts, counts=unit)
    PJt = P @ J.T
    s = (J * PJt.T).sum(1)
    i = int(s.argmax())
    allpts = torch.cat(pts)
    # mode "trust": a step may change u on the probe by at most DELTA times u's largest
    # size there; if not, the rows' noise is raised 4 times and the step tried again, up to
    # 4^5; if all fail the batch is skipped and P kept (never reset).
    noises = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0) if mode == "trust" else (1.0,)
    if mode == "trust":
        u_now = predict(theta)
        bound = DELTA * max(float(u_now.abs().max()), scale0)
    for noise in noises:
        gain, info = torch.linalg.solve_ex(J @ PJt + noise * eye, PJt.T)
        gain = gain.T
        delta = gain @ r
        cand = theta - delta
        after = float((scale * rows(weights.tree(cand), *pts, counts=unit)).norm())
        accepted = int(info) == 0 and math.isfinite(after) and after <= 2.0 * float(r.norm())
        if mode == "trust" and accepted:
            accepted = float((predict(cand) - u_now).abs().max()) <= bound
        if accepted:
            break
        tries += 1
    col = pts[0]
    front = bool(((col[:, 0].abs() < 0.03) & (col[:, 1] > 0.35)).any())
    du = predict(cand) - predict(theta)
    log["du_max"].append(float(du.abs().max()))
    log["du_rms"].append(float(du.pow(2).mean().sqrt()))
    log["dtheta"].append(float(delta.norm()))
    log["smax"].append(float(s[i]))
    log["smax_x"].append(float(allpts[i, 0]))
    log["smax_t"].append(float(allpts[i, 1]))
    log["smax_kind"].append(0 if i < sizes[0] else (1 if i < sizes[0] + sizes[1] else 2))
    log["front"].append(front)
    log["before"].append(float(r.norm()))
    log["after"].append(after)
    log["accepted"].append(accepted)
    if accepted:
        theta = cand
        f = 1.0 - 1.0 / max(grow[0], grow[1] * it)
        P = P - gain @ PJt.T
        if mode in ("plain", "trust"):
            P = P / f
        elif mode == "process":
            # Additive noise: the trace P would gain by forgetting at P = prior I, spread
            # evenly, so growth is linear, not exponential, in an unexcited direction.
            P = P + (1.0 / f - 1.0) * prior * torch.eye(n, dtype=theta.dtype)
        elif mode == "directional":
            # Forget only along what this batch measured: inflate P by 1/f on the range of
            # P J^T, leave the rest. P <- P + (1/f - 1) P J^T (J P J^T)^+ J P.
            PJt = P @ J.T
            S = J @ PJt
            P = P + (1.0 / f - 1.0) * PJt @ torch.linalg.pinv(S, hermitian=True, rtol=1e-10) @ PJt.T
        P = 0.5 * (P + P.T)
        shrink = torch.clamp(cap * prior / P.diagonal().clamp_min(1e-300), max=1.0).sqrt()
        if bool((shrink < 1.0).any()):
            P = shrink[:, None] * P * shrink[None, :]
    elif mode != "trust":
        P = prior * torch.eye(n, dtype=theta.dtype)
    log["trace"].append(float(P.diagonal().sum()))
    log["pmax"].append(float(P.diagonal().max()))
    weights.write(theta)
    spent += time.perf_counter() - start
    if it % 100 == 0 or it == steps - 1:
        with torch.no_grad():
            pred = model(grid)[:, 0].numpy()
        errors.append((it, float(np.linalg.norm(pred - values) / norm)))
        print(
            it,
            f"err {errors[-1][1]:.3g}  |dθ| {log['dtheta'][-1]:.3g}  smax {log['smax'][-1]:.3g}"
            f"  trP {log['trace'][-1]:.3g}  maxP {log['pmax'][-1]:.3g}",
            flush=True,
        )
print(f"retries {tries}  skipped {steps - sum(log['accepted'])}", flush=True)
print(f"seconds per step {spent / steps * 1e3:.2f} ms", flush=True)
np.savez(out, seconds=spent, errors=np.array(errors), **{k: np.array(v) for k, v in log.items()})
