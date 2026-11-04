"""K09: linearized stability and stiffness of the catalogue's systems and of the evaporator.

Part A, the catalogue's ODE systems (logistic, harmonic oscillator, pendulum,
Lotka-Volterra, Van der Pol, Lorenz, SIR, CSTR): the state-space form
``s_t = f(s)`` is read from the equations with SymPy, the reference trajectory
over the catalogue's window is integrated with ``solve_ivp``, and the Jacobian
``df/ds`` along it gives the spectrum: the fastest rate ``max |Re lambda|``,
the stiffness ratio ``max |Re| / min |Re|`` (eigenvalues with ``|Re| < 1e-8``
counted as zero), the number of fast units in the window ``T max |lambda|``,
the oscillation count ``T max |Im lambda| / 2 pi`` and the perturbation growth
``T max Re lambda``.

Part B, the same systems solved by a PINN with the package's clean recipe
(normalized to the trajectory's scales, 4 x 64, Adam 2,000 steps then L-BFGS
100), 2 seeds: the relative L2 error against the reference, to be read against
the numbers of Part A.

Part C, the evaporator's method-of-lines system (``FallingFilmEvaporator.mol_rhs``):
the Jacobian by central finite differences at the clean steady state for
``n_z`` in {26, 51, 101, 201, 401} and at the fouled state for 201; the
spectrum's clusters (transport at ``-v/dz``, the source's slow rates), the
stiffness ratio and the physical time scales (residence, evaporation, window).

Usage: python experiments/K09_linearized_stability/run.py [--seeds 2] [--steps 2000] [--workers 3] [--skip-pinn]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sympy
from scipy.integrate import solve_ivp

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
SYSTEMS = [
    "logistic",
    "harmonic_oscillator",
    "pendulum",
    "lotka_volterra",
    "van_der_pol",
    "lorenz",
    "sir",
    "cstr",
]
ZERO = 1e-8


def state_space(name):
    """``(state names, f(s) callable, jacobian callable, s0, (t0, t1), pde)`` of a catalogue ODE."""
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.pdes.symbolic import compile_expression

    pde = from_catalog(name)
    spec = pde.to_dict()
    eqs = spec["equations"]
    eqs = [eqs] if isinstance(eqs, str) else list(eqs)
    fields = list(pde.fields)
    params = {sympy.Symbol(k): float(v) for k, v in spec["parameters"].items()}
    ic = spec["initial_condition"]
    ic = {fields[0]: ic} if not isinstance(ic, dict) else ic
    state, f, s0 = [], [], []
    for field, text in zip(fields, eqs, strict=True):
        expr = compile_expression(text, fields, pde.coordinates, spec["parameters"]).expr
        expr = expr.subs(params)
        d1, d2 = sympy.Symbol(f"{field}_t"), sympy.Symbol(f"{field}_tt")
        u0 = float(sympy.sympify(str(ic[field])).subs(params))
        if d2 in expr.free_symbols:  # second order: state (u, u_t), zero initial velocity
            v = sympy.Symbol(f"{field}_t")
            rhs = sympy.solve(sympy.Eq(expr, 0), d2)[0]
            state += [sympy.Symbol(field), v]
            f += [v, rhs]
            s0 += [u0, 0.0]
        else:
            rhs = sympy.solve(sympy.Eq(expr, 0), d1)[0]
            state.append(sympy.Symbol(field))
            f.append(rhs)
            s0.append(u0)
    F = sympy.Matrix(f)
    J = F.jacobian(state)
    f_fn = sympy.lambdify(state, F, modules=[{"max": np.maximum, "Max": np.maximum}, "numpy"])
    j_fn = sympy.lambdify(state, J, modules=[{"max": np.maximum, "Max": np.maximum}, "numpy"])
    return (
        [str(s) for s in state],
        f_fn,
        j_fn,
        np.array(s0),
        tuple(map(float, pde.time_domain)),
        pde,
    )


def spectrum_along(name, n_samples=400):
    names, f_fn, j_fn, s0, (t0, t1), _ = state_space(name)
    sol = solve_ivp(
        lambda t, s: np.asarray(f_fn(*s), dtype=float).ravel(),
        (t0, t1),
        s0,
        t_eval=np.linspace(t0, t1, n_samples),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    eig = np.array([np.linalg.eigvals(np.asarray(j_fn(*s), dtype=float)) for s in sol.y.T])
    re, im = eig.real, np.abs(eig.imag)
    nonzero = np.abs(re) > ZERO
    T = t1 - t0
    fastest = float(np.abs(re).max())
    slowest = float(np.abs(re[nonzero]).min()) if nonzero.any() else math.nan
    return {
        "system": name,
        "state": names,
        "window": T,
        "fastest_rate": fastest,
        "slowest_rate": slowest,
        "stiffness_ratio": fastest / slowest if nonzero.any() else math.inf,
        "max_abs_eigenvalue": float(np.abs(eig).max()),
        "fast_units": T * float(np.abs(eig).max()),
        "oscillations": T * float(im.max()) / (2 * math.pi),
        "growth_exponent": T * float(re.max()),
        "max_real_part": float(re.max()),
        "zero_modes": int(round(float((~nonzero).sum(axis=1).mean()))),
        "t": sol.t.tolist(),
        "trajectory": sol.y.tolist(),
    }


def train_pinn(job):
    name, seed, steps = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    names, f_fn, j_fn, s0, (t0, t1), pde0 = state_space(name)
    t_eval = np.linspace(t0, t1, 401)
    sol = solve_ivp(
        lambda t, s: np.asarray(f_fn(*s), dtype=float).ravel(),
        (t0, t1),
        s0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    fields = list(pde0.fields)
    rows = [names.index(f) for f in fields]
    ref = sol.y[rows]  # (k, nt)
    scales = {
        f: (float(ref[i].mean()), float(max(ref[i].std(), 1e-2 * abs(ref[i]).max(), 1e-3)))
        for i, f in enumerate(fields)
    }
    pde = from_catalog(name, scales=scales)
    torch.manual_seed(seed)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        normalize=True,
        num_collocation_points=2048,
        batch_size=2048,
        num_boundary_points=64,
        num_initial_points=256,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    t_start = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=100, resample_every=50)
    xt = torch.tensor(np.stack([np.full(t_eval.size, 0.5), t_eval], 1), dtype=torch.float32)
    with torch.no_grad():
        pred = trainer.model(xt).numpy().T
    err = [
        float(np.linalg.norm(pred[i] - ref[i]) / np.linalg.norm(ref[i])) for i in range(len(fields))
    ]
    return {
        "system": name,
        "seed": seed,
        "steps": steps,
        "rel_l2": err,
        "rel_l2_mean": float(np.mean(err)),
        "train_time_s": time.time() - t_start,
        "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
    }


def evaporator_spectrum(nz, fouling=0.0):
    from pinnrl.process import FallingFilmEvaporator

    ev = FallingFilmEvaporator(R_f=fouling)
    ss = ev.steady_state(nz, t=0.0)
    s = np.concatenate([ss["W"], ss["X"]])
    rhs = ev.mol_rhs(nz)
    t0 = time.time()
    n = s.size
    J = np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(s[j]), 1e-3)
        sp, sm = s.copy(), s.copy()
        sp[j] += h
        sm[j] -= h
        J[:, j] = (rhs(0.0, sp) - rhs(0.0, sm)) / (2 * h)
    # the inlet entries are boundary values with zero dynamics: drop them
    keep = [i for i in range(n) if i not in (0, nz)]
    J = J[np.ix_(keep, keep)]
    eig = np.linalg.eigvals(J)
    re = eig.real
    dz = ev.length / (nz - 1)
    # The source terms alone (no transport): a 2 x 2 block per node, the physical rates.
    still = FallingFilmEvaporator(R_f=fouling, velocity=0.0)
    rhs_source = still.mol_rhs(nz)
    J_s = np.zeros((n, n))
    for j in range(n):
        h = 1e-6 * max(abs(s[j]), 1e-3)
        sp, sm = s.copy(), s.copy()
        sp[j] += h
        sm[j] -= h
        J_s[:, j] = (rhs_source(0.0, sp) - rhs_source(0.0, sm)) / (2 * h)
    eig_s = np.linalg.eigvals(J_s[np.ix_(keep, keep)])
    q_in = float(ev.heat_flux(np.array([0.1]), 0.0)[0])
    return {
        "nz": nz,
        "fouling": fouling,
        "max_real_part": float(re.max()),
        "fastest_rate": float(np.abs(re).max()),
        "slowest_rate": float(np.abs(re[np.abs(re) > ZERO]).min()),
        "stiffness_ratio": float(np.abs(re).max() / np.abs(re[np.abs(re) > ZERO]).min()),
        "transport_rate": ev.velocity / dz,
        "disk_radius": float(np.abs(eig + ev.velocity / dz).max()),
        "source_rate_max": float(np.abs(eig_s.real).max()),
        "source_rate_median": float(np.median(np.abs(eig_s.real))),
        "source_max_real_part": float(eig_s.real.max()),
        "max_imag": float(np.abs(eig.imag).max()),
        "residence_time_s": ev.residence_time(),
        "evaporation_time_s": 0.1 / (ev.perimeter * q_in / ev.latent_heat) / ev.velocity,
        "jacobian_time_s": time.time() - t0,
        "eigenvalues_real": re.tolist(),
        "eigenvalues_imag": eig.imag.tolist(),
        "source_eigenvalues_real": eig_s.real.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = {"ode": [], "pinn": [], "evaporator": []}
    for name in SYSTEMS:
        r = spectrum_along(name)
        result["ode"].append(r)
        print(
            f"{name:<20} T={r['window']:<5g} fastest {r['fastest_rate']:.3g} stiffness {r['stiffness_ratio']:.3g} "
            f"fast units {r['fast_units']:.3g} oscillations {r['oscillations']:.2f} growth {r['growth_exponent']:+.2f} zero modes {r['zero_modes']}",
            flush=True,
        )
    for nz in (26, 51, 101, 201, 401):
        r = evaporator_spectrum(nz)
        result["evaporator"].append(r)
        print(
            f"evaporator nz={nz:<4} max Re {r['max_real_part']:.3g} fastest {r['fastest_rate']:.3g} (v/dz {r['transport_rate']:.3g}, "
            f"disk {r['disk_radius']:.3g}) slowest {r['slowest_rate']:.3g} source rates {r['source_rate_max']:.3g}/{r['source_rate_median']:.3g} "
            f"(max Re {r['source_max_real_part']:+.3g}) {r['jacobian_time_s']:.1f}s",
            flush=True,
        )
    r = evaporator_spectrum(201, fouling=2e-4)
    result["evaporator"].append(r)
    print(
        f"evaporator nz=201 fouled: fastest {r['fastest_rate']:.3g} slowest {r['slowest_rate']:.3g} stiffness {r['stiffness_ratio']:.3g}"
    )
    if not args.skip_pinn:
        jobs = [(n, s, args.steps) for s in range(args.seeds) for n in SYSTEMS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(train_pinn, jobs):
                result["pinn"].append(rec)
                print(
                    f"pinn {rec['system']:<20} seed={rec['seed']} rel L2 {np.round(rec['rel_l2'], 4)} "
                    f"{rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                    flush=True,
                )
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print("saved", args.out / "summary.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
