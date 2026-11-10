"""I11: build and run the C++ and Rust trainers; collect their timings.

Builds ``cpp/pinn.cpp`` (clang++) and ``rust/pinn.rs`` (rustc, no crates) into ``build/``,
checks their gradients against PyTorch, then trains the heat problem with each kernel.

Usage: python experiments/I11_native_pinn/run.py [--steps 3000] [--seeds 3] [--rustc PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def build(rustc):
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    subprocess.run(
        ["clang++", "-O3", "-mcpu=native", "-std=c++17", "-DACCELERATE_NEW_LAPACK",
         str(HERE / "cpp" / "pinn.cpp"), "-framework", "Accelerate", "-o", str(out / "pinn_cpp")],
        check=True,
    )  # fmt: skip
    binaries = [out / "pinn_cpp"]
    if rustc:
        subprocess.run(
            [rustc, "-C", "opt-level=3", "-C", "target-cpu=native", "-C", "panic=abort",
             str(HERE / "rust" / "pinn.rs"), "-o", str(out / "pinn_rust")],
            check=True,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip
        binaries.append(out / "pinn_rust")
    else:
        print("rustc not found: the Rust trainer is skipped (pass --rustc PATH)")
    return binaries


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--naive-steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--long-steps", type=int, default=20000)
    parser.add_argument("--rustc", default=shutil.which("rustc"))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binaries = build(args.rustc)
    subprocess.run(
        [sys.executable, str(HERE / "check_gradients.py"), *map(str, binaries)], check=True
    )
    rows = []

    def run(binary, mode, steps, seed, env=None, tag=""):
        res = subprocess.run(
            [str(binary), "--mode", mode, "--steps", str(steps), "--seed", str(seed)],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        row = json.loads(res.stdout)
        row["impl"] += tag
        rows.append(row)
        print(
            f"{row['impl']:<22} seed={seed} steps={steps}  {row['ms_per_step']:.2f} ms/step  rel L2 {row['rel_l2']:.2e}",
            flush=True,
        )

    for binary in binaries:
        for seed in range(args.seeds):
            run(binary, "blas", args.steps, seed)
        for seed in range(args.seeds):
            run(binary, "blas", args.steps, seed, {"VECLIB_MAXIMUM_THREADS": "1"}, " (1 thread)")
        for seed in range(args.seeds):
            run(binary, "naive", args.naive_steps, seed)
        run(binary, "blas", args.long_steps, 0, tag=" long")
    (args.out / "native.json").write_text(json.dumps(rows, indent=1))
    print("saved", args.out / "native.json")


if __name__ == "__main__":
    main()
