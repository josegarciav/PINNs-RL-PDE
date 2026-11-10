"""I11: the native implementations' loss and gradient against PyTorch autograd on one batch.

Writes random weights and a batch to a binary file, runs each native binary with ``--check``
and compares its loss and weight gradient with ``torch.autograd`` in float64.

Usage: python experiments/I11_native_pinn/check_gradients.py build/pinn_cpp build/pinn_rust
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import bench_python as B  # noqa: E402


def main():
    binaries = sys.argv[1:]
    torch.manual_seed(7)
    net = B.make_net(7)
    with torch.no_grad():  # move off the zero-bias start so every gradient entry is exercised
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    xt, x_ic, xt_bc = B.sample()
    val = torch.zeros(B.N_IC + B.N_BC, 4)
    val[: B.N_IC, 0] = x_ic[:, 0]
    val[: B.N_IC, 2] = torch.sin(math.pi * x_ic[:, 0])
    val[: B.N_IC, 3] = 2 * B.W_IC / B.N_IC
    val[B.N_IC :, :2] = xt_bc
    val[B.N_IC :, 3] = 2 * B.W_BC / B.N_BC
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    blob = b"".join(
        t.detach().numpy().astype("<f4").tobytes()
        for lin in linears
        for t in (lin.weight, lin.bias)
    )
    blob += xt.numpy().astype("<f4").tobytes() + val.numpy().astype("<f4").tobytes()

    net64 = B.make_net(7).double()
    net64.load_state_dict({k: v.double() for k, v in net.state_dict().items()})
    loss = B.loss_fn(net64, (xt.double(), x_ic.double(), xt_bc.double()), B.residual_autograd)
    loss.backward()
    ref = np.concatenate(
        [
            t.grad.numpy().ravel()
            for lin in [m for m in net64 if isinstance(m, torch.nn.Linear)]
            for t in (lin.weight, lin.bias)
        ]
    )
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        fin = Path(tmp) / "in.bin"
        fin.write_bytes(blob)
        for binary in binaries:
            for mode in ("blas", "naive"):
                fout = Path(tmp) / "out.bin"
                subprocess.run([binary, "--mode", mode, "--check", str(fin), str(fout)], check=True)
                out = np.frombuffer(fout.read_bytes(), dtype="<f4")
                grad = out[1:].astype(float)
                rel = np.linalg.norm(grad - ref) / np.linalg.norm(ref)
                dloss = abs(out[0] - float(loss)) / float(loss)
                good = rel < 1e-4 and dloss < 1e-4 and grad.size == ref.size
                ok &= good
                print(
                    f"{Path(binary).name:<10} {mode:<5} loss {out[0]:.6f} (torch {float(loss):.6f})  "
                    f"gradient rel. error {rel:.1e}  {'ok' if good else 'MISMATCH'}"
                )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
