# I17: the training step against the alternatives

**Question.** Is `pinnrl` the fastest way to take a PINN step on a CPU, against everything else a user could install? If not, what does the leader do that it does not?

**Short answer.**

- **On an Apple M4 it is first on all nine equations**: 1.1–1.8× JAX's best mode, 2.4–7.7× eager PyTorch, 2.7–9.9× DeepXDE, about 3× NVIDIA PhysicsNeMo where that runs. With layer normalization it is 1.2–1.5× JAX and 3.4–12× eager PyTorch.
- **On a Linux x86 runner it is not first on the plain network.** JAX's forward mode is 1–11 % ahead on seven of the nine problems; `pinnrl` leads by 1.2× at fourth order, and by 1.3–1.7× on every problem once layer normalization is on. Against eager PyTorch and DeepXDE the margins are as on the M4.
- **It was not first anywhere when the experiment started.** The first run had JAX's forward mode 1.3× ahead on heat, and named two mechanisms: XLA fuses each layer's derivative arithmetic into one loop written for the equation, and its `tanh` is a rational approximation three times as fast as a vector library's. Both are now in the package (`pinnrl.native.specialize`, the kernels' `tanh`).

## Setup

One step, the same in every arm: a `tanh` network d-64-64-64-64-k in float32, the loss `mean(r^2) + 10 mean(ic^2) + 10 mean(bc^2)` on 4,096 collocation, 512 initial and 512 Dirichlet boundary points drawn afresh every step, Adam at 1e-3.

| Problem | Residual |
|---|---|
| heat | $u_t - 0.1 u_{xx}$ |
| Burgers | $u_t + u u_x - 0.01 u_{xx}$ |
| Allen–Cahn | $u_t - 10^{-4} u_{xx} + 5u^3 - 5u$ |
| KdV | $u_t + 6 u u_x + u_{xxx}$ |
| BBM | $u_t + u_x + u u_x - u_{xxt}$ (a mixed third derivative) |
| Cahn–Hilliard | $u_t - (6 u u_x^2 + (3u^2 - 1) u_{xx} - 0.01 u_{xxxx})$ |
| Kuramoto–Sivashinsky (K–S) | $u_t + u u_x + u_{xx} + u_{xxxx}$ |
| Schrödinger | $u_t + \tfrac12 v_{xx} + (u^2 + v^2) v$, $v_t - \tfrac12 u_{xx} - (u^2 + v^2) u$ (two fields) |
| heat 2-D | $u_t - 0.1 (u_{xx} + u_{yy})$ |

| Arm | How the derivatives are taken |
|---|---|
| `pinnrl` | the package's trainer with its defaults: C++ generated for the equation (fused Taylor-mode loops, the residual and its adjoint inlined), compiled once and cached |
| `pinnrl-general` | the same trainer with `PINNRL_NATIVE_SPECIALIZE=off`: the table-driven kernel for any equation, the residual and loss in PyTorch |
| `pinnrl-autograd` | the same trainer with `derivatives="autograd"` |
| `torch-eager` | a minimal PyTorch loop, nested `torch.autograd.grad` with `create_graph=True`, the textbook step |
| `torch-compile` | that loop under `torch.compile(fullgraph=True)` |
| `torch-func` | `torch.func.jvp` nested for the input derivatives, one reverse pass for the weights |
| `jax-reverse` | JAX under `jit`: `vmap` of nested `grad` |
| `jax-forward` | JAX under `jit`: nested `jvp`, one reverse pass |
| `jax-jet` | JAX under `jit`: `jax.experimental.jet` (Taylor mode) along each coordinate, `jvp` for a mixed term, one reverse pass |
| `deepxde` | DeepXDE with its PyTorch backend, nested `dde.grad.jacobian`, fixed points |
| `physicsnemo` | NVIDIA PhysicsNeMo's `PhysicsInformer` with `grad_method="autodiff"` on the same PyTorch network |

Each arm runs in its own process, checks its residual against the reverse-mode one of its framework before it is timed, warms up through any compilation, and keeps the best of three rounds of 100 steps over two passes. One thread.

```bash
uv run --with jax --with deepxde python experiments/I17_step_head_to_head/run.py               # ~45 min
uv run --with jax --with deepxde python experiments/I17_step_head_to_head/run.py --layer-norm  # ~40 min
uv run --with nvidia-physicsnemo python experiments/I17_step_head_to_head/run.py --arms physicsnemo torch-eager --out results/physicsnemo
python experiments/I17_step_head_to_head/plot.py
```

The Linux numbers come from the manual workflow `.github/workflows/benchmark.yml` (Actions > Benchmark > Run workflow).

## Results

![I17](../../docs/assets/benchmarks/i17_step_head_to_head.png)

### Apple M4, plain network

Milliseconds per step, PyTorch 2.10, Accelerate; the best of each column in bold.

| Implementation | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | **3.7** | **3.7** | **3.7** | **4.4** | **5.5** | **5.7** | **5.7** | **3.8** | **5.1** |
| JAX, `jit`, forward mode (nested `jvp`) | 4.3 | 4.2 | 4.2 | 6.1 | 6.6 | 10.4 | 10.4 | 4.4 | 6.6 |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 4.8 | 4.8 | 4.8 | 6.8 | 8.9 | 9.6 | 9.6 | 5.1 | 7.3 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 5.2 | 5.1 | 5.1 | 7.2 | 8.4 | 10.0 | 10.1 | 5.5 | 7.7 |
| JAX, `jit`, reverse over reverse | 6.0 | 7.7 | 6.5 | 15.7 | 15.6 | 56.8 | 56.5 | 11.4 | 9.6 |
| PyTorch, nested `autograd.grad` | 8.7 | 8.8 | 8.8 | 17.7 | 17.7 | 43.6 | 43.6 | 17.7 | 12.0 |
| `pinnrl`, `derivatives="autograd"` | 10.0 | 10.2 | 10.1 | 19.1 | 19.1 | 45.0 | 45.1 | 15.6 | 14.5 |
| DeepXDE (PyTorch backend) | 10.1 | 10.0 | 10.0 | 21.5 | 21.9 | 56.0 | 55.7 | 18.8 | 14.5 |
| NVIDIA PhysicsNeMo, `PhysicsInformer` (autodiff) ¹ | 10.9 | 10.7 | 10.7 | n/s | n/s | n/s | n/s | 17.7 | 14.3 |
| `torch.func`, nested `jvp` | 16.8 | 17.1 | 16.9 | 35.7 | 30.9 | 92.1 | 92.3 | 17.1 | 27.3 |
| `torch.compile` | fails | fails | fails | fails | fails | fails | fails | fails | fails |

¹ PhysicsNeMo 2.2.2, timed in its own environment (PyTorch 2.15; the eager loop there is within 4 % of the row above). `PhysicsInformer` takes derivatives up to second order, so the four higher-order problems are not supported (n/s). The separate `nvidia-physicsnemo-sym` package does not install without CUDA.

### Apple M4, layer normalization before each `tanh`

| Implementation | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | **7.7** | **7.5** | **7.5** | **9.6** | **12.6** | **12.7** | **12.7** | **7.6** | **11.4** |
| JAX, `jit`, forward mode (nested `jvp`) | 9.7 | 9.2 | 9.2 | 14.0 | 15.4 | 24.5 | 24.2 | 9.2 | 14.3 |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 10.3 | 10.1 | 9.8 | 13.7 | 17.6 | 18.4 | 18.2 | 10.0 | 14.6 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 10.5 | 10.7 | 10.8 | 13.9 | 17.3 | 18.9 | 19.0 | 10.9 | 15.4 |
| PyTorch, nested `autograd.grad` | 26.3 | 26.2 | 26.1 | 59.8 | 59.6 | 152.3 | 152.1 | 60.6 | 38.9 |
| `pinnrl`, `derivatives="autograd"` | 31.9 | 31.6 | 31.1 | 65.2 | 65.3 | 157.8 | 157.5 | 53.8 | 49.2 |

DeepXDE's `FNN` has no layer normalization and PhysicsNeMo was not run in this variant.

### Linux x86 (GitHub Actions runner, AMD EPYC 7763, OpenBLAS), plain network

One pass of three rounds of 60 steps on a shared machine, so read ratios within a column, not absolute times.

| Implementation | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | 24.2 | 24.0 | 23.9 | 30.7 | 38.1 | **38.4** | **38.6** | 24.2 | 37.3 |
| JAX, `jit`, forward mode (nested `jvp`) | **22.3** | **22.9** | **22.7** | **30.4** | **34.1** | 47.4 | 46.5 | **23.5** | **33.9** |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 27.4 | 27.4 | 27.5 | 35.0 | 41.9 | 43.1 | 43.0 | 27.9 | 40.7 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 30.2 | 30.8 | 30.7 | 39.0 | 45.1 | 51.5 | 50.6 | 31.2 | 43.5 |
| PyTorch, nested `autograd.grad` | 55.1 | 61.2 | 58.9 | 135.2 | 128.5 | 349.2 | 343.9 | 99.4 | 79.5 |
| `pinnrl`, `derivatives="autograd"` | 45.9 | 46.8 | 45.7 | 84.4 | 81.4 | 185.3 | 183.8 | 74.5 | 66.2 |
| DeepXDE (PyTorch backend) | 71.3 | 70.3 | 64.6 | 188.5 | 182.9 | 466.4 | 492.6 | 141.8 | 95.2 |

### Linux x86, layer normalization

| Implementation | heat | KdV | Cahn–Hilliard | Schrödinger | heat 2-D |
|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | **30.3** | **39.2** | **52.4** | **30.4** | **49.0** |
| JAX, `jit`, forward mode (nested `jvp`) | 42.1 | 58.4 | 89.7 | 41.7 | 62.9 |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 35.6 | 46.0 | 58.1 | 36.3 | 54.2 |
| PyTorch, nested `autograd.grad` | 149.2 | 457.5 | 1180.4 | 338.1 | 210.8 |

## Findings

1. **On the M4 `pinnrl` is first on every equation**, by 1.13–1.16× over JAX's best mode on the second-order problems in one dimension, 1.2–1.4× at third order and in two dimensions, and 1.8× at fourth order.
2. **The lead does not carry to x86 on the plain network.** There the matrix products are most of the step (no Apple matrix unit), both engines call a tuned product, and the fused loops matter less: the compiled step is only 1.1× the general kernel, against 1.3–1.7× on the M4. JAX's forward mode is 1–11 % ahead on seven problems. `pinnrl` keeps the lead at fourth order (1.2×) and with layer normalization (1.3–1.7×).
3. **Against the PyTorch ecosystem the margin holds on both machines**: 2.1–9× the textbook loop and 2.6–13× DeepXDE; with layer normalization 3.4–23× the textbook loop. PhysicsNeMo's `PhysicsInformer` runs at the speed of the textbook loop plus about 20 %, and stops at second order.
4. **The step compiled per equation is worth 1.3–1.7× over the general kernel on the M4**, more as the order rises. The general kernel walks index tables, one loop per term of each recurrence; the generated code is one loop per layer each way with a straight-line body for exactly the streams the equation uses.
5. **A third of the first gap was `tanh`.** Accelerate's `vvtanhf` costs 1.1 ns an element and the step evaluates 1.3 million of them; the rational form Eigen uses (which is what XLA runs) costs 0.4 ns at the same float32 accuracy (2.4e-7).
6. **Blocking the batch for the cache does not pay on the M4**: blocks of 1,024 points are as fast as the whole batch and smaller ones are slower.
7. **The technique decides before the framework does.** Reverse over reverse is 1.4–6× behind forward or Taylor mode inside JAX itself. `torch.func`'s forward mode loses to eager reverse mode, and `torch.compile` cannot run the step (with `fullgraph=True` it stops at `torch.autograd.grad`).

## What this does not show

- **Error at equal time.** Every arm takes the same Adam step; this is the cost of a step, not the accuracy reached. That comparison is the roadmap's research bets.
- **More cores.** One thread throughout. On x86 the blocks of the compiled step are independent and could run on a core each; that is not built.
- **The shared runner is noisy** and was run once; the M4 was running other light work during the final pass (the earlier pass on an idle machine had `pinnrl` at 3.5–5.6 ms, 1.2–1.9× JAX).
- **One network size**, and a CPU result: on a GPU with large batches the frameworks' own kernels win and `pinnrl` falls back to autograd.
- **Related work.** Per-residual derivative kernels inside PyTorch and JAX on a GPU have been reported (Cao, Lu and Zhang, arXiv 2606.29702, a preprint we have not reproduced); what is measured here is a generated forward and reverse pass on a CPU.
