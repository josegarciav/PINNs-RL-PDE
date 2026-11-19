# I17: the training step against the alternatives

**Question.** The compiled kernel is 1.9–7× the package's own autograd step (I11, I13). Is it the fastest way to take a PINN step on a CPU, against everything else a user could install?

**Short answer.** It is the fastest step in the PyTorch ecosystem by 1.6× (heat) to 4.2× (Cahn–Hilliard) over eager PyTorch and 1.9–5.6× over DeepXDE, and `torch.compile` cannot run the step at all. It is not the fastest overall: JAX, compiled by XLA and differentiating the inputs in forward mode, is 1.3× ahead on heat and 1.2× on KdV, and level on Cahn–Hilliard. The technique matters more than the framework: every forward-mode or Taylor-mode arm that is compiled lands within 30 % of the others, and every reverse-over-reverse arm is 1.5–6× behind them.

## Setup

One step, the same in every arm: a `tanh` network 2-64-64-64-64-1 in float32, the loss `mean(r^2) + 10 mean(ic^2) + 10 mean(bc^2)` on 4,096 collocation, 512 initial and 512 Dirichlet boundary points drawn afresh every step, Adam at 1e-3. Three residuals of rising order: heat ($u_t - 0.1 u_{xx}$), KdV ($u_t + 6 u u_x + u_{xxx}$) and Cahn–Hilliard ($u_t - (6 u u_x^2 + (3u^2 - 1) u_{xx} - 0.01 u_{xxxx})$).

| Arm | How the derivatives are taken |
|---|---|
| `pinnrl` | the package's trainer with its default `derivatives="auto"`: Taylor mode forward and a hand-written reverse pass in C++ (`pinnrl.native`) |
| `pinnrl-autograd` | the same trainer with `derivatives="autograd"` |
| `torch-eager` | a minimal PyTorch loop, nested `torch.autograd.grad` with `create_graph=True`, the textbook step |
| `torch-compile` | that loop under `torch.compile` |
| `torch-func` | `torch.func.jvp` nested along $x$ for the input derivatives, one reverse pass for the weights |
| `jax-reverse` | JAX under `jit`: `vmap` of nested `grad` |
| `jax-forward` | JAX under `jit`: nested `jvp` along $x$, one reverse pass |
| `jax-jet` | JAX under `jit`: `jax.experimental.jet` (Taylor mode), one reverse pass |
| `deepxde` | DeepXDE 1.x with its PyTorch backend, `dde.grad.jacobian` and `hessian`, fixed points |

Each arm runs in its own process, checks its residual against the reverse-mode one of its framework before it is timed, warms up through any compilation, and keeps the best of five rounds of 100 steps over two passes. Apple M4 on mains power, PyTorch 2.10, one thread. (The first pass of heat and KdV ran in low power mode at half speed; the table is the second pass.)

```bash
uv run --with jax --with deepxde python experiments/I17_step_head_to_head/run.py   # ~35 min
python experiments/I17_step_head_to_head/plot.py
```

## Results

![I17](../../docs/assets/benchmarks/i17_step_head_to_head.png)

Milliseconds per step, one thread:

| Implementation | heat, $u_{xx}$ | KdV, $u_{xxx}$ | Cahn–Hilliard, $u_{xxxx}$ |
|---|---|---|---|
| JAX, `jit`, forward mode (nested `jvp`) | **4.2** | **6.1** | 10.6 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 5.1 | 7.7 | **10.2** |
| **`pinnrl`, the default (compiled kernel)** | 5.4 | 7.4 | 10.3 |
| JAX, `jit`, reverse over reverse | 5.8 | 16.0 | 58.9 |
| PyTorch, nested `autograd.grad` | 8.7 | 17.6 | 43.6 |
| DeepXDE (PyTorch backend) | 10.2 | 21.9 | 58.3 |
| `pinnrl`, `derivatives="autograd"` | 10.4 | 19.5 | 45.7 |
| `torch.func`, nested `jvp` | 16.9 | 35.7 | 92.5 |
| `torch.compile` | fails | fails | fails |

With each framework's default threading the order is the same: `pinnrl` 5.5 / 7.5 / 10.5, JAX forward 4.2 / 6.4 / 10.4, eager PyTorch 6.5 / 17.6 / 40.5, DeepXDE 10.2 / 22.0 / 56.6.

## Findings

1. **Within PyTorch nothing comes close.** The kernel is 1.6×, 2.4× and 4.2× the textbook loop and 1.9×, 3.0× and 5.6× DeepXDE, and the gap grows with the order of the equation, because reverse over reverse builds a graph per derivative.
2. **`torch.compile` does not apply.** PyTorch 2.10 refuses the step: "torch.compile with aot_autograd does not currently support double backward". `torch.func`'s forward mode runs but is twice as slow as eager reverse mode: without a compiler behind it, forward mode in tensor operations does not pay (the same result as the Taylor-mode loop of I11).
3. **JAX in forward mode is the fastest step measured**, by 1.3× on heat and 1.2× on KdV, and level at fourth order. XLA fuses the elementwise work of the nested `jvp` the way the kernel's hand-written loops do, and specialises the whole step (the loss, Adam and the sampling are inside the compiled function) where the kernel is a general pass called from Python.
4. **The technique decides, not the framework.** JAX with reverse over reverse is no better than PyTorch at third and fourth order (16 and 59 ms). The three compiled forward-mode arms are within 30 % of each other on every equation.
5. **What the gap is made of.** I11's hand-specialised C++ step runs heat in 3.6 ms, under JAX's 4.2. The kernel's 5.4 ms is its general any-order passes (4.8 ms) plus Python around them. Closing it is the roadmap's next item: the passes generated for the equation's own derivative set, and the loss and optimizer inside the compiled step.

## What this does not show

- **PhysicsNeMo was not run.** `nvidia-physicsnemo-sym` resolves on this machine, but its solver is driven by a Hydra configuration that does not reduce to one timed step. It differentiates with PyTorch autograd, so the eager row is the floor for it; that is an inference, not a measurement.
- **One machine, one network size.** An M4 with Accelerate. On Linux without a matrix-product advantage (I14) the kernel's gain over PyTorch is smaller, and JAX's position there is not measured. The packaged default network (layer normalization, 7 × 128) is in I11–I13 against autograd only.
- **A CPU result.** On a GPU with large batches the frameworks' own kernels win and `pinnrl` falls back to autograd.
- **A step, not a solution.** Every arm takes the same Adam step; error at equal time was not compared, and the arms' initial weights differ.
