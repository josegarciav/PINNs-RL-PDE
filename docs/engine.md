# The native engine

How `pinnrl` computes a training step. You do not need this page to use the engine: it is the default.

- **To use it:** [Custom equations](custom_equations.md#a-faster-step-the-compiled-kernel).
- **Its measurements:** [Benchmarks, experiment I17](benchmarks.md#the-step-against-the-alternatives-experiment-i17).
- **Its functions:** [API reference](api/native.md).

## The problem

A PINN minimises the squared residual of an equation. For heat,

$$
L(w) = \frac{1}{N} \sum_{n=1}^{N} r(x_n, t_n)^2, \qquad r = u_t - \alpha\, u_{xx},
$$

where $u$ is the network with weights $w$. A step needs $u_t$, $u_{xx}$ and $\partial L / \partial w$.

- **Residual** $r$: what is left when the network is put into the equation. Zero for an exact solution.
- **Collocation points** $(x_n, t_n)$: the points where the residual is evaluated.
- **Reverse-mode differentiation (autograd):** the chain rule applied backwards from the output. One pass per derivative order, each over the graph of the one before.

The textbook step applies reverse mode twice for $u_{xx}$ and a third time for $w$. Its cost grows with the order: 8.7 ms for heat, 43.6 ms for Cahn–Hilliard in eager PyTorch (I17).

## The idea

Carry the derivatives forward with the value, in one pass, then take one reverse pass written by hand.

| Layer | File | What it does |
|---|---|---|
| General kernel | `jets.cpp`, `native.mlp_jets` | Derivatives of any order, for any equation; residual and loss stay in PyTorch |
| Step compiled per equation | `specialize.py`, `step.h` | C++ generated for one set of equations; one call returns the loss term and every gradient |
| Rational `tanh` | `fast_tanh.h` | The activation both use |

## Taylor-mode streams

The network is a stack of layers

$$
a = W h + b, \qquad h = \tanh(a).
$$

The kernel carries a set of **streams** through it.

- **Stream:** one array per derivative, for all points at once.
- **Multi-index** $\alpha$: how many times each input is differentiated. $u_{xxt}$ is $\alpha = (2, 1)$ over $(x, t)$.
- **Taylor coefficient:** the derivative divided by a factorial, $u_\alpha = D^\alpha u / \alpha!$. Stream 0 is the value.
- **Downward closed:** if $\alpha$ is carried, so is every $\beta \le \alpha$. $u_{xxt}$ brings $u, u_x, u_t, u_{xx}, u_{xt}$.

`native.taylor_streams` builds the set and sorts it by degree. Heat needs 4 streams, KdV 5, Cahn–Hilliard 6.

**Through a linear layer** each stream passes linearly, and the bias enters the value only:

$$
a_\alpha = W h_\alpha \quad (\alpha \ne 0), \qquad a_0 = W h_0 + b.
$$

All streams of all $N$ points are stacked in one $(S N \times \text{width})$ matrix, so a layer is one matrix product.

**Through `tanh`**, with $s = \tanh(a)$ and $q = 1 - s^2$ (so $s' = q\, a'$), for $\alpha_i > 0$:

$$
\alpha_i\, s_\alpha = \sum_{\beta \le \alpha,\ \beta_i \ge 1} \beta_i\, a_\beta\, q_{\alpha - \beta},
\qquad
q_\gamma = \delta_{\gamma 0} - \sum_{\beta \le \gamma} s_\beta\, s_{\gamma - \beta}.
$$

The reverse pass is the same recurrence read backwards.

**Through layer normalization**, per row of a layer:

$$
c = a - \operatorname{mean}(a), \qquad v = \operatorname{mean}(c^2) + \varepsilon, \qquad r = v^{-1/2}, \qquad y = \gamma\, c\, r + \beta.
$$

The streams follow by the same arithmetic for the mean, the product and the inverse square root.

## The general kernel

`jets.cpp` implements the recurrences for any stream set.

1. `Tables` builds the index tables of the recurrences from the multi-indices.
2. `pinn_jets_forward` carries the streams through the layers and keeps them in a work buffer.
3. PyTorch computes the residual and the loss from the output streams.
4. `pinn_jets_backward` returns the weight gradient when PyTorch asks for it.

`mlp_jets` wraps steps 2 and 4 as a `torch.autograd.Function`, so the outputs are ordinary tensors.

It applies to:

- a feed-forward stack of `Linear` and `tanh`, with or without `LayerNorm` before each `tanh` (`native.supported_stack` is the test);
- float32 on the CPU;
- macOS (Accelerate) or Linux (OpenBLAS).

Its generality has a cost: each term of each recurrence is one loop over arrays of megabytes, and the two passes are separate calls with PyTorch between them.

## The step compiled per equation

For equations given as text, `pinnrl.native.specialize` writes C++ for those equations and compiles it. One call returns $L$ and every gradient.

The generated file includes `jets.cpp` and `step.h` and defines five functions:

| Function | Generator | Body |
|---|---|---|
| `gen_tanh_fwd`, `gen_tanh_bwd` | `tanh_program` | One loop over a layer each way, straight-line code for exactly the streams used |
| `gen_norm_fwd`, `gen_norm_bwd` | `norm_program` | Per row, three loops over the features each way |
| `gen_residual` | `residual_code` | Per point: $r$ for every equation, $r^2$, and the adjoint of every stream |

- **Adjoint:** the derivative of the loss with respect to an intermediate quantity. The reverse pass starts from the adjoints of the output streams, $\partial L / \partial u_\alpha = 2\, r\, \partial r / \partial u_\alpha \cdot \alpha! / N$.
- **Fused loop:** one loop that does the work of several, so each array is read once.

How `residual_code` gets the adjoints:

1. It takes the SymPy expressions `CustomPDE` parsed.
2. It differentiates each residual with respect to every symbol (`sympy.diff`).
3. It shares subexpressions (`sympy.cse`) and prints single-precision C.
4. If a function has no C form, it returns `None` and the general kernel carries on.

`step.h`'s `blocked_residual` runs the batch in blocks of points: forward, residual, reverse, with the gradients summed over the blocks.

What the compiled code reads at run time, so one build serves a problem while its numbers change:

- **Parameters** of the equations. Trainable ones get $\partial L / \partial \theta$ from the same loop.
- **Per-point values:** measured inputs and unknown functions of the coordinates, evaluated by PyTorch. Their adjoints are returned, and PyTorch backpropagates into the unknown function's own weights.
- **Scaling** of a network built with `normalize=True`: mapped inputs for the kernel, physical coordinates for the residual, and one factor per stream.

`CustomPDE._step_spec` builds a problem's description (`ResidualSpec`); `CustomPDE._compiled_residual_loss` calls the step.

### When it takes over

Compiling takes about 1.5 s, once per set of equations and machine.

1. Evaluations 1 to 31 use the general kernel.
2. At evaluation 16 the build starts in a background thread.
3. At evaluation 32 the compiled step takes over, waiting for the compiler if it must.

The switch is at exactly evaluation 32 whether the library was cached or not, so a run does not depend on build time. `PINNRL_NATIVE_SPECIALIZE` changes the number.

### The cache

- **Where:** `~/.cache/pinnrl/native` (`PINNRL_NATIVE_CACHE` moves it).
- **Files:** `jets-<hash>.so`, `evaporator-<hash>.so`, and one `step-<hash>.so` per set of equations.
- **Key:** a hash of the generated source, `jets.cpp`, `step.h`, `fast_tanh.h` and a version tag. An upgrade rebuilds what changed.
- **Clearing it:** deleting the directory is always safe.

### Blocks

`PINNRL_NATIVE_BLOCK` sets the points per block (default 1,024). On an Apple M4, blocks of 1,024 are as fast as the whole batch and smaller ones are slower (I17). The blocks are independent, which a threaded version would use ([roadmap](roadmap.md#v09-the-engine-everywhere)).

## The rational `tanh`

$$
\tanh(x) \approx \frac{x\, P(x^2)}{Q(x^2)}, \qquad |x| \le 7.905,
$$

with $P$ of degree 6 and $Q$ of degree 3 in $x^2$, the form Eigen uses (`fast_math::tanhf_rational`).

| | Time per element | Error |
|---|---|---|
| Rational form | 0.4 ns | within $2.4 \times 10^{-7}$ |
| Accelerate's `vvtanhf` | 1.1 ns | |

It has no exponential and no branch, so the compiler vectorises it. A step evaluates about 1.3 million of them (I17, Apple M4).

## What the rest of the package can rely on

- **The same numbers.** Residual, loss and gradients agree with autograd to float32 rounding ($10^{-7}$–$10^{-5}$ relative). `test_native.py` and `test_native_specialize.py` check each case.
- **A gradient can be differentiated again.** With `create_graph=True`, the gradient is taken by autograd on the same function. Hessian-vector products and `pinnrl.analysis` work, at autograd speed.
- **`torch.func` sees the ordinary module.** Inside `jvp`, `vmap` or `grad`, PyTorch evaluates the network.
- **A silent fallback.** Where one layer of the engine does not apply, the next one down runs.

- **Hessian-vector product:** the second derivative of the loss applied to a vector, $H v$, without forming $H$.

## Where it does not apply

| Situation | What runs |
|---|---|
| Another architecture or activation (Fourier features, SIREN, ResNet), dropout | autograd |
| A GPU (`mps`, `cuda`) or float64 | autograd |
| No C++ compiler, or no OpenBLAS on Linux | autograd |
| A residual given as a Python function | autograd |
| An unknown function of a field, or a derivative of an unknown function | autograd |
| A loss other than the squared one; a steady problem | general kernel |
| A function in the equation with no C form | general kernel |
| A parameter made differentiable outside `trainable_parameters` | general kernel |
| `derivatives="autograd"` | autograd, by request |

## Which engine ran

```python
from pinnrl import native
from pinnrl.pdes.catalog import from_catalog

pde = from_catalog("heat")
trainer = pde.build_trainer("feedforward", num_collocation_points=256, batch_size=256)

print(native.available(), native.unavailable_reason())   # True None where the kernel builds
print(pde.native_active(trainer.model))                  # True: the kernel serves this problem

x, t = pde.generate_collocation_points(256)
loss = pde.compute_loss(trainer.model, x, t)["residual"]
print(type(loss.grad_fn).__name__)   # _ResidualStepBackward once the compiled step has taken over
```

The last line prints an ordinary PyTorch node for the first 31 evaluations and `_ResidualStepBackward` from the 32nd.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PINNRL_NATIVE_SPECIALIZE` | `32` | Evaluations before the compiled step takes over; `off` or `0` disables it |
| `PINNRL_NATIVE_BLOCK` | `1024` | Points per block of the compiled step |
| `PINNRL_NATIVE_CACHE` | `~/.cache/pinnrl/native` | Where compiled libraries are kept |
| `CXX` | first of `clang++`, `g++`, `c++` | The compiler |
| `PINNRL_NATIVE_CXXFLAGS` | empty | Extra compiler flags |
| `PINNRL_NATIVE_LDFLAGS` | empty | Extra linker flags |
| `PINNRL_REQUIRE_NATIVE` | unset | In the tests: fail instead of skipping when the kernel cannot be built |

## When a build fails

Nothing breaks: the run goes on by autograd and a warning is logged once.

1. Print the reason: `python -c "from pinnrl import native; print(native.unavailable_reason())"`.
2. `no C++ compiler found`: run `xcode-select --install` on macOS, or install `g++` on Linux.
3. `compiling jets.cpp failed` on Linux: `apt install libopenblas-dev`, or point `PINNRL_NATIVE_CXXFLAGS` and `PINNRL_NATIVE_LDFLAGS` at OpenBLAS.
4. `the specialised step could not be built` (a warning): the general kernel stays in charge. `PINNRL_NATIVE_SPECIALIZE=1` reproduces it at the first evaluation.

The kernels build on macOS and Linux only.

## Extending it

1. **A new activation:** add its Taylor recurrence, forward and reverse, to `jets.cpp` and to `tanh_program`, and an entry in `supported_stack`.
2. **A new term in the compiled step** (boundary points, the optimizer): add it to the driver in `step.h`, with the per-equation part generated in `specialize.py`.
3. **Check it:** every case of `test_native_specialize.py` compares loss and gradient with autograd; `experiments/I17_step_head_to_head/run.py` times the step.
