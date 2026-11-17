# I15: one value pass, and the registry's classes through the kernel

**Question.** With the kernel on, the boundary and initial points of a step still went through it in separate calls, and the registry's PDE classes (`HeatEquation`, `KdVEquation`, ...) did not use it at all. What do one call for every value point and the kernel behind `PDEBase.compute_derivatives` buy?

**Short answer.** The single value pass buys 2 % (heat, 5.80 → 5.69 ms a step; 1 % with layer normalization): the kernel's per-call cost was already small once the two faces shared a call (I11). The registry's classes gain what the catalogue's did: 1.7–2.7× on a plain network and 2.3–4.7× with layer normalization, the residuals agreeing with autograd to $10^{-6}$. The registry's heat problem is periodic, so its faces still need autograd for the normal derivative and it gains less (1.7×) than the catalogue's Dirichlet heat (2.0×).

## Setup

The trainer's step of I11 (a `tanh` network 2-64-64-64-64-1, plain and with layer normalization; 4,096 collocation + 512 boundary + 512 initial points drawn afresh every step; Adam; float32; one thread; Apple M4, idle this time). Two arms alternate in five rounds of 200 steps and keep their best round; three seeds.

1. **Value pass.** The catalogue's heat problem with `derivatives="native"`, with `PDEBase._batched_values` (one call for the two faces and the initial points) against the step before I15 (the faces in one call, the initial points in another).
2. **Registry.** `heat` (periodic), `kdv`, `burgers` and `pendulum` from `config.yaml` through `create_pde` and `build_model` with `model.derivatives` set to `autograd` or `native`; `PDEBase.compute_derivatives` asks the kernel for the pure derivatives of every order the class requests.

```bash
python experiments/I15_native_value_pass/run.py     # ~12 min
```

## Results

| Catalogue heat, kernel on | separate passes | one pass |
|---|---|---|
| plain | 5.80 ms | 5.69 ms (−2 %) |
| layer normalization | 11.26 ms | 11.19 ms (−1 %) |

| Registry class | plain: autograd → kernel | layer norm: autograd → kernel | residual agreement |
|---|---|---|---|
| `heat` (periodic faces) | 11.6 → 6.85 (**1.7×**) | 35.8 → 14.6 (**2.5×**) | 8e-7 |
| `kdv` ($u_{xxx}$) | 20.8 → 7.84 (**2.7×**) | 70.4 → 15.1 (**4.7×**) | 4e-7 |
| `burgers` | 11.4 → 5.66 (**2.0×**) | 34.2 → 11.1 (**3.1×**) | 1e-6 |
| `pendulum` (ODE, $\theta'$, $\theta''$) | 10.9 → 6.30 (**1.7×**) | 32.0 → 14.0 (**2.3×**) | 2e-7 |

Medians of three seeds; the spread between seeds is under 1 %.

## Conclusion

1. **The value pass was not where the time was.** I11 counted 0.5 ms for three value calls against one; after the faces were merged (I11's own follow-up) the remaining merge is worth 0.1 ms. The step with the kernel is now 5.7 ms on heat against the 3.6 ms of the framework-free program, and the rest is the trainer's bookkeeping (0.8 ms), the kernel's generality and the loss arithmetic in PyTorch.
2. **The registry's classes are first-class users of the kernel.** `pde.derivatives = "native"` (set by `pinnrl-train --derivatives native` for every problem now, not only `--pde-file`) routes `compute_derivatives` through `model.jets`; the dictionary the classes read (`dt`, `dx2`, `dx3`, `laplacian`, ...) is unchanged, so no class was edited.
3. **Where a condition differentiates the network, the value pass stays with autograd**: periodic and Neumann faces, and the initial velocity of second-order problems. Those terms could take their derivative from the kernel too (a first derivative along the face's axis); it is the remaining per-step cost on the registry's heat problem and on the wave equation.

## What changed in the package

- `PDEBase.derivatives` (class attribute, `"autograd"` or `"native"`), `PDEBase.native_active(model)`, and `compute_derivatives` through the kernel via `_native_derivatives`; `create_pde` sets `pde.derivatives` for registry problems as it did for custom ones.
- `PDEBase._batched_values(model)` and the `_value_stash` that `compute_loss` fills and `_boundary_loss` / `_initial_loss` read (in `PDEBase` and `CustomPDE`), when `model.native_values` is on and no boundary or initial term needs a derivative at its points.
- Tests: `test_registry_classes_run_through_the_kernel` (heat, wave, burgers, kdv, pendulum: residual, loss and weight gradient against autograd) and `test_boundary_and_initial_points_go_through_one_value_pass`.
