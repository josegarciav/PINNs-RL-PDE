# Native kernel

An optional compiled kernel for the derivatives of a `tanh` network (`pinnrl.native`); see [The native engine](../engine.md) for how it works, [Custom equations](../custom_equations.md#a-faster-step-the-compiled-kernel) for how to use it, and experiments I11–I17.

::: native.available

::: native.unavailable_reason

::: native.mlp_jets

::: native.supported_layers

::: native.supported_stack

::: native.taylor_streams

::: native.autograd_streams

## The step compiled per equation

`pinnrl.native.specialize` generates, compiles and caches the residual term of one set of equations: the loss term and its weight gradient in one call. [`CustomPDE`](pdes.md) uses it by default after the first evaluations of a problem; see [Custom equations](../custom_equations.md#a-faster-step-the-compiled-kernel).

::: native.specialize.tanh_program

::: native.specialize.norm_program

::: native.specialize.ResidualSpec

::: native.specialize.residual_code

::: native.specialize.step_library

::: native.specialize.run_step

::: native.specialize.specialize_after

::: native.specialize.block_size

## The evaporator's reference solve

The method-of-lines model of the falling-film evaporator integrated in C++ (`pinnrl.native.evaporator`, experiment I16); `FallingFilmEvaporator.reference(method="native")` and `native_propagator` use it.

::: native.evaporator.evaporator_available

::: native.evaporator.evaporator_unavailable_reason

::: native.evaporator.evaporator_solve
