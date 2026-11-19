# Environment variables

Everything the package, its tests and its experiments read from the environment. None is required.

## The native engine

| Variable | Default | Effect |
|---|---|---|
| `PINNRL_NATIVE_SPECIALIZE` | `32` | Evaluations of a problem served by the general kernel before the step compiled for its equations takes over. `1` switches at the first evaluation and waits for the compiler; `off`, `no`, `false` or `0` keeps the general kernel throughout |
| `PINNRL_NATIVE_BLOCK` | `1024` | Points per block of the compiled step |
| `PINNRL_NATIVE_CACHE` | `~/.cache/pinnrl/native` | Directory of the compiled libraries |
| `CXX` | the first of `clang++`, `g++`, `c++` on the path | The C++ compiler |
| `PINNRL_NATIVE_CXXFLAGS` | empty | Extra compiler flags, split on spaces |
| `PINNRL_NATIVE_LDFLAGS` | empty | Extra linker flags, split on spaces |

See [The native engine](../engine.md) for what each one controls.

## Datasets

| Variable | Default | Effect |
|---|---|---|
| `PINNRL_WELL_CACHE` | `~/.cache/pinnrl/well` | Where flattened slices of The Well are stored as `.npz` |

## Tests

| Variable | Default | Effect |
|---|---|---|
| `PINNRL_REQUIRE_NATIVE` | unset | Fail, instead of skipping, the kernel's tests when it cannot be built. CI sets it to `1` |
| `PINNRL_NETWORK_TESTS` | unset | Run the dataset tests that download from Hugging Face (about ten minutes). Without it they are skipped |

## Experiments

| Variable | Default | Effect |
|---|---|---|
| `TQDM_DISABLE` | unset | `1` silences progress bars; the timing scripts set it |
| `I17_LAYER_NORM` | unset | Set by `experiments/I17_step_head_to_head/run.py --layer-norm` for its child processes; not meant to be set by hand |
| `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OPENBLAS_NUM_THREADS` | the library's | Threads of the matrix products; the timing scripts set them to 1 |
