# I14: the native kernel on Linux (OpenBLAS)

**Question.** The kernel of I11–I13 was macOS-only (Accelerate). Does the same source build and run on Linux with OpenBLAS, in CI, and what does the step gain there against PyTorch?

**Short answer.** It builds and runs: the same `jets.cpp` compiles with `g++ -lopenblas` on first use, the 13 kernel tests pass on Debian 13 (aarch64, OpenBLAS 0.3.34, PyTorch 2.15 CPU), and CI now installs `libopenblas-dev` and fails rather than skips if the kernel does not build. The gain is smaller than on macOS: 1.4× on heat and 1.9× on KdV with a plain network, 2.0× and 3.0× with layer normalization, against 1.9×, 2.6×, 3.0× and 4.7× on the Mac. The reason is not OpenBLAS, whose `sgemm` runs as fast as `torch.mm` on the kernel's shapes: on a CPU without Apple's AMX matrix units the twelve matrix products of a step cost about 1.5 ms each and are two thirds of the kernel's step, while Accelerate does them in a tenth of that. The kernel's advantage is in the elementwise work (the derivative recurrences and `tanh`), so it pays most where the BLAS is fast.

## Setup

`linux_bench.sh` runs in a `python:3.12-slim` container: it installs `g++` and `libopenblas-dev`, the CPU wheel of PyTorch and the package, checks the kernel builds, runs `tests/unit_tests/test_native.py` with `PINNRL_REQUIRE_NATIVE=1`, and times the I13 benchmark on heat and KdV (one seed, best of three rounds of 200 steps, one thread for OpenBLAS and PyTorch alike). The container ran on the Apple M4 of I11–I13 (Docker's Linux VM, `linux/arm64`), so the hardware is the same as the macOS numbers and only the software stack differs: Debian's OpenBLAS and `g++` against Accelerate and `clang++`.

The Linux build replaces Accelerate's `vvtanhf` with a branch-free Cephes `tanhf` (`pinnrl/native/fast_tanh.h`: a polynomial below 0.625, `1 - 2 / (exp(2|x|) + 1)` above) that the compiler vectorises; it is within one unit in the last place of the C library over [-12, 12] and twice as fast as the scalar `tanhf` on the Mac (0.7 against 1.4 ns an element). glibc's own vector `tanhf` is only declared under full `-ffast-math`, which the build avoids to keep NaN and infinity meaningful.

```bash
docker run --rm --platform linux/arm64 -v "$PWD":/src:ro \
    -v "$PWD/experiments/I14_native_linux/linux_bench.sh":/run.sh:ro python:3.12-slim bash /run.sh
```

## Results

| Trainer step, 1 thread, ms (ratio) | Linux / OpenBLAS: autograd → kernel | macOS / Accelerate (I13, same machine) |
|---|---|---|
| heat, plain | 28.6 → 20.4 (**1.4×**) | 11.1 → 5.7 (1.9×) |
| heat, layer norm | 55.4 → 27.3 (**2.0×**) | 33.8 → 11.1 (3.0×) |
| KdV, plain | 51.6 → 27.9 (**1.9×**) | 20.7 → 7.9 (2.6×) |
| KdV, layer norm | 113.1 → 37.4 (**3.0×**) | 70.6 → 15.2 (4.7×) |

Residual, loss and weight gradient agree with autograd to $10^{-7}$–$10^{-6}$ on Linux as on macOS. Both columns were taken on an idle machine (a first pass under a busy virtual machine gave the Linux column within 5 % of this one and the macOS column 2× larger: the container's times were not the ones inflated). The autograd step itself is 2.5× slower in the container than on macOS, so the Linux stack is slower in every part, not only in the kernel.

**Where the Linux step goes** (heat streams $u_t, u_x, u_{xx}$ through `mlp_jets`, 4,096 points, 4 × 64): forward 8.0 ms, backward 14.4 ms (measured in the first pass). One `sgemm` of (16,384 × 64) · (64 × 64) takes 1.3–1.9 ms in OpenBLAS (core type `neoversen1`, detected correctly in the VM) and 1.4–1.6 ms in `torch.mm`; the step has three such products in the forward pass and six in the backward one, about 14 ms of the 22. On the Mac, I11 measured all the matrix products of a step at 1.5 ms together.

## Conclusion

1. **One source, two platforms.** The kernel is a C++ file with a BLAS call and a `tanh`; the platform differences are the include, the link flag and the vector `tanh`, chosen in `pinnrl/native/__init__.py` (`PINNRL_NATIVE_CXXFLAGS` / `PINNRL_NATIVE_LDFLAGS` point at a non-standard OpenBLAS). CI builds it on every Python version and `PINNRL_REQUIRE_NATIVE=1` turns a silent skip into a failure.
2. **The gain depends on the GEMM.** Taylor-mode does about as many matrix-product flops as double backward (the streams multiply the layer widths); what it saves is the elementwise work and the autograd graph. Where the matrix products are nearly free (AMX) that saving is most of the step and the kernel gains 2–7×; where they cost (NEON, and presumably AVX without MKL's full tuning) they dominate both arms and the gain is 1.4–3×, growing with the order of the equation and with layer normalization as on the Mac.
3. **A real Linux box has not been timed.** The numbers above are from Docker's VM on an M4; an x86 server with MKL in PyTorch and OpenBLAS in the kernel may land elsewhere, and multi-threaded BLAS changes the balance for both arms. The script runs anywhere Docker does.

## What changed in the package

- `pinnrl/native/jets.cpp` includes `<cblas.h>` and `fast_tanh.h` off macOS; `pinnrl/native/__init__.py` builds with `g++ -march=native -lopenblas` on Linux and names the missing piece when the build fails.
- `.github/workflows/checks.yml` installs `libopenblas-dev` and sets `PINNRL_REQUIRE_NATIVE=1` for the tests.
