#!/usr/bin/env bash
# I14: build, test and time the native kernel on Linux (OpenBLAS) in a container.
#
# Usage, from the repository root (needs Docker):
#   docker run --rm -v "$PWD":/src:ro -v "$PWD/experiments/I14_native_linux/linux_bench.sh":/run.sh:ro \
#       python:3.12-slim bash /run.sh
# Add `--platform linux/arm64` or `linux/amd64` to pick the architecture (an emulated one is slow).
set -e
uname -m; grep PRETTY /etc/os-release
apt-get update -qq >/dev/null && apt-get install -y -qq g++ libopenblas-dev >/dev/null 2>&1
pip install -q --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | grep -v WARNING | tail -1
mkdir /work && (cd /src && tar --exclude=./.git --exclude=./.venv --exclude=./site --exclude=./runs -cf - .) | tar -C /work -xf -
cd /work && pip install -q ".[dev]" 2>&1 | grep -v WARNING | tail -1
python -c "import torch; from pinnrl import native; print('torch', torch.__version__, '| native available:', native.available(), native.unavailable_reason())"
echo "--- the kernel against autograd"
PINNRL_REQUIRE_NATIVE=1 python -m pytest tests/unit_tests/test_native.py -q -p no:cacheprovider --no-cov 2>&1 | tail -1
echo "--- step times, 1 thread"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/I13_native_higher_order/run.py \
    --problems heat kdv_soliton --seeds 1 --rounds 3 --out /tmp/res 2>&1 | grep -v "^saved"
