#!/usr/bin/env bash
# Release gate: build pinnrl from a clean checkout of REF, install the wheel into a
# fresh venv outside the repo, and exercise every user-facing entry point from an
# empty working directory (so nothing from the source tree is importable).
#
# Usage: scripts/verify_release.sh [python-version]   (default 3.12; REF=HEAD)
set -euo pipefail

PY_VERSION="${1:-3.12}"
REF="${REF:-HEAD}"
ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; [[ -n "${DASH_PID:-}" ]] && kill "$DASH_PID" 2>/dev/null || true' EXIT
unset VIRTUAL_ENV

step() { printf '\n── %s\n' "$*"; }

step "build sdist + wheel from a clean archive of $REF"
mkdir -p "$WORK/src" "$WORK/run"
git -C "$ROOT" archive "$REF" | tar -x -C "$WORK/src"
(cd "$WORK/src" && uv build -q --out-dir "$WORK/dist")
ls "$WORK/dist"

step "fresh venv (python $PY_VERSION) + wheel install"
uv venv -q --python "$PY_VERSION" "$WORK/venv"
BIN="$WORK/venv/bin"
uv pip install -q --python "$BIN/python" "$WORK"/dist/*.whl
cd "$WORK/run"

FAILED=()
# check NAME CMD...: run CMD, log to $WORK/NAME.log, record a failure instead of aborting
check() {
  local name="$1"; shift
  if "$@" >"$WORK/$name.log" 2>&1; then
    echo "ok    $name"
  else
    echo "FAIL  $name"; tail -15 "$WORK/$name.log" | sed 's/^/      /'; FAILED+=("$name")
  fi
}
small=(--arch feedforward --epochs 2 --device cpu --collocation-points 256 --batch-size 128
       --boundary-points 64 --initial-points 64)

step "import + version"
check import "$BIN/python" -c "import importlib.metadata as m, pinnrl; print(m.version('pinnrl'))"
cat "$WORK/import.log"

step "console scripts --help"
for cmd in pinnrl-train pinnrl-benchmark pinnrl-dashboard; do check "help-$cmd" "$BIN/$cmd" --help; done

step "pinnrl-train: every PDE with the packaged default config"
for pde in heat wave burgers kdv convection allen_cahn cahn_hilliard black_scholes pendulum; do
  check "train-$pde" "$BIN/pinnrl-train" --pde "$pde" "${small[@]}"
done
check train-inverse "$BIN/pinnrl-train" --pde heat "${small[@]}" --mode inverse \
  --identify alpha --initial-guess alpha=0.5 --obs-points 50
check train-rl "$BIN/pinnrl-train" --pde burgers "${small[@]}" --rl

step "outputs land in ./runs of the working directory, never in site-packages"
check runs-in-cwd test -d "$WORK/run/runs"
check no-runs-in-venv bash -c "! find '$WORK/venv' -type d -name runs | grep -q ."

step "pinnrl-benchmark"
check bench-fdm "$BIN/pinnrl-benchmark" fdm --pde heat
check bench-sampling "$BIN/pinnrl-benchmark" sampling --pdes burgers --seeds 1 \
  --time-budget 1 --n-collocation 64 --out "$WORK/run/study"

step "pinnrl-dashboard serves HTTP 200"
PORT=$("$BIN/python" -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
"$BIN/pinnrl-dashboard" --port "$PORT" >"$WORK/dashboard.log" 2>&1 &
DASH_PID=$!
check dashboard bash -c "for _ in \$(seq 60); do curl -sf http://127.0.0.1:$PORT/ >/dev/null && exit 0; sleep 1; done; cat '$WORK/dashboard.log'; exit 1"

if ((${#FAILED[@]})); then
  printf '\n%d check(s) failed: %s\n' "${#FAILED[@]}" "${FAILED[*]}"; exit 1
fi
printf '\nAll release checks passed (python %s, %s).\n' "$PY_VERSION" "$REF"
