#!/usr/bin/env bash
# Release gate: build pinnrl from a clean checkout of REF, install the wheel into a
# fresh venv outside the repo, and exercise every user-facing entry point from an
# empty working directory (so nothing from the source tree is importable).
#
# Usage: scripts/verify_release.sh [python-version]   (default 3.12; REF=HEAD)
#   SKIP_NOTEBOOKS=all             skip executing the tutorial notebooks
#   SKIP_NOTEBOOKS="04_x.ipynb ..." skip only the named notebooks (e.g. ones that need network)
#   KEEP_NOTEBOOKS=dir             copy the executed notebooks to dir (outputs from the shipped wheel)
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
  --steps 20 --checkpoints 2 --resample-every 5 --n-collocation 64 --out "$WORK/run/study"

step "documentation snippets run against the installed wheel"
check doc-snippets "$BIN/python" "$WORK/src/scripts/run_doc_snippets.py" --root "$WORK/src" --python "$BIN/python"
cat "$WORK/doc-snippets.log"

step "tutorial notebooks run against the installed wheel"
if [[ "${SKIP_NOTEBOOKS:-}" == "all" ]]; then
  echo "skipped (SKIP_NOTEBOOKS=all)"
else
  WHEEL=$(ls "$WORK"/dist/*.whl)
  uv pip install -q --python "$BIN/python" "pinnrl[notebooks,well] @ file://$WHEEL" ipykernel
  # A kernel registered inside the venv: notebooks import the wheel, never the source tree.
  export JUPYTER_PATH="$WORK/venv/share/jupyter" JUPYTER_DATA_DIR="$WORK/jupyter-data"
  "$BIN/python" -m ipykernel install --prefix "$WORK/venv" --name pinnrl-gate >/dev/null
  mkdir -p "$WORK/nb" && cp "$WORK"/src/notebooks/*.ipynb "$WORK/nb/"
  for nb in "$WORK"/nb/*.ipynb; do
    name=$(basename "$nb")
    if [[ " ${SKIP_NOTEBOOKS:-} " == *" $name "* ]]; then echo "skip  $name"; continue; fi
    check "nb-${name%.ipynb}" bash -c "cd '$WORK/nb' && '$BIN/jupyter' nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=pinnrl-gate --ExecutePreprocessor.timeout=3600 \
      --output-dir '$WORK/nb-out' '$name'"
  done
  if [[ -n "${KEEP_NOTEBOOKS:-}" ]]; then
    mkdir -p "$KEEP_NOTEBOOKS" && cp "$WORK"/nb-out/*.ipynb "$KEEP_NOTEBOOKS"/ 2>/dev/null || true
  fi
fi

step "pinnrl-dashboard serves HTTP 200"
PORT=$("$BIN/python" -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")
# BROWSER=true turns webbrowser.open() into a no-op so no tab opens on the host
BROWSER=true "$BIN/pinnrl-dashboard" --no-browser --port "$PORT" >"$WORK/dashboard.log" 2>&1 &
DASH_PID=$!
check dashboard bash -c "for _ in \$(seq 60); do curl -sf http://127.0.0.1:$PORT/ >/dev/null && exit 0; sleep 1; done; cat '$WORK/dashboard.log'; exit 1"

if ((${#FAILED[@]})); then
  printf '\n%d check(s) failed: %s\n' "${#FAILED[@]}" "${FAILED[*]}"; exit 1
fi
printf '\nAll release checks passed (python %s, %s).\n' "$PY_VERSION" "$REF"
