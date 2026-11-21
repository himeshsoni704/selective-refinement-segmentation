#!/usr/bin/env bash
# Isolated run; waits for the existing diagnostic suite before using its GPU.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/himesh/Downloads/.venv/bin/python}"
OUTPUT_DIR="${1:-$REPO_DIR/runs/transitions-20261121-gpu}"
WAIT_FOR_STATUS="${WAIT_FOR_STATUS:-$REPO_DIR/runs/diagnostics-20261121-gpu/status.json}"
UNIT_NAME="${UNIT_NAME:-selective-refinement-transitions-20261121}"
DEVICE="${DEVICE:-cuda}"
WAIT_ARGS=()
SYSTEMD_ARGS=()
if [[ "$WAIT_FOR_STATUS" != "none" ]]; then
  test -f "$WAIT_FOR_STATUS"
  WAIT_ARGS=(--wait-for-status "$WAIT_FOR_STATUS")
fi
if [[ "${SYSTEMD_WAIT:-0}" == "1" ]]; then
  SYSTEMD_ARGS=(--wait)
fi
test -x "$PYTHON_BIN"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
exec systemd-run --user "${SYSTEMD_ARGS[@]}" \
  --unit="$UNIT_NAME" \
  --description="Consistent class-transition selector experiments" \
  --working-directory="$REPO_DIR" \
  --property=Type=exec \
  --property=Restart=no \
  --property=StandardOutput="append:$OUTPUT_DIR/service.log" \
  --property=StandardError="append:$OUTPUT_DIR/service.log" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  "$PYTHON_BIN" -u -m selective_seg.transition_experiment \
  --output "$OUTPUT_DIR" --device "$DEVICE" --threads 4 \
  --seeds 42 43 44 --epochs 200 --passes 3 "${WAIT_ARGS[@]}"
