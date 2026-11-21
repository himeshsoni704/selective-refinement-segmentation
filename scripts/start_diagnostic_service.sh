#!/usr/bin/env bash
# Run from any directory; uses the existing environment without modifying it.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/himesh/Downloads/.venv/bin/python}"
OUTPUT_DIR="${1:-$REPO_DIR/runs/diagnostics-20261121-gpu}"
DEVICE="${DEVICE:-cuda}"
UNIT_NAME="${UNIT_NAME:-selective-refinement-diagnostics-20261121}"
mkdir -p "$OUTPUT_DIR"
exec systemd-run --user \
  --unit="$UNIT_NAME" \
  --description="Frozen-checkpoint refinement diagnostic experiments" \
  --working-directory="$REPO_DIR" \
  --property=Type=exec \
  --property=Restart=no \
  --property=StandardOutput="append:$OUTPUT_DIR/service.log" \
  --property=StandardError="append:$OUTPUT_DIR/service.log" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  "$PYTHON_BIN" -u -m selective_seg.diagnostic_suite \
  --output "$OUTPUT_DIR" --device "$DEVICE" --threads 4 --passes 3
