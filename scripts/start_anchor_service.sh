#!/usr/bin/env bash
# Run the pilot first and retain it with the full background run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
py="${PYTHON_BIN:-$(command -v python)}"
py="$("$py" -c 'import sys; print(sys.executable)')"
mkdir -p runs
root="$(mktemp -d "$PWD/runs/anchor-XXXXXXXX")"
unit="anchor-context-${root##*-}"
args=(--config "${CONFIG:-configs/deepglobe_rgb.json}" --manifest "${MANIFEST:-runs/deepglobe/manifest.json}" --base-run "${BASE_RUN:-runs/deepglobe}" "$@")
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
systemctl --user show-environment >/dev/null
df -h "$root"
"$py" -u -m selective_seg.anchor_experiment pilot --output "$root/pilot" "${args[@]}"
cmd=("$py" -u -m selective_seg.anchor_experiment study --output "$root/study" "${args[@]}")
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
systemd-run --user --unit="$unit" --working-directory="$PWD" --property=Type=exec \
  --property="StandardOutput=append:$root/service.log" --property="StandardError=append:$root/service.log" \
  --setenv="OMP_NUM_THREADS=$OMP_NUM_THREADS" --setenv="MKL_NUM_THREADS=$MKL_NUM_THREADS" "${cmd[@]}"
printf 'Output: %s\nStatus: systemctl --user status %s\nLog: tail -f %q\n' "$root" "$unit" "$root/service.log"
