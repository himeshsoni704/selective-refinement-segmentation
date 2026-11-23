#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
full=0
if [[ ${1:-} == --full-suite ]]; then
    full=1
    shift
fi
if [[ -z ${PYTHON_BIN:-} && -x "$HOME/Downloads/.venv/bin/python" && -z ${VIRTUAL_ENV:-} ]]; then
    PYTHON_BIN="$HOME/Downloads/.venv/bin/python"
fi
py="${PYTHON_BIN:-$(command -v python)}"
py="$("$py" -c 'import sys; print(sys.executable)')"
systemctl --user show-environment >/dev/null
"$py" - <<'PY'
import torch, torchvision
from torchvision.ops import nms
if not torch.cuda.is_available():
    raise SystemExit("Activate the existing CUDA-enabled training environment first.")
b = torch.tensor([[0.,0.,1.,1.]],device="cuda")
nms(b,torch.ones(1,device="cuda"),.5)
print(torch.cuda.get_device_name(), torch.__version__, torchvision.__version__)
PY
mkdir -p runs
root="$(mktemp -d "$PWD/runs/yolo-pair-XXXXXXXX")"
# Preserve the training environment's existing CUDA PyTorch/torchvision builds.
"$py" -c 'from importlib.metadata import version; print("torch=="+version("torch")); print("torchvision=="+version("torchvision"))' > "$root/torch-constraints.txt"
if ! "$py" -c 'from importlib.metadata import version; import ultralytics; assert version("ultralytics")=="8.3.228"' 2>/dev/null; then
    "$py" -m pip install -c "$root/torch-constraints.txt" 'ultralytics==8.3.228'
fi
weights="${YOLO_WEIGHTS:-$PWD/runs/model-assets/yolov8n.pt}"
if [[ -z "${YOLO_WEIGHTS:-}" ]]; then
    mkdir -p "$(dirname "$weights")"
    expected=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
    if [[ ! -f "$weights" ]]; then
        tmp="$(mktemp "$root/yolov8n-XXXXXXXX")"
        curl --fail --location --retry 2 --output "$tmp" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
        printf '%s  %s\n' "$expected" "$tmp" | sha256sum -c -
        mv -n "$tmp" "$weights"
    fi
    printf '%s  %s\n' "$expected" "$weights" | sha256sum -c -
fi
weights="$(realpath -e "$weights")"
unit="yolo-anchor-${root##*-}"
args=(--config "${CONFIG:-configs/deepglobe_rgb.json}" --manifest "${MANIFEST:-runs/deepglobe/manifest.json}" --base-run "${BASE_RUN:-runs/deepglobe}" "$@")
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
if (( full )); then
    base="${BASE_RUN:-runs/deepglobe}"
    specialist="${SPECIALIST:-runs/specialist_combined/specialist.pt}"
    for file in "${CONFIG:-configs/deepglobe_rgb.json}" "${MANIFEST:-runs/deepglobe/manifest.json}" "$base"/{coarse,low,high,selector}.pt "$base/calibration.json" "$specialist"; do
        [[ -f "$file" ]] || { echo "Missing file: $PWD/$file" >&2; exit 1; }
    done
    args+=(--specialist "$specialist" --candidate-fraction .30 --steps 2000 --epochs 60 --passes 3 --budgets 100 250)
fi
df -h "$root"
"$py" -u -m selective_seg.anchor_experiment pilot --output "$root/pilot-baseline" "${args[@]}"
"$py" -u -m selective_seg.anchor_experiment pilot --output "$root/pilot-yolo" --yolo-weights "$weights" "${args[@]}"
cmd=(bash scripts/run_yolo_pair.sh "$py" "$root" "$weights" "${args[@]}")
if (( full )); then
    cmd=(bash scripts/run_full_anchor_suite.sh "$py" "$root" "$weights" "$base" "${CONFIG:-configs/deepglobe_rgb.json}" "${MANIFEST:-runs/deepglobe/manifest.json}" "${args[@]}")
fi
printf 'Worker: '; printf '%q ' "${cmd[@]}"; printf '\n'
systemd-run --user --unit="$unit" --working-directory="$PWD" --property=Type=exec \
    --property="StandardOutput=append:$root/service.log" --property="StandardError=append:$root/service.log" \
    --setenv="OMP_NUM_THREADS=$OMP_NUM_THREADS" --setenv="MKL_NUM_THREADS=$MKL_NUM_THREADS" "${cmd[@]}"
printf 'Output: %s\nStatus: systemctl --user status %s\nLog: tail -f %q\n' "$root" "$unit" "$root/service.log"
