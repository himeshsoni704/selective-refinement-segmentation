#!/usr/bin/env bash
set -euo pipefail
py="$1"; root="$2"; weights="$3"; base="$4"; config="$5"; manifest="$6"; shift 6
"$py" -u -m selective_seg.cli benchmark --config "$config" --manifest "$manifest" \
    --run "$base" --calibration "$base/calibration.json" --split val --mode selective \
    --device cuda --passes 3 --warmup 20 --probability-metrics --output "$root/legacy-selective"
for seed in 42 43 44; do
    dir="$root/seed-$seed"
    mkdir "$dir"
    bash scripts/run_yolo_pair.sh "$py" "$dir" "$weights" "$@" --seed "$seed"
    "$py" -u -m selective_seg.anchor_experiment study "$@" --seed "$seed" \
        --yolo-weights "$weights" --proposal-pooling uniform --output "$dir/yolo-uniform"
done
echo 'ALL VALIDATION STUDIES COMPLETE'
