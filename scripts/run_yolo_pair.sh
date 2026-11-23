#!/usr/bin/env bash
# Invoked by systemd after both pilots have completed.
set -euo pipefail
py="$1"; root="$2"; weights="$3"; shift 3
"$py" -u -m selective_seg.anchor_experiment study --output "$root/no-yolo" "$@"
"$py" -u -m selective_seg.anchor_experiment study --output "$root/yolo" --yolo-weights "$weights" "$@"
"$py" -m selective_seg.yolo_compare "$root/no-yolo" "$root/yolo" "$root/comparison.json"
