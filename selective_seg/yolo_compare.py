"""Paired YOLO ablation reports from two validation studies."""
import argparse
import json
from pathlib import Path

from .diagnostics import write
from .hybrid_metrics import paired_bootstrap


def load(path):
    return json.loads(path.read_text())


def compare(baseline, guided):
    a, b = (load(d/"protocol.json") for d in (baseline, guided))
    if a.get("yolo") or not b.get("yolo"):
        raise ValueError("Expected baseline followed by YOLO study")
    for key in ("config","splits","manifest_sha256","checkpoints","specialist_sha256","views","actions","code"):
        if a[key] != b[key]:
            raise ValueError(f"Incompatible studies: {key}")
    for key in ("steps","epochs","seed","budgets","passes","candidate_fraction","device"):
        if a["args"][key] != b["args"][key]:
            raise ValueError(f"Incompatible settings: {key}")
    rows = []
    for path in sorted(guided.glob("*.json")):
        right = load(path)
        if "images" not in right:
            continue
        left = load(baseline/path.name)
        counts = {}
        for key in ("correction_pairs_predicted_to_true","damage_pairs_true_to_predicted"):
            counts[key+"_agriculture_to_rangeland_delta"] = (
                sum(r[key][1][2] for r in right["images"])-sum(r[key][1][2] for r in left["images"]))
        classes = zip(left["per_class_IoU"],right["per_class_IoU"])
        rows.append(dict(mode=path.stem,baseline_mIoU=left["mIoU"],yolo_mIoU=right["mIoU"],
                         delta_mIoU=right["mIoU"]-left["mIoU"],per_class_IoU_delta=[
                             None if x is None or y is None else y-x for x,y in classes],
                         baseline_mean_ms=left["mean_ms"],yolo_mean_ms=right["mean_ms"],
                         baseline_p95_ms=left["p95_ms"],yolo_p95_ms=right["p95_ms"],
                         detector_calls=right["detector_calls"],retained_rois=right["retained_proposals"],
                         empty_calls=right["empty_detector_calls"],**counts,
                         **paired_bootstrap(right["images"],left["images"],1000,b["args"]["seed"])))
    return dict(comparisons=rows, timing_note="Same requested budget. Compare measured latencies across operating points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","guided","output"):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output")
    write(args.output,compare(args.baseline,args.guided))
