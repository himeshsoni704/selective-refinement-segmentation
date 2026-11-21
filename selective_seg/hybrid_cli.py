"""Isolated CLI for hybrid specialist experiments. See HYBRID.md."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch

from .data import load_config, read_manifest, read_sample
from .engine import Pipeline, amp, fingerprint, load_segmenter, sync
from .geometry import tiles
from .hybrid_backbone import HybridSpecialist
from .hybrid_evidence import area, crop_input, tile_cap
from .hybrid_metrics import paired_bootstrap
from .hybrid_pipeline import ACTION_NAMES, build, cache, evaluate, load_selector, train_selector
from .hybrid_specialist import calibrate, evaluate_specialist, load, train, training_loss, write_json
from .specialist import binary_target, settings

BUDGETS = (.05, .10, .20, .30, .50, 1.)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train-specialist", "calibrate", "specialist",
                                             "cache", "train-selector", "evaluate"))
    parser.add_argument("--config", default="configs/deepglobe_specialist.json")
    parser.add_argument("--manifest")
    parser.add_argument("--base-run")
    parser.add_argument("--specialist-run")
    parser.add_argument("--calibration")
    parser.add_argument("--selector-run")
    parser.add_argument("--output")
    parser.add_argument("--split", choices=("selector", "val", "test"), default="val")
    parser.add_argument("--backbones", choices=("combined", "resnet", "efficientnet"), default="combined")
    parser.add_argument("--fusion", choices=("concat", "fpn", "inception"), default="fpn")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--evidence", choices=("none", "predictions", "full"), default="full")
    parser.add_argument("--neighbourhood", choices=("none", "mean", "learned", "conv"), default="none")
    parser.add_argument("--actions", nargs="+", choices=ACTION_NAMES, default=["low", "high"])
    parser.add_argument("--train-exits", action="store_true")
    parser.add_argument("--exit-policy", choices=("full", "shallow", "intermediate", "confidence", "benefit"), default="full")
    parser.add_argument("--exit-iou-tolerance", type=float, default=.005)
    parser.add_argument("--candidate-fraction", type=float, default=.30)
    parser.add_argument("--min-pair-mass", type=float, default=.10)
    parser.add_argument("--support-mass", type=float, default=.35)
    parser.add_argument("--refine-fraction", type=float, default=.20)
    parser.add_argument("--all-budgets", action="store_true")
    parser.add_argument("--action-time-penalty", type=float, default=0.)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--paired-baseline")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--pilot", action="store_true", help="Run at most four batch-one training pilot steps")
    args = parser.parse_args()
    for name in ("candidate_fraction", "min_pair_mass", "support_mass", "refine_fraction", "exit_iou_tolerance"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    if (args.passes < 1 or args.warmup < 0 or args.bootstrap < 1
            or args.steps is not None and args.steps < 1 or args.epochs is not None and args.epochs < 1):
        parser.error("Invalid pass, warmup, bootstrap, step or epoch count")
    if len(set(args.actions)) != len(args.actions) or ("specialist" in args.actions and args.evidence == "none"):
        parser.error("Use unique actions and enable evidence for the specialist action")
    if not math.isfinite(args.action_time_penalty) or args.action_time_penalty < 0:
        parser.error("Action time penalty must be finite and nonnegative")
    if args.exit_policy != "full" and not args.train_exits:
        parser.error("Use --train-exits to identify the checkpoint architecture for adaptive inference")
    if args.command != "preflight" and not args.output:
        parser.error("--output is required and must identify a new run directory")
    if args.command != "train-selector" and not args.manifest:
        parser.error("--manifest is required")
    if args.command in {"calibrate", "specialist", "cache", "evaluate"} and not args.base_run:
        parser.error("--base-run is required")
    if args.command in {"calibrate", "specialist"} and not args.specialist_run:
        parser.error("--specialist-run is required")
    if args.command in {"train-selector", "evaluate"} and not args.selector_run:
        parser.error("--selector-run is required")
    return args


def configuration(args):
    cfg = load_config(args.config)
    cfg["research"] = {}
    cfg["specialist"] = {**cfg.get("specialist", {}), "backbones": args.backbones, "levels": "all"}
    if args.input_size is not None:
        cfg["specialist"]["input_size"] = args.input_size
    if args.no_pretrained:
        cfg["specialist"]["pretrained"] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    names = ("fusion", "precision", "train_exits", "exit_policy", "exit_iou_tolerance", "evidence",
             "neighbourhood", "actions", "candidate_fraction", "min_pair_mass", "support_mass",
             "refine_fraction", "action_time_penalty")
    cfg["hybrid"] = {name: getattr(args, name) for name in names}
    spec = settings(cfg)
    if cfg["classes"] != 6 or (spec["agriculture"], spec["rangeland"]) != (1, 2):
        raise ValueError("Expected six DeepGlobe classes with agriculture=1 and rangeland=2")
    return cfg


def split(records, name, forbidden=()):
    chosen = [r for r in records if r["split"] == name]
    if not chosen:
        raise ValueError(f"Missing {name} split")
    if {r["image"] for r in chosen} & set(forbidden):
        raise ValueError(f"{name} overlaps fitting or calibration data")
    return chosen


def preflight(cfg, args, records, device):
    model = HybridSpecialist(cfg).to(device)
    spec = settings(cfg)
    parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
    base_disk_bytes = {}
    if args.base_run:
        for name in ("coarse", "low", "high"):
            path = Path(args.base_run)/f"{name}.pt"
            if path.is_file():
                base_disk_bytes[name] = path.stat().st_size
    sizes = {name: sum(Path(r["image"]).stat().st_size + Path(r["mask"]).stat().st_size
                       for r in records if r["split"] == name) for name in ("train", "selector", "val", "test")}
    location = Path(args.output or ".").resolve()
    while not location.exists():
        location = location.parent
    disk = shutil.disk_usage(location)
    free_gpu = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
    print(json.dumps({"device": str(device), "free_gpu_bytes": free_gpu,
                      "free_disk_bytes": disk.free, "existing_dataset_bytes": sizes,
                      "specialist_parameter_bytes": parameter_bytes,
                      "existing_base_checkpoint_disk_bytes": base_disk_bytes,
                      "approximate_specialist_checkpoint_disk_bytes": parameter_bytes,
                      "rough_parameter_gradient_adam_bytes": parameter_bytes * 4,
                      "feature_spec": model.feature_spec,
                      "central_tile_size": cfg["tile_size"], "specialist_input": spec["input_size"],
                      "example_2448_grid_candidate_cap": tile_cap(args.candidate_fraction,
                          len(tiles(2448, 2448, cfg["tile_size"]))),
                      "estimate_scope": "Parameter estimate excludes activations, CUDA workspace, base models and I/O."}, indent=2))
    if not args.pilot:
        return
    if args.selector_run:
        del model
        pipeline, provenance, _, _ = build(cfg, args, device)
        load_selector(pipeline, Path(args.selector_run)/"selector.pt", provenance)
        validation = split(records, "val")
        image, _ = read_sample(validation[0], cfg)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pipeline.predict(image)
        measured = [pipeline.predict(image)[2]["end_to_end_ms"] for _ in range(3)]
        repeats = args.passes * len(validation) + args.warmup
        print(json.dumps({"guided_validation_image_ms": measured,
                          "screened_validation_seconds_at_requested_passes": sum(measured) / 3 * repeats / 1000,
                          "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if free_gpu is not None else None,
                          "scope": "One validation shape, one warmup, three repetitions. Excludes decoding, metrics, map writing and budget sweeps."}, indent=2))
        return
    image, target, tile = None, None, None
    for record in split(records, "train"):
        sample, truth = read_sample(record, cfg)
        for candidate in tiles(*truth.shape, cfg["tile_size"]):
            binary = binary_target(area(truth, candidate), cfg)
            if (binary != 255).any():
                image, target, tile = sample, binary[None].to(device), candidate
                break
        if image is not None:
            break
    if image is None:
        raise ValueError("No labeled training crop for pilot")
    optimizer = torch.optim.AdamW(model.parameters())
    if free_gpu is not None:
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(4):
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            loss = training_loss(model, crop_input(image, tile, cfg, device), target, tile, cfg, None)
        loss.backward()
        optimizer.step()
        sync(device)
        timings.append(time.perf_counter()-start)
    model.eval()
    with torch.inference_mode():
        sync(device)
        start = time.perf_counter()
        from .hybrid_evidence import specialist_maps
        for _ in range(4):
            specialist_maps(model, image, tile, cfg, device, torch.ones_like(target, dtype=torch.bool))
        sync(device)
        tile_time = (time.perf_counter()-start) / 4
    train_time = sum(timings[1:])/3
    from PIL import Image
    import numpy as np
    def image_tile_count(record):
        path = record["image"]
        if Path(path).suffix.lower() == ".npy":
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
            h, w = shape[:2] if cfg["array_layout"] == "HWC" else shape[-2:]
        else:
            with Image.open(path) as file:
                w, h = file.size
        return len(tiles(h, w, cfg["tile_size"]))
    val_tiles = sum(image_tile_count(r) for r in records if r["split"] == "val")
    steps = args.steps or spec["steps"]
    validation_count = math.ceil(steps / spec["validate_every"])
    print(json.dumps({"pilot_step_seconds": train_time, "pilot_full_tile_seconds": tile_time,
                      "pilot_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if free_gpu is not None else None,
                      "train_plus_validation_seconds_screen": steps*train_time + validation_count*val_tiles*tile_time,
                      "validation_tiles": val_tiles, "train_steps": steps,
                      "estimate_scope": "Four batch-one pilot iterations. Excludes decode, scanning, downloads, baseline models and other load."}, indent=2))


def main():
    args = arguments()
    cfg = configuration(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this environment")
    records = read_manifest(args.manifest) if args.manifest else []
    if args.command == "preflight":
        preflight(cfg, args, records, device)
        return
    if args.command == "calibrate" and args.split != "val":
        raise ValueError("Specialist calibration requires --split val")
    if args.command == "cache" and args.split not in {"selector", "val"}:
        raise ValueError("Cache fitting targets only on selector or val")
    if args.command in {"evaluate", "specialist"} and args.split not in {"val", "test"}:
        raise ValueError("Evaluation requires val or test")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "config.json", cfg)
    if args.command == "train-specialist":
        train(cfg, records, destination, device, args.steps)
        return
    if args.command == "train-selector":
        root = Path(args.selector_run)
        train_selector(cfg, root/"selector"/"cache.pt", root/"val"/"cache.pt",
                       destination/"selector.pt", device, args.epochs)
        return
    if args.command in {"calibrate", "specialist"}:
        path = Path(args.specialist_run) / "specialist.pt"
        coarse_path = Path(args.base_run) / "coarse.pt"
        model, ck = load(cfg, path, device)
        coarse_model, coarse_ck = load_segmenter(coarse_path, cfg, "coarse", device)
        coarse = Pipeline(cfg, coarse_model, None, None, device=device)
        hashes = {"specialist_sha256": fingerprint(path), "coarse_sha256": fingerprint(coarse_path)}
        calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
        if calibration and (any(calibration[k] != v for k, v in hashes.items())
                            or calibration["support_mass"] != args.support_mass):
            raise ValueError("Calibration refers to a different checkpoint")
        if args.exit_policy in {"confidence", "benefit"} and args.command == "specialist" and (
                calibration is None or args.exit_policy not in calibration["thresholds"]):
            raise ValueError("Adaptive inference requires calibrated exits")
        forbidden = set(ck["train_paths"]) | set(coarse_ck["train_paths"])
        if args.split == "test":
            forbidden |= set(ck["val_paths"]) | set((calibration or {}).get("val_paths", []))
        chosen = split(records, args.split, forbidden)
        result = (calibrate(model, cfg, chosen, device, coarse) if args.command == "calibrate" else
                  evaluate_specialist(model, cfg, chosen, device, coarse, calibration, args.exit_policy, args.passes))
        if args.command == "calibrate":
            result["val_paths"] = [r["image"] for r in chosen]
            result["support_mass"] = args.support_mass
        result.update(hashes, split=args.split, exit_policy=args.exit_policy, feature_spec=model.feature_spec)
        write_json(destination / ("calibration.json" if args.command == "calibrate" else "metrics.json"), result)
        return
    pipeline, provenance, forbidden, tuned = build(cfg, args, device)
    if args.command == "cache":
        selected = split(records, args.split, forbidden | (tuned if args.split == "selector" else set()))
        cache(pipeline, selected, provenance, destination / "cache.pt")
        return
    selector_path = Path(args.selector_run) / "selector.pt"
    ck = load_selector(pipeline, selector_path, provenance)
    forbidden.update(ck["train_paths"])
    if args.split == "test":
        forbidden.update(set(ck["val_paths"]) | tuned)
    chosen = split(records, args.split, forbidden)
    baseline = json.loads(Path(args.paired_baseline).read_text()) if args.paired_baseline else None
    if baseline is not None and baseline.get("split") not in {None, args.split}:
        raise ValueError("Paired baseline uses a different split")
    for budget in BUDGETS if args.all_budgets else (args.refine_fraction,):
        pipeline.options["refine_fraction"] = budget
        output = destination / f"budget_{budget:g}" if args.all_budgets else destination
        if args.all_budgets:
            output.mkdir()
        masks = output / "maps" if args.save_maps else None
        if masks is not None:
            masks.mkdir()
        report = evaluate(pipeline, chosen, args.passes, args.warmup, masks=masks)
        report.update(split=args.split, budget=budget, config=cfg, provenance={
            **provenance, "selector": fingerprint(selector_path)})
        if baseline is not None:
            report["paired"] = paired_bootstrap(report["images"], baseline["images"], args.bootstrap, cfg["seed"])
        write_json(output / "metrics.json", report)
        print(json.dumps({k: report[k] for k in ("budget", "mIoU", "mean_image_latency_ms",
                       "refined_tile_fraction", "specialist_invocations")}), flush=True)


if __name__ == "__main__":
    main()
