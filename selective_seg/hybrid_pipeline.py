"""Opt-in specialist-guided routing with paired six-class evaluation."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .benchmark import environment, summarize_latency
from .data import read_sample
from .engine import Pipeline, fingerprint, load_segmenter, sync
from .evaluation import correction_diagnostics
from .geometry import tile_descriptor, tiles
from .hybrid_evidence import (PatchSelector, area, candidates, conditional, evidence,
                              pair_ids, pair_update, seam_statistics, specialist_maps, tile_cap)
from .hybrid_metrics import Metrics, paired_bootstrap
from .hybrid_specialist import load, prediction_support, save_new
from .objectives import confusion_matrix, tensor_miou
from .training import seed_all

ACTION_NAMES = ("specialist", "low", "high")


def config_key(cfg):
    normalized = dict(cfg)
    normalized["hybrid"] = dict(cfg["hybrid"])
    normalized["hybrid"].pop("refine_fraction")
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def choose(gains, costs, allowed, fraction, penalty=0.):
    """Consider each candidate's best positive gain after an optional cost penalty."""
    rated = (gains - penalty * costs).masked_fill(~allowed, -float("inf"))
    value, action = rated.max(1)
    order = value.argsort(descending=True, stable=True)[:tile_cap(fraction, len(value))]
    selected = order[value[order] > 0]
    output = torch.zeros(len(value), dtype=torch.long)
    output[selected] = action[selected] + 1
    return output


class GuidedPipeline:
    def __init__(self, cfg, base, specialist, calibration, device):
        self.cfg, self.base, self.specialist = cfg, base, specialist
        self.calibration, self.device = calibration, device
        self.options = cfg["hybrid"]
        self.pair = pair_ids(cfg)
        self.actions = self.options["actions"]
        self.selector = self.cost = None
        self.dimension = base.coarse.feature_dim + 15 + (3 if self.options["evidence"] == "predictions"
                                else 15 if self.options["evidence"] == "full" else 0)
        self.seams = self.options["neighbourhood"] != "none"
        self.dimension += 8 * self.seams

    def shape(self, grid):
        columns = sum(tile[0] == 0 for tile in grid)
        return len(grid) // columns, columns

    def prepare(self, image, capture_maps=False):
        pc, features = self.base.overview(image)
        grid = tiles(*image.shape[-2:], self.cfg["tile_size"])
        selected, _ = candidates(image, pc, grid, self.cfg)
        chosen = set(selected)
        specialist_results, descriptors, maps = {}, [], {}
        depths = [0, 0, 0]
        for index, tile in enumerate(grid):
            descriptor = tile_descriptor(image[None], pc, features, tile, self.cfg)[0]
            if self.options["evidence"] != "none":
                if index in chosen:
                    coarse = area(pc, tile)
                    support = prediction_support(coarse, self.cfg)
                    probabilities, depth, gap = specialist_maps(
                        self.specialist, image, tile, self.cfg, self.device, support,
                        self.calibration, self.options["exit_policy"])
                    depths[depth] += 1
                    specialist_results[index] = probabilities
                    mode = self.options["evidence"]
                    extra, evidence_maps = evidence(coarse, probabilities, support, gap, depth, mode, self.cfg)
                    if capture_maps:
                        maps[index] = evidence_maps
                else:
                    extra = descriptor.new_zeros(self.dimension - len(descriptor) - 8 * self.seams)
                descriptor = torch.cat((descriptor, extra))
            descriptors.append(descriptor)
        descriptors = torch.stack(descriptors)
        if self.seams:
            descriptors = torch.cat((descriptors, seam_statistics(pc, grid, self.shape(grid))), 1)
        return pc, grid, chosen, specialist_results, descriptors, depths, maps

    def proposed(self, image, pc, tile, action, specialist):
        if action == "specialist":
            return specialist
        raw = self.base.refine(image, pc, tile, 1 if action == "low" else 2, return_details=True)
        return conditional(raw["probabilities"], self.pair)

    def option(self, image, pc, tile, action, specialist):
        coarse = area(pc, tile)
        proposal = self.proposed(image, pc, tile, action, specialist)
        return pair_update(coarse, proposal, self.pair, prediction_support(coarse, self.cfg))

    @torch.inference_mode()
    def predict(self, image, capture_maps=False):
        sync(self.device)
        start = time.perf_counter()
        pc, grid, chosen, predictions, z, depths, maps = self.prepare(image, capture_maps)
        gains = self.selector(z.to(self.device), self.shape(grid)).cpu()
        allowed = torch.tensor([i in chosen for i in range(len(grid))])[:, None]
        selected = choose(gains, self.cost, allowed, self.options["refine_fraction"],
                          self.options["action_time_penalty"])
        final = pc.clone()
        counts = dict.fromkeys(ACTION_NAMES, 0)
        for index, value in enumerate(selected.tolist()):
            if value:
                action = self.actions[value-1]
                tile = grid[index]
                area(final, tile).copy_(self.option(image, pc, tile, action, predictions.get(index)))
                counts[action] += 1
        mask = final.argmax(1)[0]
        sync(self.device)
        elapsed = (time.perf_counter() - start) * 1000
        return final[0], mask, {"end_to_end_ms": elapsed, "tiles": len(grid),
                                "candidate_indices": sorted(chosen), "grid": grid,
                                "specialist_invocations": sum(depths), "exit_counts": depths,
                                "refinement_calls": counts, "selected_tiles": int((selected > 0).sum()),
                                "selected_pixels": sum(tile[2]*tile[3] for tile, value in zip(grid, selected) if value),
                                "maps": maps, "coarse": pc}


def build(cfg, args, device):
    options = cfg["hybrid"]
    requested = {"coarse", *(a for a in options["actions"] if a != "specialist")}
    models, fingerprints, forbidden, tuned = {}, {}, set(), set()
    for stage in sorted(requested):
        path = Path(args.base_run) / f"{stage}.pt"
        models[stage], ck = load_segmenter(path, cfg, stage, device)
        fingerprints[stage] = fingerprint(path)
        forbidden.update(ck["train_paths"])
    base = Pipeline(cfg, models["coarse"], models.get("low"), models.get("high"), device=device)
    specialist, calibration = None, None
    if options["evidence"] != "none":
        if not args.specialist_run:
            raise ValueError("--specialist-run is required")
        path = Path(args.specialist_run) / "specialist.pt"
        specialist, ck = load(cfg, path, device)
        fingerprints["specialist"] = fingerprint(path)
        forbidden.update(ck["train_paths"])
        tuned.update(ck["val_paths"])
        if args.calibration:
            calibration = json.loads(Path(args.calibration).read_text())
            if (calibration["specialist_sha256"] != fingerprints["specialist"]
                    or calibration["coarse_sha256"] != fingerprints["coarse"]
                    or calibration["support_mass"] != options["support_mass"]):
                raise ValueError("Specialist calibration hashes differ")
            fingerprints["exit_calibration"] = fingerprint(args.calibration)
            tuned.update(calibration["val_paths"])
        if options["exit_policy"] in {"confidence", "benefit"} and (calibration is None or options["exit_policy"] not in calibration["thresholds"]):
            raise ValueError("Adaptive inference requires calibrated exits")
    return GuidedPipeline(cfg, base, specialist, calibration, device), fingerprints, forbidden, tuned


@torch.inference_mode()
def cache(pipeline, records, fingerprints, destination):
    cfg = pipeline.cfg
    ignore, classes = cfg["ignore_index"], cfg["classes"]
    rows = []
    for record in records:
        image, truth = read_sample(record, cfg)
        pc, grid, chosen, probabilities, z, _, _ = pipeline.prepare(image)
        coarse = pc.argmax(1)[0]
        coarse_cm = confusion_matrix(coarse, truth, classes, ignore)
        base_score = tensor_miou(coarse_cm.double())
        gains = torch.zeros(len(grid), len(pipeline.actions))
        costs = torch.zeros_like(gains)
        supervised = torch.zeros(len(grid), dtype=torch.bool)
        for index in sorted(chosen):
            tile = grid[index]
            target = area(truth, tile)
            if not (target != ignore).any():
                continue
            supervised[index] = True
            original = confusion_matrix(area(coarse, tile), target, classes, ignore)
            for a, action in enumerate(pipeline.actions):
                sync(pipeline.device)
                start = time.perf_counter()
                proposal = pipeline.option(image, pc, tile, action, probabilities.get(index)).argmax(1)[0]
                sync(pipeline.device)
                costs[index, a] = 1000 * (time.perf_counter()-start)
                revised = confusion_matrix(proposal, target, classes, ignore)
                gains[index, a] = tensor_miou((coarse_cm - original + revised).double()) - base_score
        rows.append({"id": record["id"], "shape": pipeline.shape(grid), "z": z,
                     "supervised": supervised, "gains": gains, "costs": costs})
        print(json.dumps({"image": record["id"], "candidates": len(chosen)}), flush=True)
    save_new(destination, {"config": cfg, "key": config_key(cfg), "provenance": fingerprints,
                           "paths": [r["image"] for r in records], "rows": rows})


def train_selector(cfg, train_path, val_path, destination, device, epochs):
    datasets = [torch.load(path, map_location="cpu", weights_only=True) for path in (train_path, val_path)]
    train, val = datasets
    if any(d["key"] != config_key(cfg) for d in datasets) or train["provenance"] != val["provenance"]:
        raise ValueError("Cache configuration or checkpoint provenance mismatch")
    if set(train["paths"]) & set(val["paths"]):
        raise ValueError("Selector fitting and validation paths overlap")
    training, validation = [[r for r in d["rows"] if r["supervised"].any()] for d in datasets]
    if not training or not validation:
        raise ValueError("Both caches need labeled candidate tiles")
    seed_all(cfg["seed"])
    inputs = training[0]["z"].shape[1]
    selector = PatchSelector(inputs, cfg["hybrid"]["neighbourhood"], len(cfg["hybrid"]["actions"])).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=cfg["training"]["selector_lr"])
    limit = epochs or cfg["training"]["selector_epochs"]
    best, stale = float("inf"), 0

    def loss(row):
        mask = row["supervised"].to(device)
        prediction = selector(row["z"].to(device), row["shape"])[mask]
        return F.smooth_l1_loss(prediction, row["gains"].to(device)[mask], beta=.005)

    for epoch in range(limit):
        selector.train()
        for index in torch.randperm(len(training)).tolist():
            optimizer.zero_grad(set_to_none=True)
            loss(training[index]).backward()
            optimizer.step()
        selector.eval()
        with torch.no_grad():
            score = statistics.mean(float(loss(r)) for r in validation)
        print(json.dumps({"epoch": epoch + 1, "validation_loss": score}), flush=True)
        if score < best:
            best, stale = score, 0
            # Specialist forward is already paid before routing. Merge cost is incremental.
            observed = torch.cat([r["costs"][r["supervised"]] for r in validation])
            best_state = {"model": {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()},
                          "dimension": inputs, "config": cfg, "key": config_key(cfg),
                          "provenance": train["provenance"], "epoch": epoch + 1,
                          "train_paths": train["paths"], "val_paths": val["paths"],
                          "cost_ms": observed.median(0).values.clamp_min(1e-3)}
        else:
            stale += 1
            if stale >= cfg["training"]["patience"]:
                break
    save_new(destination, best_state)


def load_selector(pipeline, path, fingerprints):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["key"] != config_key(pipeline.cfg) or ck["provenance"] != fingerprints or ck["dimension"] != pipeline.dimension:
        raise ValueError("Selector provenance, configuration or dimensions differ")
    pipeline.selector = PatchSelector(ck["dimension"], pipeline.options["neighbourhood"],
                                      len(pipeline.actions)).to(pipeline.device)
    pipeline.selector.load_state_dict(ck["model"])
    pipeline.selector.eval()
    pipeline.cost = ck["cost_ms"]
    return ck


@torch.inference_mode()
def evaluate(pipeline, records, passes, warmup, masks=None):
    cfg, device = pipeline.cfg, pipeline.device
    metrics = Metrics(cfg["classes"], ignore=cfg["ignore_index"])
    calls = dict.fromkeys(ACTION_NAMES, 0)
    selected = selected_pixels = invocations = total = total_pixels = candidate_count = peak = 0
    candidate_errors = all_errors = candidate_confident = all_confident = 0
    depths = [0, 0, 0]
    rows = []
    for i in range(warmup):
        image, _ = read_sample(records[i % len(records)], cfg)
        pipeline.predict(image)
    pooled = {}
    for record in records:
        image, truth = read_sample(record, cfg)
        timings = []
        for _ in range(passes):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            probabilities, mask, details = pipeline.predict(image)
            timings.append(details["end_to_end_ms"])
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))
        if masks is not None:
            import numpy as np
            _, _, saved = pipeline.predict(image, capture_maps=True)
            maps = {f"tile{i}_{kind}": x.numpy() for i, maps in saved["maps"].items() for kind, x in maps.items()}
            np.savez_compressed(masks / f"image_{len(rows):04d}.npz", mask=mask.numpy(),
                                tile_grid=np.array(details["grid"]), **maps)
        cm = metrics.add(probabilities, truth)
        coarse = details["coarse"]
        confidence, original = coarse.max(1)
        diag = correction_diagnostics(original[0], mask, truth, cfg, confidence[0])
        for key in ("corrected_pixels", "damaged_pixels", "confident_errors", "recovered_confident_errors"):
            pooled[key] = pooled.get(key, 0) + diag[key]
        for key in ("correction_pairs_predicted_to_true", "damage_pairs_true_to_predicted"):
            value = torch.tensor(diag[key])
            pooled[key] = pooled.get(key, torch.zeros_like(value)) + value
        is_pair = torch.isin(truth, truth.new_tensor(pipeline.pair))
        errors = is_pair & (original[0] != truth)
        confident = errors & (confidence[0] >= .9)
        all_errors += int(errors.sum())
        all_confident += int(confident.sum())
        for index in details["candidate_indices"]:
            tile = details["grid"][index]
            candidate_errors += int(area(errors, tile).sum())
            candidate_confident += int(area(confident, tile).sum())
        selected += details["selected_tiles"]
        selected_pixels += details["selected_pixels"]
        total_pixels += truth.numel()
        invocations += details["specialist_invocations"]
        candidate_count += len(details["candidate_indices"])
        total += details["tiles"]
        depths = [a+b for a, b in zip(depths, details["exit_counts"])]
        for action, count in details["refinement_calls"].items():
            calls[action] += count
        rows.append({"id": record["id"], "latency_ms": timings, "confusion": cm,
                     **{k: details[k] for k in ("selected_tiles", "specialist_invocations", "exit_counts", "refinement_calls")}})
    pooled = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in pooled.items()}
    summary = metrics.report()
    return {**summary, **summarize_latency([r["latency_ms"] for r in rows]), **pooled,
            "images": rows, "image_ids": [r["id"] for r in rows],
            "refined_tile_count": selected, "refined_tile_fraction": selected/total,
            "refined_pixel_area_fraction": selected_pixels/total_pixels,
            "candidate_tile_fraction": candidate_count/total,
            "candidate_pair_error_recall": candidate_errors/all_errors if all_errors else None,
            "candidate_confident_pair_error_recall": candidate_confident/all_confident if all_confident else None,
            "specialist_invocations": invocations, "exit_counts": depths, "refinement_calls": calls,
            "pixel_accuracy": summary["accuracy"],
            "peak_gpu_bytes": peak if device.type == "cuda" else None, "environment": environment(device),
            "timing_scope": "decoded CPU image to final CPU six-class probabilities and mask; synchronized",
            "budget_scope": "separate per-image candidate and accepted-action tile caps",
            "estimated_incremental_action_ms": pipeline.cost.tolist()}
