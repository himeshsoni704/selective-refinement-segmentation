"""Independent specialist fitting and validation-only exit calibration."""
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import read_sample
from .engine import amp, fingerprint, sync
from .geometry import center_prediction, tiles
from .hybrid_backbone import HybridSpecialist
from .hybrid_evidence import area, crop_input, exit_score, pair_ids, specialist_maps
from .hybrid_metrics import Metrics
from .specialist import binary_target, counts_metrics, masked_loss, settings, signature as base_signature
from .training import seed_all


def write_json(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def save_new(path, value):
    with Path(path).open("xb") as file:
        torch.save(value, file)


def spec_signature(cfg):
    return (base_signature(cfg), cfg["hybrid"]["fusion"], cfg["hybrid"]["train_exits"])


def load(cfg, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("signature") != spec_signature(cfg) or checkpoint.get("stage") != "hybrid_specialist":
        raise ValueError("Specialist checkpoint/configuration mismatch")
    model = HybridSpecialist(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    return model, checkpoint


def prediction_support(coarse, cfg):
    ids = pair_ids(cfg)
    return coarse[:, ids].sum(1) >= cfg["hybrid"]["support_mass"]


def training_loss(model, crop, target, tile, cfg, weights):
    packets = list(model.packets(crop))
    logits = [center_prediction(packet[0].float(), tile, cfg) for packet in packets]
    loss = masked_loss(logits[-1], target, weights)
    if model.has_exits:
        valid = target != 255
        deep_accuracy = (logits[-1].detach().argmax(1)[valid] == target[valid]).float().mean()
        for early, (_, predicted_gain, _) in zip(logits[:-1], packets[:-1]):
            accuracy = (early.detach().argmax(1)[valid] == target[valid]).float().mean()
            loss = loss + .30 * masked_loss(early, target, weights)
            loss = loss + F.smooth_l1_loss(predicted_gain.float(), (deep_accuracy-accuracy).expand_as(predicted_gain))
    return loss


@torch.inference_mode()
def evaluate_specialist(model, cfg, records, device, coarse, calibration, policy, passes):
    model.eval()
    metrics = Metrics(2)
    exit_counts = [0, 0, 0]
    rows = []
    peak = 0
    # Warm each active end-to-end specialist path on a real crop.
    image, _ = read_sample(records[0], cfg)
    tile = tiles(*image.shape[-2:], cfg["tile_size"])[0]
    pc = coarse.overview(image)[0]
    specialist_maps(model, image, tile, cfg, device, prediction_support(area(pc, tile), cfg),
                    calibration, policy)
    for record in records:
        image, truth = read_sample(record, cfg)
        measurements = []
        cm = torch.zeros(2, 2, dtype=torch.long)
        for repeat in range(passes):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start = time.perf_counter()
            pc = coarse.overview(image)[0]
            grid = tiles(*image.shape[-2:], cfg["tile_size"])
            predictions = []
            for tile in grid:
                support = prediction_support(area(pc, tile), cfg)
                probabilities, depth, _ = specialist_maps(
                    model, image, tile, cfg, device, support, calibration, policy)
                predictions.append((tile, probabilities, depth))
            sync(device)
            measurements.append(1000 * (time.perf_counter() - start))
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))
        for tile, probabilities, depth in predictions:
            exit_counts[depth] += 1
            cm += torch.tensor(metrics.add(probabilities[0], binary_target(area(truth, tile), cfg)))
        rows.append({"id": record["id"], "latency_ms": measurements, "confusion": cm.tolist()})
    from .benchmark import summarize_latency
    return {**metrics.report(), **summarize_latency([r["latency_ms"] for r in rows]),
            "images": rows, "exit_counts": exit_counts, "peak_gpu_bytes": peak if device.type == "cuda" else None,
            "timing_scope": "all tile crops and specialist outputs; includes coarse for exit support in all policies",
            "exit_calibration": calibration}


@torch.inference_mode()
def validate_model(model, cfg, records, device):
    model.eval()
    metrics = Metrics(2)
    for record in records:
        image, truth = read_sample(record, cfg)
        for tile in tiles(*truth.shape, cfg["tile_size"]):
            crop = crop_input(image, tile, cfg, device)
            with amp(device):
                logits = center_prediction(model(crop).float(), tile, cfg)
            metrics.add(logits.softmax(1)[0], binary_target(area(truth, tile), cfg))
    return metrics.report()


def train(cfg, records, destination, device, steps):
    s = settings(cfg)
    train_records = [record for record in records if record["split"] == "train"]
    val = [record for record in records if record["split"] == "val"]
    if not train_records or not val:
        raise ValueError("Training requires train and val splits")
    seed_all(cfg["seed"])
    eligible, counts = [], torch.zeros(2, dtype=torch.long)
    for record in train_records:
        _, truth = read_sample(record, cfg)
        target = binary_target(truth, cfg)
        counts += torch.bincount(target[target != 255], minlength=2)
        eligible.extend((record, tile) for tile in tiles(*truth.shape, cfg["tile_size"])
                        if (area(target, tile) != 255).any())
    if (counts == 0).any():
        raise ValueError("Both binary classes must occur on train")
    weights = (counts.max() / counts).sqrt() if s["class_weighting"] == "sqrt_inverse" else torch.ones(2)
    weights = weights.clamp(max=s["max_class_weight"]).to(device)
    model = HybridSpecialist(cfg, initialize_pretrained=True).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=s["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1
    total = s["steps"] if steps is None else steps
    for step in range(1, total + 1):
        model.train()
        record, tile = random.choice(eligible)
        image, truth = read_sample(record, cfg)
        target = binary_target(area(truth, tile), cfg)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp(device):
            loss = training_loss(model, crop_input(image, tile, cfg, device), target, tile, cfg, weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite specialist loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
        if step % s["validate_every"] == 0 or step == total:
            report = validate_model(model, cfg, val, device)
            score = report["mIoU"]
            if score is None:
                raise ValueError("No supervised agriculture/rangeland pixels on val")
            if score > best:
                best = score
                temporary = destination / "specialist.partial"
                torch.save({"stage": "hybrid_specialist", "signature": spec_signature(cfg),
                            "config": cfg, "model": model.state_dict(), "feature_spec": model.feature_spec,
                            "step": step, "train_paths": [r["image"] for r in train_records],
                            "val_paths": [r["image"] for r in val], "class_counts": counts,
                            "class_weights": weights.cpu()}, temporary)
                temporary.replace(destination / "specialist.pt")
            print(json.dumps({"step": step, "val_binary_mIoU": score, "best": best}), flush=True)
    write_json(destination / "training.json", {"best_val_binary_mIoU": best, "steps": total,
               "feature_spec": model.feature_spec, "parameters": sum(p.numel() for p in model.parameters()),
               "specialist_sha256": fingerprint(destination / "specialist.pt")})


@torch.inference_mode()
def calibrate(model, cfg, records, device, coarse):
    """Fit temperature and exit thresholds only on the existing validation split."""
    temperatures = [.67, .8, 1., 1.2, 1.5, 2., 3.]
    loss = torch.zeros(3, len(temperatures), dtype=torch.double)
    rows = []
    grid_confidence = [.6, .7, .8, .9, .95, .99, 2.]
    grid_benefit = [-.10, -.05, -.02, -.01, 0., .02, 2.]
    for record in records:
        image, truth = read_sample(record, cfg)
        pc, _ = coarse.overview(image)
        for tile in tiles(*truth.shape, cfg["tile_size"]):
            target = binary_target(area(truth, tile), cfg)
            valid = target != 255
            if not valid.any():
                continue
            support = prediction_support(area(pc, tile), cfg)
            crop = crop_input(image, tile, cfg, device)
            with amp(device):
                packets = list(model.packets(crop))
            cms, confidences, benefits = [], [], []
            for index, (logits, gain, _) in enumerate(packets):
                depth = index if gain is not None else 2
                logits = center_prediction(logits.float(), tile, cfg).cpu()
                sample = logits[0, :, valid].T
                stride = max(1, math.ceil(len(sample) / 128))
                targets = target[valid][::stride]
                for i, temperature in enumerate(temperatures):
                    loss[depth, i] += F.cross_entropy(sample[::stride] / temperature, targets, reduction="sum")
                from .specialist import confusion
                cms.append(confusion(logits.argmax(1)[0], target))
                if gain is not None:
                    # Cache the confidence under each candidate temperature; no full image logits retained.
                    confidences.append([exit_score((logits / t).softmax(1), support, gain, "confidence")
                                        for t in temperatures])
                    benefits.append(exit_score(logits.softmax(1), support, gain, "benefit"))
            rows.append((cms, confidences, benefits))
    if not rows:
        raise ValueError("No A/R validation labels for calibration")
    best_temperature = loss.argmin(1).tolist()
    result = {"temperatures": [temperatures[i] for i in best_temperature], "thresholds": {},
              "val_tiles": len(rows), "val_binary_miou_tolerance": cfg["hybrid"]["exit_iou_tolerance"]}
    if not model.has_exits:
        result["temperatures"][:2] = [1., 1.]
        return result
    counts = torch.stack([torch.stack(row[0]) for row in rows])
    deep_score = counts_metrics(counts[:, 2].sum(0))["mIoU"]
    for policy, choices in [("confidence", grid_confidence), ("benefit", grid_benefit)]:
        scores = torch.tensor([[row[1][i][best_temperature[i]] if policy == "confidence" else row[2][i]
                                for i in range(2)] for row in rows])
        best = None
        for first in choices:
            for middle in choices:
                depth = torch.where(scores[:, 0] >= first, 0, torch.where(scores[:, 1] >= middle, 1, 2))
                cm = counts[torch.arange(len(counts)), depth].sum(0)
                score = counts_metrics(cm)["mIoU"]
                key = (-float(depth.float().mean()), score)
                if score >= deep_score - cfg["hybrid"]["exit_iou_tolerance"] and (best is None or key > best[0]):
                    best = (key, [first, middle], torch.bincount(depth, minlength=3).tolist())
        result["thresholds"][policy] = best[1]
        result[f"{policy}_val_exit_counts"] = best[2]
    return result
