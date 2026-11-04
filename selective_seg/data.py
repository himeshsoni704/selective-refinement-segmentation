import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["classes"] < 2:
        raise ValueError("Use at least 2 classes, including background for binary tasks")
    if not (0 <= cfg["routing"]["T2"] < cfg["routing"]["T1"] <= 1):
        raise ValueError("Require 0 <= T2 < T1 <= 1")
    if cfg["low_size"] >= cfg["tile_size"] + 2 * cfg["border"]:
        raise ValueError("low_size must be smaller than the original context crop")
    if cfg["low_size"] < 32 or cfg["coarse_size"] < 32:
        raise ValueError("SegFormer inputs must be at least 32 pixels for this setup")
    if cfg["ignore_index"] in range(cfg["classes"]):
        raise ValueError("ignore_index must not be a class ID")
    w = cfg["utility"]["class_weights"]
    if len(w) != cfg["classes"] or min(w) < 0 or sum(w) <= 0:
        raise ValueError("class_weights must match classes, be nonnegative and sum positive")
    if cfg["routing"]["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= cfg["routing"]["max_fraction"] <= 1:
        raise ValueError("max_fraction must be in [0, 1]")
    return cfg


def read_manifest(path):
    path = Path(path).resolve()
    records = json.loads(path.read_text())
    seen_ids, seen_images = set(), set()
    for r in records:
        if r["split"] not in {"train", "selector", "val", "test"}:
            raise ValueError("split must be train, selector, val, or test")
        if r["id"] in seen_ids:
            raise ValueError("Duplicate image ID: " + r["id"])
        seen_ids.add(r["id"])
        for field in ("image", "mask"):
            p = (path.parent / r[field]).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            r[field] = str(p)
        if r["image"] in seen_images:
            raise ValueError("Same image appears more than once/across splits")
        seen_images.add(r["image"])
    return records


def read_image(path, cfg):
    if Path(path).suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
        if cfg.get("array_layout", "CHW") == "HWC":
            a = a.transpose(2, 0, 1)
    else:
        if cfg["bands"] != 3:
            raise ValueError("Export spectral images to CHW float32 .npy; ordinary images are RGB")
        a = np.asarray(Image.open(path).convert("RGB")).transpose(2, 0, 1)
    if a.ndim != 3 or a.shape[0] != cfg["bands"]:
        raise ValueError(f"Expected {cfg['bands']} bands in CHW, got {a.shape}")
    x = torch.from_numpy(a.astype(np.float32).copy())
    scale = torch.as_tensor(cfg["image_scale"], dtype=torch.float32)
    if scale.ndim:
        scale = scale[:, None, None]
    if (scale <= 0).any():
        raise ValueError("image_scale must be positive")
    x = x / scale
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1.001:
        raise ValueError("Inputs must map to finite [0,1] using training-set/sensor image_scale")
    return x


def read_sample(record, cfg):
    x = read_image(record["image"], cfg)
    p = record["mask"]
    raw = np.load(p, allow_pickle=False) if Path(p).suffix.lower() == ".npy" else np.asarray(Image.open(p))
    if cfg.get("mask_colors"):
        if raw.ndim != 3 or raw.shape[2] < 3:
            raise ValueError("mask_colors requires an RGB mask")
        y = np.full(raw.shape[:2], cfg["ignore_index"], dtype=np.int64)
        recognized = np.zeros(y.shape, bool)
        for rgb, label in cfg["mask_colors"].items():
            matched = (raw[..., :3] == np.array([int(v) for v in rgb.split(',')])).all(-1)
            y[matched], recognized[matched] = label, True
        if not recognized.all():
            raise ValueError("Mask contains an unmapped RGB color")
    elif cfg.get("mask_values"):
        if raw.ndim != 2:
            raise ValueError("mask_values requires a single-channel mask")
        y = np.full(raw.shape, cfg["ignore_index"], dtype=np.int64)
        recognized = np.zeros(y.shape, bool)
        for value, label in cfg["mask_values"].items():
            matched = raw == int(value)
            y[matched], recognized[matched] = label, True
        if not recognized.all():
            raise ValueError("Mask contains an unmapped value")
    else:
        if raw.ndim != 2:
            raise ValueError("Use indexed masks, mask_values, or mask_colors")
        y = raw.astype(np.int64)
    if tuple(y.shape) != tuple(x.shape[-2:]):
        raise ValueError("Image and mask dimensions differ")
    y = torch.from_numpy(y.copy())
    if not (((y >= 0) & (y < cfg["classes"])) | (y == cfg["ignore_index"])).all():
        raise ValueError("Mask labels out of range")
    return x, y
