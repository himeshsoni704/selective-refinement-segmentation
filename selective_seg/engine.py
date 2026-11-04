import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .models import Segmenter, OutcomeSelector, freeze
from .geometry import resize_long, tiles, crop_context, center_prediction, tile_descriptor
from .routing import choose_actions


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def config_signature(cfg):
    # Routing can be tuned without invalidating frozen-model outcomes.
    keys = ["classes", "bands", "backbone", "spectral", "tile_size", "border", "low_size",
            "coarse_size", "ignore_index", "image_scale", "array_layout", "mask_colors", "mask_values"]
    return hashlib.sha256(json.dumps({k:cfg.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()


def load_segmenter(path, cfg, stage, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["stage"] != stage or ckpt["signature"] != config_signature(cfg):
        raise ValueError(f"Wrong stage or incompatible configuration: {path}")
    model = Segmenter(cfg, context=stage != "coarse").to(device)
    model.load_state_dict(ckpt["model"])
    return freeze(model), ckpt


class Pipeline:
    def __init__(self, cfg, coarse, low, high, selector=None, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.coarse, self.low, self.high, self.selector = coarse, low, high, selector
        self.provenance = {}

    @classmethod
    def from_run(cls, cfg, run, device, need_selector=False):
        run, device = Path(run), torch.device(device)
        models, provenance = [], {}
        for stage in ("coarse", "low", "high"):
            p = run / f"{stage}.pt"
            model, _ = load_segmenter(p, cfg, stage, device)
            models.append(model)
            provenance[stage] = fingerprint(p)
        selector = None
        if need_selector:
            ckpt = torch.load(run / "selector.pt", map_location="cpu", weights_only=True)
            if ckpt["provenance"] != provenance or ckpt["signature"] != config_signature(cfg):
                raise ValueError("Selector was trained for different segmenters/configuration")
            if ckpt["utility"] != cfg["utility"]:
                raise ValueError("Utility changed: retrain selector and recalibrate routing")
            selector = OutcomeSelector(models[0].feature_dim + cfg["classes"] + 9, cfg["classes"]).to(device)
            selector.load_state_dict(ckpt["model"])
            freeze(selector)
        out = cls(cfg, *models, selector=selector, device=device)
        out.provenance = provenance
        return out

    @torch.no_grad()
    def overview(self, image):
        # UHR image remains on CPU. Only thumbnails and selected crops enter GPU memory.
        if image.ndim == 3:
            image = image[None]
        small = resize_long(image, self.cfg["coarse_size"]).to(self.device)
        with amp(self.device):
            result = self.coarse(small)
        # Match segmentation convention: interpolate logits, then softmax/argmax.
        logits = F.interpolate(result["logits"].float().cpu(), image.shape[-2:],
                               mode="bilinear", align_corners=False)
        probabilities = logits.softmax(1)
        return probabilities, result["features"].float().cpu()

    @torch.no_grad()
    def refine(self, image, pc, tile, action):
        cfg = self.cfg
        if image.ndim == 3:
            image = image[None]
        crop = crop_context(image, tile, cfg["tile_size"], cfg["border"])
        context = crop_context(pc, tile, cfg["tile_size"], cfg["border"])
        if action == 1:
            crop = F.interpolate(crop, (cfg["low_size"],)*2, mode="bilinear", align_corners=False, antialias=True)
            context = F.interpolate(context, crop.shape[-2:], mode="bilinear", align_corners=False)
            model = self.low
        elif action == 2:
            model = self.high
        else:
            raise ValueError("refine action must be 1 or 2")
        with amp(self.device):
            out = model(crop.to(self.device), context.to(self.device))
        return center_prediction(out["logits"].float(), tile, cfg).argmax(1)[0].cpu()

    @torch.no_grad()
    def predict(self, image):
        if self.selector is None:
            raise ValueError("Load a trained selector first")
        sync(self.device)
        started = time.perf_counter()
        pc, features = self.overview(image)
        grid = tiles(*image.shape[-2:], self.cfg["tile_size"])
        z = torch.cat([tile_descriptor(image[None], pc, features, t, self.cfg) for t in grid])
        pred = self.selector(z.to(self.device))
        actions, q, utility = choose_actions(pred, self.cfg)
        output = pc.argmax(1)[0].clone()
        records = []
        for i, tile in enumerate(grid):
            y, x, h, w = tile
            a = int(actions[i])
            if a:
                output[y:y+h, x:x+w] = self.refine(image, pc, tile, a)
            records.append({"tile": list(tile), "action": ["skip","low","high"][a],
                            "p_any_benefit": float(q[i]), "utility_low_high": utility[i].cpu().tolist(),
                            "corrected_fraction_low_high": pred["corrected"][i].cpu().tolist(),
                            "damaged_fraction_low_high": pred["damaged"][i].cpu().tolist(),
                            "expected_tp_counts_coarse_low_high": (pred["tp"][i]*h*w).cpu().tolist(),
                            "expected_fp_counts_coarse_low_high": (pred["fp"][i]*h*w).cpu().tolist(),
                            "expected_fn_counts_coarse_low_high": (pred["fn"][i]*h*w).cpu().tolist()})
        sync(self.device)
        elapsed = (time.perf_counter()-started)*1000
        return output, {"end_to_end_ms": elapsed, "tiles": records}
