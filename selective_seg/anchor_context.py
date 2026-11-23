"""Concentric context acquisitions. All outputs belong to one valid tile."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .engine import amp
from .hybrid_evidence import area

# Footprint in original pixels, encoder input side. Detail uses the frozen high
# refiner. Only context branches share reader weights.
VIEWS = {"medium": (768, 384), "large": (1024, 256)}
ACTIONS = ("keep", "detail", "medium", "large", "scene", "all")
SOURCES = ("medium", "large", "scene")


def window(tile, side):
    y, x, h, w = tile
    return y + (h - side) // 2, x + (w - side) // 2


def crop(x, tile, side):
    """Replicate outside the image; centre on valid edge tiles too."""
    y, x0 = window(tile, side)
    height, width = x.shape[-2:]
    patch = x[..., max(y, 0):min(y + side, height), max(x0, 0):min(x0 + side, width)]
    return F.pad(patch, (max(-x0, 0), max(x0 + side - width, 0),
                         max(-y, 0), max(y + side - height, 0)), mode="replicate")


def resize(x, size):
    return F.interpolate(x, size, mode="bilinear", align_corners=False, antialias=True)


def high_logits(base, image, pc, tile, side=640, network_side=None, control=False):
    rgb = crop(image[None], tile, side)
    if control:
        pad = (side - 640) // 2
        rgb = F.pad(rgb[..., pad:-pad, pad:-pad], (pad,) * 4, mode="replicate")
    prior = crop(pc, tile, side)
    if network_side and network_side != side:
        rgb = resize(rgb, (network_side,) * 2)
        prior = resize(prior, (network_side,) * 2)
    with torch.no_grad(), amp(base.device):
        logits = base.high(rgb.to(base.device), prior.to(base.device))["logits"]
    logits = F.interpolate(logits.float(), (side, side), mode="bilinear", align_corners=False)
    y, x = window(tile, side)
    return area(logits, (tile[0]-y, tile[1]-x, *tile[2:]))


class ContextReader(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.proposals = None
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, 5, stride=4, padding=2), nn.ReLU(),
                                     nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU())
        self.projection = nn.Linear(16 * 4 * 4, 24)
        self.scale = nn.Embedding(3, 24)
        self.weight = nn.Linear(24, 1)
        self.head = nn.Sequential(nn.Conv2d(classes + 24, 24, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(24, classes, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def scene_region(self, image, tile, cache):
        # One full-scene encode. Read a scene-sized box around each anchor
        # from the feature lattice; outside-image locations contribute zeros.
        if "scene" not in cache:
            thumb = resize(image[None], (160, 160)).to(self.scale.weight.device)
            cache["scene"] = self.encoder(thumb)
        feature = cache["scene"]
        h, w = image.shape[-2:]
        y, x, th, tw = tile
        # 2*max(H,W) contains the full scene from any anchor centre.
        extent = 2 * max(h, w)
        # Use even cells with ROI pooling for better spatial coverage.
        offset = ((torch.arange(32, device=feature.device) + .5) / 32 - .5) * extent
        yy, xx = torch.meshgrid((y + th/2 + offset) * 2/h - 1,
                                (x + tw/2 + offset) * 2/w - 1, indexing="ij")
        return F.grid_sample(feature, torch.stack((xx, yy), -1)[None],
                             align_corners=False, padding_mode="zeros")

    def forward(self, local, image, tile, action, cache):
        device = local.device
        sources = SOURCES if action == "all" else (() if action in ("detail", "target_only") else (action,))
        embeddings = []
        for name in sources:
            if name == "scene":
                features = self.scene_region(image, tile, cache)
            else:
                side, resolution = VIEWS[name]
                view = resize(crop(image[None], tile, side), (resolution,) * 2)
                features = self.encoder(view.to(device))
            pooled = F.adaptive_avg_pool2d(features, (4, 4)).flatten(1)
            embeddings.append(self.projection(pooled) + self.scale.weight[SOURCES.index(name)])
        summary = local.new_zeros((1, 24))
        if embeddings:
            values = torch.stack(embeddings, dim=1)
            summary = (values * self.weight(values).softmax(1)).sum(1)
        if self.proposals is not None and action != "target_only":
            summary = summary + self.proposals(image, tile, sources, local, cache)
        local_grid = F.adaptive_avg_pool2d(local, (32, 32))
        residual = self.head(torch.cat((local_grid, summary[:, :, None, None].expand(-1,-1,32,32)), 1))
        return local + F.interpolate(residual, local.shape[-2:], mode="bilinear", align_corners=False)


def allocate(gains, costs, budget, allowed):
    """Multiple-choice knapsack; whole milliseconds rounded conservatively."""
    n, actions = gains.shape
    price = np.ceil(costs).astype(int)
    if len(price) != actions or price[0] != 0 or np.any(price[1:] <= 0):
        raise ValueError("Expected positive finite acquisition prices and free keep")
    limit = min(math.floor(budget), int(price.max()) * n)
    best = np.zeros(limit+1)
    choices = np.zeros((n, limit+1), dtype=np.int16)
    for i in range(n):
        if not allowed[i]:
            continue
        previous = best.copy()
        for action in range(1, actions):
            cost = price[action]
            if cost > limit:
                continue
            proposed = previous[:limit+1-cost] + gains[i, action]
            improve = proposed > best[cost:]
            best[cost:][improve] = proposed[improve]
            choices[i, cost:][improve] = action
    result = []
    for row in choices[::-1]:
        action = int(row[limit])
        result.append(action)
        limit -= price[action]
    return result[::-1]
