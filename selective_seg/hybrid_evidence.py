"""Prediction-only candidate selection and class-aligned specialist evidence."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import center_prediction, crop_context
from .engine import amp
from .hybrid_backbone import align
from .specialist import settings

EPS = 1e-7


def pair_ids(cfg):
    spec = settings(cfg)
    return [spec["agriculture"], spec["rangeland"]]


def conditional(probabilities, pair):
    selected = probabilities[:, pair]
    mass = selected.sum(1, keepdim=True)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    return torch.where(mass > 0, selected / safe_mass, .5)


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(EPS).log()).sum(1) / math.log(probabilities.shape[1])


def edges(labels):
    x = F.pad((labels[..., 1:] != labels[..., :-1]).float(), (0, 1))
    y = F.pad((labels[..., 1:, :] != labels[..., :-1, :]).float(), (0, 0, 0, 1))
    return torch.maximum(x, y)


def area(tensor, tile):
    y, x, h, w = tile
    return tensor[..., y:y+h, x:x+w]


def seam_statistics(probabilities, grid, dimension):
    """Four directional seams, each with a presence bit and an aligned consistency score."""
    rows, columns = dimension
    result = probabilities.new_zeros(len(grid), 8)
    for index, tile in enumerate(grid):
        row, column = divmod(index, columns)
        here = area(probabilities, tile)
        neighbours = (
            (row > 0, index - columns, here[:, :, 0, :], lambda x: x[:, :, -1, :]),
            (row + 1 < rows, index + columns, here[:, :, -1, :], lambda x: x[:, :, 0, :]),
            (column > 0, index - 1, here[:, :, :, 0], lambda x: x[:, :, :, -1]),
            (column + 1 < columns, index + 1, here[:, :, :, -1], lambda x: x[:, :, :, 0]),
        )
        for direction, (present, other, side, adjoining) in enumerate(neighbours):
            if present:
                opposite = adjoining(area(probabilities, grid[other]))
                result[index, 2*direction] = 1
                result[index, 2*direction+1] = 1 - .5 * (side-opposite).abs().sum(1).mean()
    return result


def tile_cap(fraction, total):
    # Floor except that a positive budget can purchase at least one whole tile.
    return min(total, max(int(fraction > 0), math.floor(fraction * total + 1e-10)))


def candidates(image, pc, grid, cfg):
    pair = pair_ids(cfg)
    mass = pc[:, pair].sum(1)
    cond = conditional(pc, pair)
    ambiguity = mass * (1 - (cond[:, 0] - cond[:, 1]).abs())
    uncertainty = entropy(pc)
    boundary = edges(pc.argmax(1))
    confidence = pc.max(1).values
    gray = image.mean(0)[None, None]
    blurred = F.avg_pool2d(gray, 5, stride=1, padding=2, count_include_pad=False)
    texture = (gray - blurred).abs()[0, 0]
    eligible, scores = [], []
    for tile in grid:
        m = area(mass, tile)
        a = area(ambiguity, tile)
        u = area(uncertainty, tile)
        b = area(boundary, tile)
        t = (8 * area(texture, tile)).clamp_max(1)
        weak = 1 - area(confidence, tile)
        score = (.30*a + .20*u*m + .15*b*m + .15*t*m + .20*weak*m)
        scores.append(float(.65 * score.mean() + .35 * score.amax()))
        eligible.append(float(m.mean()) >= cfg["hybrid"]["min_pair_mass"])
    count = tile_cap(cfg["hybrid"]["candidate_fraction"], len(grid))
    ranked = sorted((i for i, valid in enumerate(eligible) if valid), key=lambda i: (-scores[i], i))
    return ranked[:count], torch.tensor(scores)


def crop_input(image, tile, cfg, device):
    crop = crop_context(image[None], tile, cfg["tile_size"], cfg["border"])
    size = settings(cfg)["input_size"]
    return F.interpolate(crop, (size, size), mode="bilinear", align_corners=False,
                         antialias=True).to(device)


def exit_score(probability, supported, gain, policy):
    confidence = probability.max(1).values
    if not supported.any():
        return float("-inf")
    if policy == "benefit":
        return -float(gain)
    return float(torch.quantile(confidence[supported], .10))


def specialist_maps(model, image, tile, cfg, device, supported, calibration=None, policy="full"):
    crop = crop_input(image, tile, cfg, device)
    calibration = calibration or {"temperatures": [1., 1., 1.]}
    with amp(device):
        for index, (logits, gain, disagreement) in enumerate(model.packets(crop, policy != "full")):
            depth = 2 if gain is None else index
            logits = center_prediction(logits.float(), tile, cfg)
            probability = (logits / calibration["temperatures"][depth]).softmax(1)
            fixed_depth = {"shallow": 0, "intermediate": 1}.get(policy)
            should_exit = depth == 2 or fixed_depth == depth
            if not should_exit and fixed_depth is None:
                threshold = calibration["thresholds"][policy][depth]
                should_exit = exit_score(probability, supported.to(device), gain, policy) >= threshold
            if should_exit:
                if disagreement is not None:
                    disagreement = center_prediction(disagreement.float(), tile, cfg).cpu()[0, 0]
                return probability.cpu(), depth, disagreement
    raise RuntimeError("Specialist produced no prediction")


def evidence(coarse, probability, supported, feature_gap, depth, mode, cfg):
    pair = pair_ids(cfg)
    mapped = torch.tensor(pair)[probability.argmax(1)]
    coarse_prediction = coarse.argmax(1)
    conf = probability.max(1).values
    uncertainty = entropy(probability)
    disagreement = (mapped != coarse_prediction) & supported
    boundary = edges(mapped)
    conditional_coarse = conditional(coarse, pair)
    gap = (conditional_coarse - probability).abs().mean(1)

    def mean(x):
        return x[supported].mean() if supported.any() else x.new_zeros(())

    basic = [conf.new_ones(()), supported.float().mean(), mean(probability[:, 1])]
    extra = [mean(conf), mean(uncertainty), mean(gap), mean(disagreement.float()),
             mean(boundary), mean(boundary * uncertainty),
             mean(conditional_coarse[:, 0] * probability[:, 1]),
             mean(conditional_coarse[:, 1] * probability[:, 0]),
             mean(uncertainty + disagreement.float()) / 2,
             (feature_gap[supported[0]].mean() if feature_gap is not None and supported.any()
              else conf.new_zeros(())), conf.new_tensor(float(feature_gap is not None)),
             conf.new_tensor(depth / 2)]
    return torch.stack(basic if mode == "predictions" else basic + extra), {
        "uncertainty": uncertainty[0].masked_fill(~supported[0], 0),
        "disagreement": disagreement[0], "support": supported[0],
        "boundary_uncertainty": (boundary*uncertainty)[0].masked_fill(~supported[0], 0),
        "confidence": conf[0].masked_fill(~supported[0], 0),
        "rangeland_probability": probability[0, 1].masked_fill(~supported[0], 0),
    }


def pair_update(coarse, pair_probability, pair, support):
    """Only allow an A↔R change; keep all other probabilities and pair mass.

    If a conditional flip would let a third class overtake the pair, preserve
    the original distribution at that pixel.
    """
    updated = coarse.clone()
    updated[:, pair] = pair_probability * coarse[:, pair].sum(1, keepdim=True)
    # A third-class original winner is protected regardless of support.
    original = coarse.argmax(1)
    proposed = updated.argmax(1)
    ids = original.new_tensor(pair)
    allowed = support & torch.isin(original, ids) & torch.isin(proposed, ids)
    return torch.where(allowed[:, None], updated, coarse)


class PatchSelector(nn.Module):
    def __init__(self, dimension, neighbourhood, actions):
        super().__init__()
        self.kind = neighbourhood
        self.encode = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 96), nn.GELU())
        if neighbourhood in {"learned", "conv"}:
            self.neighbours = (nn.Conv2d(96, 32, 3, padding=1) if neighbourhood == "conv"
                               else nn.Sequential(nn.Conv2d(96 * 9, 32, 1), nn.GELU()))
        extra = {"none": 0, "mean": 96, "learned": 32, "conv": 32}[neighbourhood]
        self.gain = nn.Sequential(nn.Linear(96 + extra, 64), nn.GELU(), nn.Linear(64, actions))

    def forward(self, descriptor, shape):
        encoded = self.encode(descriptor)
        if self.kind == "none":
            return self.gain(encoded)
        grid = encoded.T.reshape(1, 96, *shape)
        if self.kind == "mean":
            summed = F.avg_pool2d(grid, 3, stride=1, padding=1, count_include_pad=False)
            # Exclude the central tile, including on corners and 1x1 grids.
            n = F.conv2d(torch.ones_like(grid[:, :1]), grid.new_ones(1, 1, 3, 3), padding=1)
            context = (summed * n - grid) / (n - 1).clamp_min(1)
        elif self.kind == "learned":
            context = self.neighbours(F.unfold(grid, 3, padding=1).reshape(1, 96*9, *shape))
        else:
            context = self.neighbours(grid)
        return self.gain(torch.cat([encoded, context.flatten(2)[0].T], 1))
