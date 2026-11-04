import math
import torch
from torch.nn import functional as F


def resize_long(x, long_side):
    h, w = x.shape[-2:]
    size = (max(32, round(h * long_side / max(h, w))),
            max(32, round(w * long_side / max(h, w))))
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def tiles(h, w, size):
    return [(y, x, min(size, h-y), min(size, w-x))
            for y in range(0, h, size) for x in range(0, w, size)]


def crop_context(x, tile, size, border, fill=None):
    """Crop fixed field of view; replicate images/probs, ignore-pad labels."""
    y, z, _, _ = tile
    h, w = x.shape[-2:]
    y0, x0 = y-border, z-border
    y1, x1 = y+size+border, z+size+border
    cropped = x[..., max(0,y0):min(h,y1), max(0,x0):min(w,x1)]
    pad = (max(0,-x0), max(0,x1-w), max(0,-y0), max(0,y1-h))
    if fill is None:
        padded = F.pad(cropped, pad, mode="replicate")
    else:
        padded = F.pad(cropped, pad, mode="constant", value=fill)
    return padded


def center_prediction(logits, tile, cfg):
    total = cfg["tile_size"] + 2 * cfg["border"]
    p = F.interpolate(logits, (total, total), mode="bilinear", align_corners=False)
    b = cfg["border"]
    return p[..., b:b+tile[2], b:b+tile[3]]


def feature_region(features, tile, image_hw):
    # Pool on native coarse features; never expand 256 channels to the UHR grid.
    y, x, h, w = tile
    ih, iw = image_hw
    fh, fw = features.shape[-2:]
    y0, x0 = int(y * fh / ih), int(x * fw / iw)
    y1, x1 = min(fh, math.ceil((y+h)*fh/ih)), min(fw, math.ceil((x+w)*fw/iw))
    return features[..., y0:max(y0+1,y1), x0:max(x0+1,x1)].mean((-2,-1))


def tile_descriptor(image, probabilities, features, tile, cfg):
    y, x, h, w = tile
    pc = probabilities[..., y:y+h, x:x+w].float()
    entropy = -(pc.clamp_min(1e-8) * pc.clamp_min(1e-8).log()).sum(1) / math.log(pc.shape[1])
    top = pc.topk(2, dim=1).values
    margin = top[:, 0] - top[:, 1]
    labels = pc.argmax(1)
    edges = torch.zeros_like(labels, dtype=torch.bool)
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edges[:, :, 1:] |= labels[:, :, 1:] != labels[:, :, :-1]
    coarse_stats = torch.stack((entropy.mean((-2,-1)), entropy.amax((-2,-1)),
                               margin.mean((-2,-1)), (margin < .2).float().mean((-2,-1)),
                               edges.float().mean((-2,-1))), -1)
    # Cheap original-resolution scan. All spectral bands remain available here.
    patch = image[..., y:y+h, x:x+w].float()
    ratio = min(1., cfg["coarse_size"] / max(image.shape[-2:]))
    small = F.interpolate(patch, (max(1,round(h*ratio)), max(1,round(w*ratio))), mode="area")
    smooth = F.interpolate(small, (h,w), mode="bilinear", align_corners=False)
    residual = (patch-smooth).abs().mean(1)
    edge_x = (patch[..., 1:] - patch[..., :-1]).abs().mean() if w > 1 else patch.new_zeros(())
    edge_y = (patch[..., 1:, :] - patch[..., :-1, :]).abs().mean() if h > 1 else patch.new_zeros(())
    detail = torch.stack((residual.mean(), residual.max(), edge_x, edge_y))[None]
    pooled = feature_region(features.float(), tile, image.shape[-2:])
    return torch.cat((pooled, pc.mean((-2,-1)), coarse_stats, detail), -1)
