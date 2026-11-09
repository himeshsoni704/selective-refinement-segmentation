import torch
from torch.nn import functional as F


def confusion_matrix(pred, truth, classes, ignore=255):
    valid = (truth != ignore) & (truth >= 0) & (truth < classes)
    index = truth[valid].long() * classes + pred[valid].long()
    return torch.bincount(index, minlength=classes ** 2).reshape(classes, classes)


def outcome_targets(coarse, low, high, truth, cfg):
    valid = truth != cfg["ignore_index"]
    n = valid.sum()
    if n == 0:
        return None
    y, c = truth[valid], coarse[valid]
    transition = []
    for pred in (low[valid], high[valid]):
        # Encoding: CC=0, CW=1, WC=2, WW=3.
        state = (c != y).long() * 2 + (pred != y).long()
        transition.append(torch.bincount(state, minlength=4).float() / n)
    counts = torch.stack([confusion_matrix(p, truth, cfg["classes"], cfg["ignore_index"])
                          for p in (coarse, low, high)])
    return {"transitions": torch.stack(transition), "confusion": counts.float() / n,
            "counts": counts, "valid_pixels": n}


def utility(transitions, confusion, cfg, support=None):
    """Damage-penalized benefit plus weighted TP/FP/FN changes; not an mIoU estimate."""
    mode = cfg['utility'].get('mode', 'legacy')
    if mode == 'miou':
        if support is None:
            raise ValueError('mIoU utility requires image-grouped support counts')
        return metric_gains(confusion, support)
    tp = confusion.diagonal(dim1=-2, dim2=-1)
    fp, fn = confusion.sum(-2) - tp, confusion.sum(-1) - tp
    w = torch.as_tensor(cfg["utility"]["class_weights"], device=tp.device, dtype=tp.dtype)
    w = w / w.sum()
    u = cfg["utility"]
    delta = (u["tp_reward"] * (tp[..., 1:, :] - tp[..., :1, :])
             + u["fp_penalty"] * (fp[..., :1, :] - fp[..., 1:, :])
             + u["fn_penalty"] * (fn[..., :1, :] - fn[..., 1:, :]))
    class_benefit = (delta * w).sum(-1)
    return (transitions[..., 2] - u["damage_penalty"] * transitions[..., 1]
            + u["confusion_weight"] * class_benefit)


def selector_loss(pred, target, cfg):
    trans = pred["transitions"].clamp_min(1e-8)
    cm = pred["confusion"].clamp_min(1e-8)
    t_loss = -(target["transitions"] * trans.log()).sum(-1).mean()
    c_loss = -(target["confusion"] * cm.log()).sum((-2, -1)).mean()
    true_u = utility(target["transitions"], target["confusion"], cfg, target.get("valid_pixels"))
    labels = torch.cat((true_u > 0, (true_u.max(-1).values > 0).unsqueeze(-1)), -1).float()
    b_loss = F.binary_cross_entropy_with_logits(pred["benefit_logits"], labels)
    predicted_u = utility(trans, cm, cfg, target.get("valid_pixels"))
    u_loss = F.huber_loss(predicted_u, true_u, delta=.05)
    # Make separately predicted transition and class-confusion marginals agree.
    error = 1 - cm.diagonal(dim1=-2, dim2=-1).sum(-1)
    consistency = (F.mse_loss(trans[..., 2] + trans[..., 3], error[:, :1].expand(-1, 2))
                   + F.mse_loss(trans[..., 1] + trans[..., 3], error[:, 1:]))
    loss = t_loss + c_loss + b_loss + 5 * u_loss + .2 * consistency
    return loss


def segmentation_loss(output, target, cfg):
    valid = target != cfg["ignore_index"]
    if not valid.any():
        return output["logits"].sum() * 0
    weights = cfg['training'].get('class_weights')
    weights = output['logits'].new_tensor(weights) if weights is not None else None
    ce = F.cross_entropy(output["logits"], target, weight=weights, ignore_index=cfg["ignore_index"])
    iou_weight = cfg['training'].get('soft_iou_weight', 0.)
    if iou_weight:
        probabilities = output['logits'].softmax(1)
        safe = target.masked_fill(~valid, 0)
        onehot = F.one_hot(safe, cfg['classes']).permute(0,3,1,2).to(probabilities)
        probabilities = probabilities * valid[:,None]
        onehot = onehot * valid[:,None]
        intersection = (probabilities*onehot).sum((0,2,3))
        union = (probabilities+onehot-probabilities*onehot).sum((0,2,3))
        ce = ce + iou_weight * (1-(intersection/(union+1e-8)).mean())
    class_loss = ce.new_zeros(())
    class_prob = output.get("spectral_class_prob")
    if class_prob is not None:
        histograms = []
        for y in target:
            pixels = y[y != cfg["ignore_index"]]
            hist = torch.bincount(pixels, minlength=cfg["classes"]).float()
            histograms.append(hist / hist.sum().clamp_min(1))
        class_loss = -(torch.stack(histograms) * class_prob.clamp_min(1e-8).log()).sum(-1).mean()
    return (ce + cfg["spectral"]["diversity_weight"] * output["spectral_reg"]
            + cfg["spectral"]["class_weight"] * class_loss)


def miou(counts):
    cm = counts.double()
    tp = cm.diagonal()
    union = cm.sum(0) + cm.sum(1) - tp
    valid = union > 0
    return (tp[valid] / union[valid]).mean().item() if valid.any() else 0.


def tensor_miou(counts):
    """Differentiable macro-IoU, excluding classes with zero union."""
    tp = counts.diagonal(dim1=-2, dim2=-1)
    union = counts.sum(-1) + counts.sum(-2) - tp
    present = union > 0
    return ((tp / union.clamp_min(1e-8))*present).sum(-1) / present.sum(-1).clamp_min(1)


def metric_gains(confusion, support, aggregate=None, current=None):
    """Image-context single-tile marginal gains. Rows=true, cols=predicted.

    Training support is valid_pixels. Deployment uses geometric tile area, never
    test masks; the latter assumes ignored regions are representative/missing at random.
    IoU of expected counts is a plug-in surrogate, not expected dataset IoU.
    """
    counts = confusion * support.to(confusion)[:, None, None, None]
    if current is None:
        current = torch.zeros(len(counts), device=counts.device, dtype=torch.long)
    chosen = counts[torch.arange(len(counts), device=counts.device), current]
    aggregate = chosen.sum(0) if aggregate is None else aggregate
    after = (aggregate[None,None] - chosen[:,None] + counts[:,1:]).clamp_min(0)
    return tensor_miou(after) - tensor_miou(aggregate)


def acceptance_loss(logits, coarse, refined, truth, ignore=255, class_weights=None):
    valid = truth != ignore
    c, r = coarse.argmax(1), refined.argmax(1)
    # Only one-correct disagreements identify a preference; both-wrong and ties excluded.
    decisive = valid & ((c == truth) ^ (r == truth))
    if not decisive.any():
        return logits.sum()*0
    target = (r == truth).float()
    loss = F.binary_cross_entropy_with_logits(logits[decisive], target[decisive], reduction='none')
    if class_weights is not None:
        weights = logits.new_tensor(class_weights)[truth[decisive]]
        return (loss*weights).sum()/weights.sum().clamp_min(1e-8)
    return loss.mean()
