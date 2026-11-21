"""Consistent class-transition prediction and label-free budgeted allocation.

Axes of T are tile, action, truth, coarse prediction, refined prediction.
All counts used by the allocator are predictions, never evaluation labels.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .hybrid_evidence import tile_cap
from .objectives import tensor_miou


def transition_counts(truth, coarse, refined, classes, ignore=255):
    valid = (truth != ignore) & (truth >= 0) & (truth < classes)
    index = (truth[valid].long() * classes + coarse[valid].long()) * classes + refined[valid].long()
    return torch.bincount(index, minlength=classes ** 3).reshape(classes, classes, classes)


def correction_damage(joint):
    """Return counts by true class, including wrong-to-wrong transitions in T."""
    c = joint.shape[-1]
    y, before, after = torch.meshgrid(*(torch.arange(c, device=joint.device),) * 3, indexing='ij')
    corrected = (joint * ((y != before) & (y == after))).sum((-2, -1))
    damaged = (joint * ((y == before) & (y != after))).sum((-2, -1))
    return corrected, damaged


class TransitionSelector(nn.Module):
    def __init__(self, dimension, classes=6, actions=2, hidden=96, independent=False, pair=(1, 2)):
        super().__init__()
        self.classes, self.actions, self.independent = classes, actions, independent
        self.encoder = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU())
        self.before = nn.Linear(hidden, classes * classes)
        # Infer valid-label mass from image features. Do not use the GT ignore mask at inference.
        self.valid = nn.Linear(hidden, classes)
        self.transition = nn.Linear(hidden, actions * classes ** 3)
        self.action_before = nn.Linear(hidden, actions * classes ** 2) if independent else None
        allowed = torch.eye(classes, dtype=torch.bool)
        for left in pair:
            for right in pair:
                allowed[left, right] = True
        self.register_buffer('allowed', allowed)

    def forward(self, z, coarse_histogram):
        h = self.encoder(z)
        n, c, a = len(z), self.classes, self.actions
        log_y = self.before(h).reshape(n, c, c).log_softmax(1)
        valid_logits = self.valid(h)
        mass = coarse_histogram.to(h) * valid_logits.sigmoid()
        before = log_y.exp() * mass[:, None, :]
        logits = self.transition(h).reshape(n, a, c, c, c)
        # Exactly match the frozen A/R-only merge: other coarse winners cannot change.
        log_r = logits.masked_fill(~self.allowed, -1e9).log_softmax(-1)
        if self.independent:
            action_log_y = self.action_before(h).reshape(n, a, c, c).log_softmax(2)
            action_before = action_log_y.exp() * mass[:, None, None, :]
        else:
            action_log_y = None
            action_before = before[:, None]
        joint = action_before[..., None] * log_r.exp()
        return {'before': before, 'joint': joint, 'log_y': log_y, 'log_r': log_r,
                'valid_logits': valid_logits, 'action_log_y': action_log_y}


def transition_loss(prediction, row):
    before = row['tile_coarse_counts'].float()
    histogram = row['histogram'].float()
    valid_by_c = before.sum(1)
    validity = F.binary_cross_entropy_with_logits(prediction['valid_logits'],
                valid_by_c / histogram.clamp_min(1), reduction='none')
    validity = (validity * histogram).sum() / histogram.sum().clamp_min(1)
    coarse_loss = -(before * prediction['log_y']).sum() / before.sum().clamp_min(1)
    joint = row['joint'].float()
    transition = -(joint * prediction['log_r']).sum() / joint.sum().clamp_min(1)
    independent = prediction['action_log_y']
    if independent is not None:
        transition = transition - (joint.sum(-1) * independent).sum() / joint.sum().clamp_min(1)
    return validity + coarse_loss + transition


def allocate(before, joint, allowed, costs, fraction, max_ms=None, damage_fraction=None,
             damage_penalty=0., recompute=True):
    """Greedy predicted image-mIoU gain with tile/time/predicted class-damage caps.

    Caps constrain expected counts, not actual risk. This is not an optimal solver
    or a statistical guarantee. No truth or observed action outcomes are accepted.
    """
    if not 0 <= fraction <= 1 or not math.isfinite(damage_penalty) or damage_penalty < 0:
        raise ValueError('Invalid tile fraction or damage penalty')
    if damage_fraction is not None and not 0 <= damage_fraction <= 1:
        raise ValueError('Damage fraction must be in [0,1]')
    if max_ms is not None and (not math.isfinite(max_ms) or max_ms < 0):
        raise ValueError('Time budget must be finite and nonnegative')
    before, joint = before.double(), joint.double()
    costs = torch.as_tensor(costs, dtype=torch.double).expand(joint.shape[:2])
    if not torch.isfinite(before).all() or not torch.isfinite(joint).all() or (before < 0).any() or (joint < 0).any():
        raise ValueError('Expected finite nonnegative counts')
    if not torch.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('Costs must be finite and positive')
    after = joint.sum(-2)
    delta = after - before[:, None]
    _, damage = correction_damage(joint)
    base = before.sum(0)
    current = base.clone()
    scale = base.sum().clamp_min(1)
    cap = base.diagonal() * damage_fraction if damage_fraction is not None else None
    used_damage = torch.zeros_like(base.diagonal())
    permitted = allowed.bool().reshape(-1, 1).expand(joint.shape[:2]).clone()
    selected = torch.zeros(len(before), dtype=torch.long)
    spent = 0.
    for _ in range(tile_cap(fraction, len(before))):
        reference = current if recompute else base
        gains = tensor_miou((reference[None, None] + delta).clamp_min(0)) - tensor_miou(reference)
        gains -= damage_penalty * damage.sum(-1) / scale
        eligible = permitted.clone()
        if max_ms is not None:
            eligible &= costs + spent <= max_ms + 1e-9
        if cap is not None:
            eligible &= (used_damage + damage <= cap + 1e-7).all(-1)
        gains = gains.masked_fill(~eligible, -float('inf'))
        flat = int(gains.argmax())
        i, a = divmod(flat, joint.shape[1])
        if gains[i, a] <= 0:
            break
        selected[i] = a + 1
        permitted[i] = False
        current += delta[i, a]
        used_damage += damage[i, a]
        spent += float(costs[i, a])
    return selected, {'predicted_damage': used_damage.tolist(), 'estimated_action_ms': spent,
                      'predicted_before_mIoU': float(tensor_miou(base)),
                      'predicted_after_mIoU': float(tensor_miou(current)),
                      'damage_fraction': damage_fraction,
                      'predicted_damage_cap': cap.tolist() if cap is not None else None}


def scalar_allocate(gains, allowed, costs, fraction, max_ms=None):
    """Existing gain-regression control with the same tile and estimated time caps."""
    costs = torch.as_tensor(costs).expand_as(gains)
    available = allowed[:, None].expand_as(gains).clone()
    chosen = torch.zeros(len(gains), dtype=torch.long)
    spent = 0.
    for _ in range(tile_cap(fraction, len(gains))):
        eligible = available if max_ms is None else available & (costs + spent <= max_ms + 1e-9)
        rated = gains.masked_fill(~eligible, -float('inf'))
        flat = int(rated.argmax())
        i, a = divmod(flat, gains.shape[1])
        if rated[i, a] <= 0:
            break
        chosen[i] = a + 1
        available[i] = False
        spent += float(costs[i, a])
    return chosen, {'estimated_action_ms': spent}
