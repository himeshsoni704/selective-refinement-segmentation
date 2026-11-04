import math
import torch
from .objectives import utility


def choose_actions(pred, cfg, costs=None):
    """0=skip, 1=low, 2=high. Thresholds concern P(any action has positive utility)."""
    r = cfg["routing"]
    q = (pred["benefit_logits"][:, 2] / r["temperature"]).sigmoid()
    u = utility(pred["transitions"], pred["confusion"], cfg)
    actions = torch.zeros_like(q, dtype=torch.long)
    actions[q >= r["T2"]] = 1
    actions[q >= r["T1"]] = 2
    if r["utility_veto"]:
        # Never upgrade a low-band tile. A harmful high action can fall back to low.
        high_bad = (actions == 2) & (u[:, 1] <= 0)
        actions[high_bad] = torch.where(u[high_bad, 0] > 0, 1, 0)
        actions[(actions == 1) & (u[:, 0] <= 0)] = 0
    costs = costs if costs is not None else r["action_ms"]
    cost = torch.as_tensor(costs, device=q.device, dtype=u.dtype).clamp_min(1e-6)
    priority = u.gather(1, (actions-1).clamp_min(0)[:, None]).squeeze(1)
    priority = priority / cost[(actions-1).clamp_min(0)]
    cap = math.floor(r["max_fraction"] * len(actions))
    budget = r.get("max_refine_ms")
    selected = (actions > 0).nonzero().flatten()
    order = selected[priority[selected].argsort(descending=True)]
    retained, spent = 0, 0.
    for i in order.tolist():
        expense = float(cost[actions[i]-1])
        if retained >= cap or (budget is not None and spent + expense > budget):
            actions[i] = 0
        else:
            spent += expense
            retained += 1
    return actions, q, u
