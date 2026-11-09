"""Simple routing controls and label-based diagnostics, never used as inference inputs."""
import math
import torch


def baseline_actions(pc, grid, cfg, mode):
    n = len(grid)
    if mode == 'random':
        gen = torch.Generator().manual_seed(cfg['seed'])
        order = torch.randperm(n, generator=gen)
    else:
        ent = -(pc.clamp_min(1e-8).log()*pc).sum(1)[0]
        scores = torch.stack([ent[y:y+h,x:x+w].mean() for y,x,h,w in grid])
        order = scores.argsort(descending=True)
    action = cfg['routing'].get('baseline_action', 2)
    cost = cfg['routing']['action_ms'][action-1]
    budget = cfg['routing'].get('max_refine_ms')
    cap = math.floor(cfg['routing']['max_fraction']*n)
    if budget is not None:
        cap = min(cap, math.floor(budget/cost))
    result = torch.zeros(n, dtype=torch.long)
    result[order[:cap]] = action
    return result


def correction_diagnostics(coarse, final, truth, cfg, confidence=None):
    valid = truth != cfg['ignore_index']
    wrong = coarse != truth
    corrected = valid & wrong & (final == truth)
    damaged = valid & ~wrong & (final != truth)
    c = cfg['classes']
    def pairs(mask, source, destination):
        return torch.bincount(source[mask]*c+destination[mask], minlength=c*c).reshape(c,c).tolist()
    result = {'corrected_pixels':int(corrected.sum()), 'damaged_pixels':int(damaged.sum()),
              'valid_pixels':int(valid.sum()),
              'correction_pairs_predicted_to_true':pairs(corrected, coarse, truth),
              'damage_pairs_true_to_predicted':pairs(damaged, truth, final)}
    if confidence is not None:
        confident_wrong = valid & wrong & (confidence >= cfg.get('evaluation',{}).get('confidence_threshold',.9))
        result.update(confident_errors=int(confident_wrong.sum()),
                      recovered_confident_errors=int((confident_wrong & corrected).sum()))
    return result


def analyze_cache(cfg, cache):
    """Validation-only diagnostic, not a deployable policy or an exact upper bound."""
    import copy
    from .routing import choose_actions
    from .objectives import miou
    if cache['split'] != 'val':
        raise ValueError('Oracle-assisted diagnostics are validation-only')
    totals = [torch.zeros(cfg['classes'],cfg['classes'],dtype=torch.long) for _ in range(4)]
    oracle_cfg = copy.deepcopy(cfg)
    oracle_cfg['routing']['policy'] = 'direct'
    corrected = torch.zeros(2)
    damaged = torch.zeros(2)
    for im in cache['images']:
        for a in range(3):
            totals[a] += im['counts'][:,a].sum(0)
        # True labels are used explicitly here; never in deployed prediction.
        pred = {'confusion':im['confusion'],'transitions':im['transitions'],
                'benefit_logits':torch.zeros(len(im['z']),3)}
        actions,_,_ = choose_actions(pred,oracle_cfg,support=im['valid_pixels'])
        totals[3] += im['counts'][torch.arange(len(actions)),actions].sum(0)
        corrected += (im['transitions'][...,2]*im['valid_pixels'][:,None]).sum(0)
        damaged += (im['transitions'][...,1]*im['valid_pixels'][:,None]).sum(0)
    names = ['coarse','always_low','always_high','oracle_assisted_greedy']
    return {'validation_results':{name:{'mIoU':miou(cm),'confusion':cm.tolist()} for name,cm in zip(names,totals)},
            'low_high_corrected_pixels':corrected.round().long().tolist(),
            'low_high_damaged_pixels':damaged.round().long().tolist(),
            'note':'Uses validation ground truth and supplied estimated costs. Greedy heuristic, not an exact upper bound. Gate-enabled caches describe accepted outcomes.'}
