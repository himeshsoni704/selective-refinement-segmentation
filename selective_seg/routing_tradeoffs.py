"""Offline observed-outcome tradeoffs; no deployable risk guarantees."""
import argparse
import json
from pathlib import Path

import torch

from .diagnostics import write
from .hybrid_evidence import tile_cap
from .hybrid_metrics import paired_bootstrap
from .specialist import counts_metrics


def allocate(row, fraction, damage_rate=None, penalty=0.):
    corrected, damaged = row['corrected_by_class'], row['damaged_by_class']
    scores = corrected.sum(-1).double() - penalty * damaged.sum(-1)
    costs = row['costs']
    selected = torch.zeros(len(scores), dtype=torch.long)
    available = row['supervised'][:, None].expand_as(scores).clone()
    # Same estimated compute allowance for every policy on this image.
    observed = costs[row['supervised']]
    budget = fraction * len(scores) * float(observed[:, -1].median()) if len(observed) else 0.
    cap = row['coarse_counts'].diag().double() * damage_rate if damage_rate is not None else None
    damage_used = torch.zeros(damaged.shape[-1], dtype=torch.double)
    spent = 0.
    for _ in range(tile_cap(fraction, len(scores))):
        feasible = available & (costs + spent <= budget + 1e-8)
        if cap is not None:
            feasible &= (damaged + damage_used <= cap + 1e-8).all(-1)
        rated = scores.masked_fill(~feasible, -float('inf'))
        index = int(rated.flatten().argmax())
        tile, action = divmod(index, scores.shape[1])
        if rated[tile, action] <= 0:
            break
        selected[tile] = action+1
        available[tile] = False
        spent += float(costs[tile, action])
        damage_used += damaged[tile, action]
    return selected, spent


def observed_tradeoffs(cache):
    results = []
    for budget in (.1, .2, .3):
        policies = [('gain', None, 0.)]
        policies += [('penalty', None, x) for x in (.5, 1., 2., 4., 8., 16.)]
        policies += [('class_caps', x, 0.) for x in (.001, .005, .01, .02, .05)]
        for policy, rate, penalty in policies:
            counts = torch.zeros_like(cache['rows'][0]['coarse_counts'])
            corrected = torch.zeros(len(counts), dtype=torch.long)
            damaged = corrected.clone()
            selected_count = 0
            cost = 0.
            for row in cache['rows']:
                actions, spent = allocate(row, budget, rate, penalty)
                cm = row['coarse_counts'].clone()
                for tile in (actions > 0).nonzero().flatten().tolist():
                    action = int(actions[tile])-1
                    cm += row['action_counts'][tile, action] - row['tile_coarse_counts'][tile]
                    corrected += row['corrected_by_class'][tile, action]
                    damaged += row['damaged_by_class'][tile, action]
                counts += cm
                selected_count += int((actions > 0).sum())
                cost += spent
            results.append({'policy': policy, 'tile_budget': budget, 'class_damage_rate_cap': rate,
                            'damage_penalty': penalty, **counts_metrics(counts),
                            'corrected_by_class': corrected.tolist(), 'damaged_by_class': damaged.tolist(),
                            'selected_tiles': selected_count, 'mean_estimated_action_ms': cost/len(cache['rows'])})
    return {'policies': results, 'scope': 'validation labels used for action selection; observed-outcome diagnostic only',
            'objective': 'maximize corrected pixel counts minus swept total-damage penalty, greedy',
            'constraints': 'per-image tile cap, estimated action-time cap, optional class damage / coarse-correct class pixels',
            'limitations': ['No learned class-damage predictor or held-out calibration in this experiment.',
                           'Greedy heuristic; no claim of an optimum or IoU maximization.',
                           'Estimated action costs exclude coarse/selector/specialist overhead.',
                           'Class caps are empirical on these images, not future-image risk guarantees.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    a, b = [torch.load(root/mode/'val/cache.pt', map_location='cpu', weights_only=True) for mode in ('none', 'full')]
    if a['paths'] != b['paths'] or len(a['rows']) != len(b['rows']):
        raise ValueError('Evidence comparison has different images')
    for first, second in zip(a['rows'], b['rows']):
        for key in ('supervised', 'coarse_counts', 'tile_coarse_counts', 'action_counts',
                    'corrected_by_class', 'damaged_by_class', 'gains'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        # Evidence appends features; all coarse descriptors must be identical.
        torch.testing.assert_close(first['z'], second['z'][:, :first['z'].shape[1]], rtol=0, atol=0)
    pairs = []
    for budget in (.05, .1, .2, .3, .5, 1.):
        reports = [json.loads((root/mode/f'evaluation/budget_{budget:g}/metrics.json').read_text()) for mode in ('none', 'full')]
        off, on = reports
        pairs.append({'tile_budget': budget, 'off_mIoU': off['mIoU'], 'on_mIoU': on['mIoU'],
                      'off_mean_ms': off['mean_image_latency_ms'], 'on_mean_ms': on['mean_image_latency_ms'],
                      'on_specialist_invocations': on['specialist_invocations'],
                      'paired': paired_bootstrap(on['images'], off['images'], 1000, 42)})
    write(root/'paired_comparison.json', {'pairs': pairs, 'fixed_candidates_and_outcomes_verified': True,
          'merge_rule': 'A/R conditional pair swap, original other-class winners protected',
          'scope': 'development validation, CPU timing is not GPU timing; both selectors fitted and selected independently',
          'candidate_fraction': .3, 'note': 'Budgets above .3 cannot exceed the fixed .3 candidate cap.'})
    write(root/'observed_class_damage_tradeoffs.json', observed_tradeoffs(a))


if __name__ == '__main__':
    main()
