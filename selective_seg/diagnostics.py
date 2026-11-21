"""Checkpoint diagnostics and validation-only cached routing controls.

No labels from these controls may be used by a deployed selector. Scores are
pooled pixel confusion metrics; cached costs are estimates, never latency claims.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .data import read_manifest, read_sample
from .engine import config_signature, fingerprint, sync
from .geometry import center_prediction, tiles
from .hybrid_evidence import crop_input
from .hybrid_metrics import Metrics
from .objectives import tensor_miou, utility
from .precision import specialist_amp, specialist_precision
from .specialist import MultiScaleSpecialist, binary_target, counts_metrics, signature


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


@torch.inference_mode()
def specialist_report(checkpoint_path, manifest, output, device, precision='fp32', limit=None):
    from .hybrid_specialist import load
    path = Path(checkpoint_path)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg = copy.deepcopy(ck['config'])
    cfg.setdefault('hybrid', {})['precision'] = precision
    bad = {k: int((~torch.isfinite(v)).sum()) for k, v in ck['model'].items()
           if v.is_floating_point() and not torch.isfinite(v).all()}
    if bad:
        write(output, {'status': 'invalid_checkpoint', 'nonfinite_state': bad})
        raise ValueError(f'Nonfinite checkpoint state: {bad}')
    if ck['stage'] == 'hybrid_specialist':
        model, _ = load(cfg, path, device)
    elif ck['stage'] == 'specialist' and ck['signature'] == signature(cfg):
        model = MultiScaleSpecialist(cfg).to(device).eval().requires_grad_(False)
        model.load_state_dict(ck['model'])
    else:
        raise ValueError('Unsupported or incompatible specialist checkpoint')
    records = [r for r in read_manifest(manifest) if r['split'] == 'val']
    if {r['image'] for r in records} & set(ck['train_paths']):
        raise ValueError('Validation overlaps specialist training')
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError('Empty validation selection')
    metrics = Metrics(2)
    numerical = {'tiles_checked': 0, 'logits_checked': 0, 'probabilities_checked': 0,
                 'nonfinite_logits': 0, 'nonfinite_probabilities': 0, 'bad_tiles': []}
    rows = []
    for record in records:
        image, truth = read_sample(record, cfg)
        image_cm = torch.zeros(2, 2, dtype=torch.long)
        elapsed = 0.
        for tile in tiles(*truth.shape, cfg['tile_size']):
            sync(device)
            start = time.perf_counter()
            with specialist_amp(device, cfg):
                raw = model(crop_input(image, tile, cfg, device))
                logits = center_prediction(raw.float(), tile, cfg)
            probabilities = logits.softmax(1)[0].cpu()
            sync(device)
            elapsed += 1000 * (time.perf_counter() - start)
            # Check raw output too: ignored pixels and cropped margins must not hide NaNs.
            nl = int((~torch.isfinite(raw)).sum()) + int((~torch.isfinite(logits)).sum())
            np = int((~torch.isfinite(probabilities)).sum())
            numerical['tiles_checked'] += 1
            numerical['logits_checked'] += raw.numel() + logits.numel()
            numerical['probabilities_checked'] += probabilities.numel()
            numerical['nonfinite_logits'] += nl
            numerical['nonfinite_probabilities'] += np
            if nl or np:
                numerical['bad_tiles'].append({'id': record['id'], 'tile': list(tile)})
                continue
            image_cm += torch.tensor(metrics.add(probabilities, binary_target(truth[
                tile[0]:tile[0]+tile[2], tile[1]:tile[1]+tile[3]], cfg)))
        rows.append({'id': record['id'], 'confusion': image_cm.tolist(), 'specialist_ms': elapsed})
        report = {**metrics.report(), 'status': 'running', 'numerical': numerical,
                  'completed_images': len(rows), 'total_images': len(records)}
        write(str(output) + '.progress.json', report)
        print(json.dumps({'checkpoint': str(path), 'image': record['id'],
                          'completed': len(rows), 'total': len(records), 'specialist_ms': elapsed}), flush=True)
    invalid = bool(numerical['bad_tiles'])
    report = {**metrics.report(), 'status': 'invalid_numerics' if invalid else 'complete',
              'metrics_valid': not invalid, 'numerical': numerical, 'images': rows,
              'split': 'val', 'subset': limit is not None, 'image_ids': [r['id'] for r in records],
              'class_names': ['agriculture', 'rangeland'], 'precision': specialist_precision(device, cfg),
              'device': str(device), 'checkpoint': str(path.resolve()), 'step': ck['step'],
              'runtime': {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                          'torch_threads': torch.get_num_threads(),
                          'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
              'checkpoint_sha256': fingerprint(path), 'manifest_sha256': fingerprint(manifest),
              'config': cfg, 'feature_spec': model.feature_spec,
              'timing_scope': 'single-pass crop/resize/forward/transfer, excludes decode and metrics; no GPU latency claim on CPU',
              'numerical_count_scope': 'raw and centered logits counted separately; probabilities on centered output',
              'NLL_scope': 'true A/R pixels, probability floor 1e-12',
              'selection_note': 'development validation previously used for checkpoint selection'}
    if invalid:
        report['partial_metrics_warning'] = 'Invalid tiles excluded; accuracy metrics are not a valid full-validation result.'
    write(output, report)
    return report


def greedy_miou(counts, cap, costs=None, max_ms=None):
    """Recompute image-level marginal IoU after every chosen nonoverlapping tile.

    This is a label-assisted greedy heuristic, NOT an exact oracle upper bound.
    """
    selected = torch.zeros(len(counts), dtype=torch.long)
    aggregate = counts[:, 0].sum(0).double()
    spent = 0.
    for _ in range(cap):
        after = aggregate[None, None] - counts[:, :1] + counts[:, 1:]
        gains = tensor_miou(after) - tensor_miou(aggregate)
        gains[selected != 0] = -float('inf')
        if max_ms is not None:
            gains = gains.masked_fill(costs + spent > max_ms + 1e-8, -float('inf'))
        index = int(gains.flatten().argmax())
        tile, action = divmod(index, 2)
        if gains[tile, action] <= 0:
            break
        selected[tile] = action + 1
        aggregate += counts[tile, action + 1] - counts[tile, 0]
        if costs is not None:
            spent += float(costs[tile, action])
    return selected


def cache_controls(cache_path, base_run, output, budgets=(.05, .1, .2, .3, .5, 1.)):
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if cache['split'] != 'val':
        raise ValueError('Label-assisted controls require validation cache')
    ck = torch.load(Path(base_run)/'coarse.pt', map_location='cpu', weights_only=True)
    cfg = ck['config']
    if cache['signature'] != config_signature(cfg):
        raise ValueError('Cache config mismatch')
    hashes = {stage: fingerprint(Path(base_run)/f'{stage}.pt') for stage in ('coarse', 'low', 'high')}
    if cache['provenance'] != hashes:
        raise ValueError('Cache does not match the current frozen segmenters')
    # Replay the existing calibrated selector on exactly the same cached outcomes.
    from .cli import apply_calibration
    from .models import OutcomeSelector
    from .research import settings
    from .routing import choose_actions
    selector_ck = torch.load(Path(base_run)/'selector.pt', map_location='cpu', weights_only=True)
    if selector_ck['provenance'] != hashes or selector_ck['signature'] != cache['signature']:
        raise ValueError('Existing selector does not match cached outcomes')
    cfg = apply_calibration(copy.deepcopy(cfg), Path(base_run)/'calibration.json', Path(base_run))
    selector = OutcomeSelector(cache['images'][0]['z'].shape[-1], cfg['classes'], settings(cfg)['shared_marginals']).eval()
    selector.load_state_dict(selector_ck['model'])
    with torch.inference_mode():
        predictions = {row['id']: selector(row['z']) for row in cache['images']}
    modes = [('coarse', 0., None), ('always_low', 1., None), ('always_high', 1., None)]
    modes += [(mode, b, None) for b in budgets for mode in ('current_selector', 'random_high', 'label_assisted_miou', 'label_assisted_scalar')]
    reports = []
    for mode, fraction, max_ms in modes:
        pooled = torch.zeros(cfg['classes'], cfg['classes'], dtype=torch.long)
        corrected = damaged = chosen = total = 0
        estimated_cost = 0.
        images = []
        for row in cache['images']:
            counts, costs = row['counts'], row['cost_ms']
            n = len(counts)
            cap = int(fraction*n + 1e-10)
            actions = torch.zeros(n, dtype=torch.long)
            if mode in {'always_low', 'always_high'}:
                actions[:] = 1 if mode == 'always_low' else 2
            elif mode == 'random_high':
                import hashlib
                seed = cfg['seed'] + int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
                actions[torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:cap]] = 2
            elif mode == 'label_assisted_miou':
                actions = greedy_miou(counts, cap)
            elif mode == 'current_selector':
                candidate_cfg = copy.deepcopy(cfg)
                candidate_cfg['routing']['max_fraction'] = fraction
                actions = choose_actions(predictions[row['id']], candidate_cfg, support=row['valid_pixels'])[0]
            elif mode == 'label_assisted_scalar':
                scores = utility(row['transitions'], row['confusion'], cfg, row['valid_pixels'])
                value, action = scores.max(1)
                order = value.argsort(descending=True, stable=True)[:cap]
                order = order[value[order] > 0]
                actions[order] = action[order]+1
            image_cm = counts[torch.arange(n), actions].sum(0)
            pooled += image_cm
            active = actions > 0
            idx = torch.arange(n)[active]
            act = actions[active]-1
            corrected += int((row['transitions'][idx, act, 2]*row['valid_pixels'][active]).sum().round())
            damaged += int((row['transitions'][idx, act, 1]*row['valid_pixels'][active]).sum().round())
            estimated_cost += float(costs[idx, act].sum())
            chosen += int(active.sum())
            total += n
            images.append({'id': row['id'], 'confusion': image_cm.tolist(), 'selected_tiles': int(active.sum())})
        reports.append({'mode': mode, 'tile_budget': fraction, **counts_metrics(pooled),
                        'corrected_pixels': corrected, 'damaged_pixels': damaged,
                        'selected_tiles': chosen, 'total_tiles': total,
                        'mean_cached_estimated_refinement_ms': estimated_cost/len(images), 'images': images})
    result = {'split': 'val', 'checkpoint_sha256': hashes, 'cache_sha256': fingerprint(cache_path),
              'policies': reports,
              'limitations': ['Cached costs are historical estimates, not new end-to-end timings.',
                             'Label-assisted policies use validation truth and cannot be deployed.',
                             'Budgets are tile caps; low and high have different costs.',
                             'Greedy label assistance is not an exact achievable upper bound.',
                             'Original cache does not identify damage per class; class caps require new ledgers.']}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['specialist', 'controls'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--manifest')
    parser.add_argument('--cache')
    parser.add_argument('--base-run')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.threads < 1 or args.limit is not None and args.limit < 1:
        parser.error('Threads and limit must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'specialist':
        specialist_report(args.checkpoint, args.manifest, args.output, torch.device(args.device), args.precision, args.limit)
    else:
        cache_controls(args.cache, args.base_run, args.output)


if __name__ == '__main__':
    main()
