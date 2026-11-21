"""Frozen-model class-transition experiment; see TRANSITIONS.md.

Fresh pixel-aligned transition targets, selector-only fit/tune split, paired
validation replay and actual end-to-end inference. The test split is never read.
"""
import argparse
import copy
import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .benchmark import environment, summarize_latency
from .data import load_config, read_manifest, read_sample
from .diagnostics import write
from .engine import fingerprint, sync
from .hybrid_evidence import PatchSelector, area
from .hybrid_metrics import Metrics, paired_bootstrap
from .hybrid_pipeline import build
from .objectives import confusion_matrix, tensor_miou
from .specialist import counts_metrics
from .training import seed_all
from .transition_selector import (TransitionSelector, allocate, correction_damage,
                                  scalar_allocate, transition_counts, transition_loss)

BUDGETS = (.05, .1, .2, .3, .5, 1.)
POLICIES = {
    'scalar': ('scalar', {}),
    'independent': ('independent', {}),
    'shared_static': ('shared', {'recompute': False}),
    'shared': ('shared', {}),
    'shared_penalty1': ('shared', {'damage_penalty': 1.}),
    'shared_penalty2': ('shared', {'damage_penalty': 2.}),
    'shared_cap01': ('shared', {'damage_fraction': .01}),
    'shared_cap05': ('shared', {'damage_fraction': .05}),
}


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def configuration(path):
    cfg = load_config(path)
    cfg['research'] = {}
    cfg['hybrid'] = dict(evidence='none', actions=['low', 'high'], neighbourhood='none',
        candidate_fraction=.3, min_pair_mass=.1, support_mass=.35, refine_fraction=.2,
        action_time_penalty=0., exit_policy='full', fusion='fpn', precision='fp32', train_exits=False)
    if cfg['classes'] != 6:
        raise ValueError('This experiment uses the six-class A/R merge')
    return cfg


@torch.inference_mode()
def cache_row(pipeline, record):
    cfg, classes = pipeline.cfg, pipeline.cfg['classes']
    image, truth = read_sample(record, cfg)
    pc, grid, chosen, predictions, z, _, _ = pipeline.prepare(image)
    coarse = pc.argmax(1)[0]
    before, histograms, joints, costs = [], [], [], []
    for i, tile in enumerate(grid):
        target, initial = area(truth, tile), area(coarse, tile)
        before.append(confusion_matrix(initial, target, classes, cfg['ignore_index']))
        # All pixels, including ignored labels: available without truth at inference.
        histograms.append(torch.bincount(initial.flatten(), minlength=classes))
        transitions, measured = [], []
        for action in pipeline.actions:
            if i in chosen:
                sync(pipeline.device)
                start = time.perf_counter()
                final = pipeline.option(image, pc, tile, action, predictions.get(i)).argmax(1)[0]
                sync(pipeline.device)
                measured.append((time.perf_counter() - start) * 1000)
                transitions.append(transition_counts(target, initial, final, classes, cfg['ignore_index']))
            else:
                measured.append(0.)
                transitions.append(torch.zeros(classes, classes, classes, dtype=torch.long))
        joints.append(torch.stack(transitions))
        costs.append(measured)
    before, joint = torch.stack(before), torch.stack(joints)
    allowed = torch.tensor([i in chosen for i in range(len(grid))])
    if not torch.equal(joint[allowed].sum(-1), before[allowed, None].expand(-1, len(pipeline.actions), -1, -1)):
        raise RuntimeError('Pixel transition targets violate shared coarse marginals')
    cm = before.sum(0)
    gains = tensor_miou((cm[None, None] - before[:, None] + joint.sum(-2)).double()) - tensor_miou(cm.double())
    corrected, damaged = correction_damage(joint)
    return dict(id=record['id'], z=z.cpu().float(), shape=pipeline.shape(grid), allowed=allowed,
                histogram=torch.stack(histograms), tile_coarse_counts=before, joint=joint,
                gains=gains.float(), costs=torch.tensor(costs), coarse_counts=cm,
                corrected_by_class=corrected, damaged_by_class=damaged)


def cache_split(pipeline, records, directory, status):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        path = directory / f'{index:04d}.pt'
        if path.exists():
            row = torch.load(path, weights_only=True)
            if row['id'] != record['id']:
                raise ValueError('Cache record identity differs')
        else:
            row = cache_row(pipeline, record)
            atomic_save(path, row)
            # Materialize ordinary tensors outside inference_mode for head training.
            row = torch.load(path, weights_only=True)
        rows.append(row)
        status(f'cache_{directory.name}', image=index + 1, total=len(records))
    return rows


def partition(records):
    """Fixed image-level partition, independent of model initialization seed."""
    if len(records) < 2:
        raise ValueError('Need at least two selector images for disjoint fit/tune sets')
    indices = torch.randperm(len(records), generator=torch.Generator().manual_seed(20261121)).tolist()
    cut = max(1, int(.75 * len(indices)))
    return indices[:cut], indices[cut:]


def make_model(mode, dimension, classes):
    if mode == 'scalar':
        return PatchSelector(dimension, 'none', 2)
    return TransitionSelector(dimension, classes, independent=mode == 'independent')


def fit(mode, rows, fit_indices, tune_indices, cfg, seed, epochs, destination, status):
    model = make_model(mode, rows[0]['z'].shape[1], cfg['classes'])
    if destination.exists():
        ck = torch.load(destination, weights_only=True)
        model.load_state_dict(ck['model'])
        return model.eval(), ck
    seed_all(seed)
    model = make_model(mode, rows[0]['z'].shape[1], cfg['classes'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['selector_lr'])
    best, stale, history, best_state, best_epoch = float('inf'), 0, [], None, None

    def loss(row):
        if mode == 'scalar':
            mask = row['allowed'] & (row['tile_coarse_counts'].sum((-1, -2)) > 0)
            if not mask.any():
                return model(row['z'], row['shape']).sum() * 0
            return F.smooth_l1_loss(model(row['z'], row['shape'])[mask], row['gains'][mask], beta=.005)
        return transition_loss(model(row['z'], row['histogram']), row)

    for epoch in range(epochs):
        model.train()
        for index in torch.randperm(len(fit_indices)).tolist():
            optimizer.zero_grad(set_to_none=True)
            value = loss(rows[fit_indices[index]])
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite selector training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = sum(float(loss(rows[i])) for i in tune_indices) / len(tune_indices)
        if not torch.isfinite(torch.tensor(validation)):
            raise RuntimeError('Nonfinite selector tuning loss')
        history.append({'epoch': epoch + 1, 'tuning_loss': validation})
        if validation < best:
            best, best_state, stale, best_epoch = validation, copy.deepcopy(model.state_dict()), 0, epoch + 1
        else:
            stale += 1
        status(f'train_{mode}_seed{seed}', epoch=epoch + 1, tuning_loss=validation, best_epoch=best_epoch)
        if stale >= cfg['training']['patience']:
            break
    model.load_state_dict(best_state)
    ck = dict(model=best_state, mode=mode, seed=seed, dimension=rows[0]['z'].shape[1],
              classes=cfg['classes'], best_epoch=best_epoch, tuning_loss=best, history=history,
              fit_ids=[rows[i]['id'] for i in fit_indices], tune_ids=[rows[i]['id'] for i in tune_indices],
              parameters=sum(p.numel() for p in model.parameters()))
    atomic_save(destination, ck)
    return model.eval(), ck


@torch.inference_mode()
def select(model, mode, z, histogram, shape, allowed, costs, fraction, options):
    # Only pre-action descriptors, coarse predictions, candidates and fixed costs enter here.
    maximum = fraction * len(z) * float(costs[-1])
    if mode == 'scalar':
        return scalar_allocate(model(z, shape), allowed, costs, fraction, maximum)
    prediction = model(z, histogram)
    return allocate(prediction['before'], prediction['joint'], allowed, costs, fraction, maximum, **options)


def observed(row, selected, estimated):
    cm = row['coarse_counts'].clone()
    corrected = torch.zeros(cm.shape[0], dtype=torch.long)
    damaged = torch.zeros_like(corrected)
    for i, value in enumerate(selected.tolist()):
        if value:
            a = value - 1
            cm += row['joint'][i, a].sum(-2) - row['tile_coarse_counts'][i]
            corrected += row['corrected_by_class'][i, a].long()
            damaged += row['damaged_by_class'][i, a].long()
    fraction = estimated.get('damage_fraction')
    true_correct = row['coarse_counts'].diagonal()
    violation = (damaged > fraction * true_correct).tolist() if fraction is not None else None
    return dict(id=row['id'], confusion=cm.tolist(), selected=selected.tolist(),
                corrected_by_class=corrected.tolist(), damaged_by_class=damaged.tolist(),
                actual_damage_cap_violation_by_class=violation, **estimated)


def summarize(rows, base):
    cm = torch.tensor([r['confusion'] for r in rows]).sum(0)
    result = counts_metrics(cm)
    result.update(images=rows, paired_vs_coarse=paired_bootstrap(rows, base, 1000, 42),
                  corrected_by_class=torch.tensor([r['corrected_by_class'] for r in rows]).sum(0).tolist(),
                  damaged_by_class=torch.tensor([r['damaged_by_class'] for r in rows]).sum(0).tolist(),
                  selected_tiles=sum(sum(v > 0 for v in r['selected']) for r in rows),
                  estimated_action_ms=sum(r['estimated_action_ms'] for r in rows))
    if rows[0].get('predicted_damage') is not None:
        expected = torch.tensor([r['predicted_damage'] for r in rows])
        actual = torch.tensor([r['damaged_by_class'] for r in rows])
        result['damage_count_MAE_by_class'] = (expected - actual).abs().mean(0).tolist()
    if rows[0]['actual_damage_cap_violation_by_class'] is not None:
        result['actual_damage_cap_violations_by_class'] = torch.tensor(
            [r['actual_damage_cap_violation_by_class'] for r in rows]).sum(0).tolist()
    return result


class TransitionPipeline:
    """Deployable wrapper: predict(image) never accepts a mask or cached outcomes."""
    def __init__(self, base, selector, mode, costs, fraction=.2, options=None):
        self.base, self.selector, self.mode = base, selector.cpu().eval(), mode
        self.costs, self.fraction, self.options = costs, fraction, options or {}

    @torch.inference_mode()
    def predict(self, image):
        pipeline = self.base
        sync(pipeline.device)
        start = time.perf_counter()
        pc, grid, chosen, probabilities, z, _, _ = pipeline.prepare(image)
        coarse = pc.argmax(1)[0]
        histogram = torch.stack([torch.bincount(area(coarse, t).flatten(), minlength=pipeline.cfg['classes']) for t in grid])
        selected, details = select(self.selector, self.mode, z.cpu().float(), histogram,
            pipeline.shape(grid), torch.tensor([i in chosen for i in range(len(grid))]),
            self.costs, self.fraction, self.options)
        result = pc.clone()
        for i, value in enumerate(selected.tolist()):
            if value:
                area(result, grid[i]).copy_(pipeline.option(image, pc, grid[i], pipeline.actions[value-1], probabilities.get(i)))
        sync(pipeline.device)
        details['end_to_end_ms'] = (time.perf_counter() - start) * 1000
        return result[0], selected, details


def live_evaluate(pipeline, model, mode, options, costs, records, cached, passes, status):
    wrapper = TransitionPipeline(pipeline, model, mode, costs, .2, options)
    first, _ = read_sample(records[0], pipeline.cfg)
    wrapper.predict(first)  # excluded warmup
    timings, metrics, rows = [], Metrics(pipeline.cfg['classes'], ignore=pipeline.cfg['ignore_index']), []
    for repetition in range(passes):
        for i, record in enumerate(records):
            image, truth = read_sample(record, pipeline.cfg)
            probability, selected, details = wrapper.predict(image)
            timings.append(details['end_to_end_ms'])
            if selected.tolist() != cached['images'][i]['selected']:
                raise RuntimeError('Live and cached routing disagree')
            if repetition == 0:
                cm = metrics.add(probability, truth)
                if cm != cached['images'][i]['confusion']:
                    raise RuntimeError('Live and cached segmentation confusion disagree')
                rows.append(dict(id=record['id'], confusion=cm))
            status('live_evaluation', mode=mode, policy=options, repetition=repetition + 1,
                   image=i + 1, total=len(records))
    by_image = torch.tensor(timings).reshape(passes, len(records)).T.tolist()
    return {**metrics.report(), 'images': rows, 'timing': summarize_latency(by_image),
            'passes': passes, 'budget': .2, 'live_replay_match': True,
            'scope': 'Includes overview, CPU selector/allocator, transfers and selected refinements; excludes decoding/metrics.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/deepglobe_specialist.json')
    p.add_argument('--manifest', default='runs/deepglobe/manifest.json')
    p.add_argument('--base-run', default='runs/deepglobe')
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--wait-for-status', help='Wait for this existing suite to complete before using the GPU')
    p.add_argument('--wait-timeout-hours', type=float, default=48.)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--limit', type=int, help='Pilot only: N images per split (at least two)')
    p.add_argument('--skip-live', action='store_true', help='Replay only; do not report end-to-end latency')
    args = p.parse_args(argv)
    if min(args.threads, args.epochs, args.passes) < 1 or args.limit is not None and args.limit < 2:
        p.error('Positive counts required; pilot limit must be at least two')
    if args.wait_timeout_hours <= 0 or not torch.isfinite(torch.tensor(args.wait_timeout_hours)):
        p.error('Wait timeout must be finite and positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'experiment.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)

    def status(stage, **extra):
        value = dict(status='running', current=stage, pid=os.getpid(), timestamp=time.time(), **extra)
        write(output/'status.json', value)
        print(json.dumps(value), flush=True)

    try:
        cfg = configuration(args.config)
        # Identity applies to every resumed row, head, and evaluation.
        hashes = {str(path): fingerprint(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
        hashes.update({str(path): fingerprint(path) for path in
                       [Path(args.manifest), Path(args.config), *[Path(args.base_run)/f'{s}.pt' for s in ('coarse', 'low', 'high')]]})
        identity = dict(arguments=vars(args), hashes=hashes, config=cfg)
        identity_path = output/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Source, input or arguments changed; use a new output directory')
        write(identity_path, identity)
        if args.wait_for_status:
            started = time.monotonic()
            while True:
                dependency = json.loads(Path(args.wait_for_status).read_text())
                if dependency['status'] == 'complete':
                    break
                if dependency['status'] == 'failed':
                    raise RuntimeError('Earlier experiment failed; inspect its logs before continuing')
                if time.monotonic() - started > args.wait_timeout_hours * 3600:
                    raise TimeoutError('Earlier experiment did not finish within the wait limit')
                status('waiting_for_existing_experiment', dependency=dependency.get('current'))
                time.sleep(30)
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; choose CPU explicitly')
        pipeline, provenance, forbidden, tuned = build(cfg, SimpleNamespace(base_run=args.base_run), device)
        records = read_manifest(args.manifest)
        sets = {s: [r for r in records if r['split'] == s][:args.limit] for s in ('selector', 'val')}
        for split, samples in sets.items():
            if not samples or {r['image'] for r in samples} & (forbidden | tuned):
                raise ValueError(f'Missing or contaminated {split} split')
        fit_indices, tune_indices = partition(sets['selector'])
        write(output/'protocol.json', dict(provenance=provenance, classes=cfg['class_names'],
            fit_ids=[sets['selector'][i]['id'] for i in fit_indices],
            tune_ids=[sets['selector'][i]['id'] for i in tune_indices],
            evaluation_ids=[r['id'] for r in sets['val']], subset=args.limit is not None,
            seeds=args.seeds, budgets=BUDGETS, policies=POLICIES,
            note='Development validation, not a final generalization claim. No test labels used. Expected damage caps are not risk guarantees.'))
        caches = {s: cache_split(pipeline, samples, output/'cache'/s, status) for s, samples in sets.items()}
        fit_costs = torch.cat([caches['selector'][i]['costs'][caches['selector'][i]['allowed']] for i in fit_indices])
        if len(fit_costs) == 0:
            raise ValueError('No candidate timing observations in the fitting subset')
        costs = fit_costs.median(0).values
        if not torch.isfinite(costs).all() or (costs <= 0).any():
            raise ValueError('No valid timing observations in the fitting subset')
        write(output/'costs.json', dict(action_median_ms=costs.tolist(),
            scope='Fit-only cache timings. Estimated action budget = fraction * tile count * high cost; not a hard runtime guarantee.'))
        baseline = [dict(id=r['id'], confusion=r['coarse_counts'].tolist()) for r in caches['val']]
        summary = dict(protocol='development validation', environment=environment(device),
                       subset=args.limit is not None, seeds={}, costs_ms=costs.tolist())
        for seed in args.seeds:
            trained = {mode: fit(mode, caches['selector'], fit_indices, tune_indices, cfg, seed,
                        args.epochs, output/f'seed{seed}'/f'{mode}.pt', status)
                       for mode in ('scalar', 'independent', 'shared')}
            seed_report = {}
            for policy, (mode, options) in POLICIES.items():
                model = trained[mode][0]
                policy_report = {}
                for fraction in BUDGETS:
                    rows = []
                    for row in caches['val']:
                        chosen, estimated = select(model, mode, row['z'], row['histogram'], row['shape'],
                            row['allowed'], costs, fraction, options)
                        rows.append(observed(row, chosen, estimated))
                    report = summarize(rows, baseline)
                    path = output/f'seed{seed}'/policy/f'budget_{fraction:g}.json'
                    write(path, report)
                    policy_report[f'{fraction:g}'] = {k: v for k, v in report.items() if k != 'images'}
                    status('replay', seed=seed, policy=policy, budget=fraction)
                seed_report[policy] = policy_report
            # Paired comparison against the matched scalar control, not just coarse.
            for policy in POLICIES:
                for fraction in BUDGETS:
                    directory = output/f'seed{seed}'
                    filename = f'budget_{fraction:g}.json'
                    rows = json.loads((directory/policy/filename).read_text())['images']
                    reference = json.loads((directory/'scalar'/filename).read_text())['images']
                    seed_report[policy][f'{fraction:g}']['paired_vs_scalar'] = paired_bootstrap(rows, reference, 1000, seed)
            summary['seeds'][str(seed)] = seed_report
            write(output/'summary.json', summary)
            if not args.skip_live and seed == args.seeds[0]:
                for policy in ('scalar', 'independent', 'shared', 'shared_cap01'):
                    mode, options = POLICIES[policy]
                    target = output/f'seed{seed}'/policy/'live_budget_0.2.json'
                    if not target.exists():
                        cached = json.loads((target.parent/'budget_0.2.json').read_text())
                        result = live_evaluate(pipeline, trained[mode][0], mode, options, costs,
                            sets['val'], cached, args.passes, status)
                        write(target, result)
        write(output/'status.json', dict(status='complete', subset=args.limit is not None,
              summary=str(output/'summary.json'), timestamp=time.time(),
              note='Review observed damage and per-class IoU. Predicted caps provide no statistical safety guarantee.'))
    except Exception as error:
        write(output/'status.json', dict(status='failed', error=repr(error), timestamp=time.time()))
        raise


if __name__ == '__main__':
    main()
