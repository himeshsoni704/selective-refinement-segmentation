"""Sequential, restartable development experiments with frozen checkpoints.

Run with python -m selective_seg.diagnostic_suite --help. Does not train new
segmenters, use the test split, or claim a calibrated statistical guarantee.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from .diagnostics import write
from .engine import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run', default='runs/deepglobe')
    p.add_argument('--manifest', default='runs/deepglobe/manifest.json')
    p.add_argument('--specialist-run', default='runs/hybrid-fpn-seed42-retry-fixed/specialist')
    p.add_argument('--original-specialist-run', default='runs/specialist_combined')
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--limit', type=int, help='Pilot only: first N selector/val images; label outputs as subsets')
    args = p.parse_args()
    if min(args.threads, args.passes, args.epochs) < 1 or args.limit is not None and args.limit < 1:
        p.error('Counts must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; select CPU explicitly for CPU experiments')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'suite.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another suite is using this output directory')
    files = {name: fingerprint(Path(args.base_run)/f'{name}.pt') for name in ('coarse', 'low', 'high')}
    files['specialist'] = fingerprint(Path(args.specialist_run)/'specialist.pt')
    files['original_specialist'] = fingerprint(Path(args.original_specialist_run)/'specialist.pt')
    files['manifest'] = fingerprint(args.manifest)
    # Bind resume to both inputs and source, including uncommitted modifications.
    files['source'] = {str(path): fingerprint(path) for path in sorted(Path('selective_seg').glob('*.py'))}
    identity = {'arguments': vars(args), 'sha256': files}
    identity_path = output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        p.error('Output belongs to different inputs/options/source; use a fresh output directory')
    write(identity_path, identity)
    manifest = args.manifest
    if args.limit:
        records = json.loads(Path(manifest).read_text())
        chosen = []
        for split in ('selector', 'val'):
            chosen.extend([r for r in records if r['split'] == split][:args.limit])
        # Resolve paths relative to original manifest before moving the file.
        for record in chosen:
            for key in ('image', 'mask'):
                record[key] = str((Path(manifest).resolve().parent/record[key]).resolve())
        manifest = str(output/'pilot_manifest.json')
        write(manifest, chosen)
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
           'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    stages = []

    def stage(name, module, arguments, artifact):
        command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
        marker = output/f'{name}.done.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['command'] == command and Path(artifact).is_file() and saved['artifact_sha256'] == fingerprint(artifact):
                stages.append({'name': name, 'status': 'complete', 'resumed': True})
                return
            raise ValueError(f'Invalid completion marker for {name}')
        write(output/'status.json', {'status': 'running', 'current': name, 'stages': stages, 'pid': os.getpid()})
        print(json.dumps({'stage': name, 'status': 'starting'}), flush=True)
        if module == 'selective_seg.hybrid_cli':
            # Preserve unfinished stage outputs before retrying its strict new-dir CLI.
            destination = Path(arguments[arguments.index('--output') + 1])
            if destination.exists():
                destination.rename(destination.with_name(destination.name + f'.interrupted-{time.time_ns()}'))
        started = time.time()
        with (output/f'{name}.log').open('a') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(output/'status.json', {'status': 'failed', 'current': name, 'returncode': result.returncode,
                                         'log': str(output/f'{name}.log'), 'stages': stages})
            raise RuntimeError(f'{name} failed: see {output / (name + ".log")}')
        if not Path(artifact).is_file():
            raise RuntimeError(f'{name} did not produce {artifact}')
        if module == 'selective_seg.diagnostics' and arguments[0] == 'specialist':
            if not json.loads(Path(artifact).read_text()).get('metrics_valid', False):
                raise RuntimeError(f'{name} produced invalid numerical outputs; see {artifact}')
        write(marker, {'command': command, 'artifact_sha256': fingerprint(artifact), 'seconds': time.time()-started})
        stages.append({'name': name, 'status': 'complete'})

    stage('cache_controls', 'selective_seg.diagnostics', ['controls', '--cache', Path(args.base_run)/'val_targets.pt',
          '--base-run', args.base_run, '--output', output/'cache_controls.json'], output/'cache_controls.json')
    for name, run in [('fpn', args.specialist_run), ('original', args.original_specialist_run)]:
        options = ['specialist', '--checkpoint', Path(run)/'specialist.pt', '--manifest', args.manifest,
                   '--output', output/f'{name}_fp32_val.json', '--device', args.device, '--threads', args.threads]
        if args.limit:
            options += ['--limit', args.limit]
        stage(name+'_validation', 'selective_seg.diagnostics', options, output/f'{name}_fp32_val.json')
    # Use the actual specialist configuration, never silently reconstruct its geometry.
    checkpoint_cfg = json.loads((Path(args.specialist_run)/'config.json').read_text())
    write(output/'frozen_config.json', checkpoint_cfg)
    common = ['--config', output/'frozen_config.json', '--manifest', manifest, '--base-run', args.base_run,
              '--device', args.device, '--precision', 'fp32', '--passes', args.passes, '--warmup', '1',
              '--candidate-fraction', '.3', '--refine-fraction', '.2', '--actions', 'low', 'high']
    for mode in ('none', 'full'):
        evidence = ['--evidence', mode]
        if mode != 'none':
            evidence += ['--specialist-run', args.specialist_run]
        for split in ('selector', 'val'):
            destination = output/mode/split
            stage(f'{mode}_cache_{split}', 'selective_seg.hybrid_cli',
                  ['cache', *common, *evidence, '--split', split, '--output', destination], destination/'cache.pt')
        trained = output/mode/'trained'
        stage(f'{mode}_selector', 'selective_seg.hybrid_cli',
              ['train-selector', *common, *evidence, '--selector-run', output/mode,
               '--epochs', args.epochs, '--output', trained], trained/'selector.pt')
        evaluated = output/mode/'evaluation'
        stage(f'{mode}_evaluate', 'selective_seg.hybrid_cli',
              ['evaluate', *common, *evidence, '--selector-run', trained, '--split', 'val',
               '--all-budgets', '--output', evaluated], evaluated/'budget_1'/'metrics.json')
    stage('paired_comparison', 'selective_seg.routing_tradeoffs',
          ['--root', output], output/'paired_comparison.json')
    write(output/'status.json', {'status': 'complete', 'stages': stages, 'subset': args.limit is not None,
                                'note': 'Development validation only. Review before new backbone training or final test.'})


if __name__ == '__main__':
    main()
