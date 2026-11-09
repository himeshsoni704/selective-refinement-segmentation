"""CPU end-to-end research workflow; synthetic fixtures are NOT accuracy evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from selective_seg.data import load_config,read_manifest
from selective_seg.training import train_segmenter,cache_targets,train_selector,calibrate
from selective_seg.auxiliary import train_auxiliary,train_escalation
from selective_seg.engine import Pipeline
from selective_seg.cli import apply_calibration
from selective_seg.benchmark import evaluate


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='selective-research-') as work:
        root = Path(work)
        subprocess.run([sys.executable,'make_demo.py','--output',str(root/'data'),'--auxiliary','2'],check=True)
        records = read_manifest(root/'data/manifest.json')
        cfg = load_config('configs/smoke.json')
        cfg['training'].update(coarse_steps=1,refiner_steps=1,selector_epochs=1,
                               validation_all_tiles=True,checkpoint_metric='miou',validation_images=0,
                               augment=True,class_balanced_tiles=True,soft_iou_weight=.1)
        run = root/'run'
        for stage in ('coarse','low','high'):
            train_segmenter(cfg,records,run,stage,'cpu')
        cfg['research'] = {'descriptor_grid':2,'scout':True,'acceptance':True,'escalation':True,'shared_marginals':True}
        cfg['routing'].update(policy='direct',max_refine_ms=100.,lambda_grid=[0.,.001])
        cfg['utility']['mode'] = 'miou'
        for name in ('scout','acceptance'):
            train_auxiliary(cfg,records,run,name,'cpu',1)
        for split in ('selector','val'):
            cache_targets(cfg,records,run,split,run/(split+'_targets.pt'),'cpu')
        train_selector(cfg,run/'selector_targets.pt',run/'val_targets.pt',run,'cpu',1)
        train_escalation(cfg,run/'selector_targets.pt',run/'val_targets.pt',run,'cpu',1)
        calibrate(cfg,run/'val_targets.pt',run,run/'calibration.json')
        cfg = apply_calibration(cfg,run/'calibration.json',run)
        pipe = Pipeline.from_run(cfg,run,'cpu',True)
        test = [r for r in records if r['split']=='test']
        report = evaluate(pipe,test,root/'results',warmup=1,passes=2,profile_stages=True,profile_operations=True)
        assert report['mean_estimated_flops'] > 0
        assert report['warmup'] == 1 and report['passes'] == 2
        # Baselines and their benchmark reports execute independently of learned selection.
        for mode in ('coarse','low','high','entropy','random'):
            report = evaluate(pipe,test,root/mode,mode,warmup=0,passes=1)
            assert report['mode'] == mode
        # Auxiliary leakage must be rejected before generating selector targets.
        bad = [dict(r,split='selector') if r['split']=='auxiliary' else r for r in records]
        try:
            cache_targets(cfg,bad,run,'selector',run/'bad.pt','cpu')
        except ValueError as exc:
            assert 'overlaps' in str(exc)
        else:
            raise AssertionError('Auxiliary training overlap was accepted')
    print('RESEARCH SMOKE PASSED: auxiliary training, grouped metric selector, escalation, calibration, profiling and baselines')


if __name__ == '__main__':
    main()
