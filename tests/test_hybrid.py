import json
import sys

import numpy as np
import pytest
import torch

from selective_seg.data import load_config
from selective_seg.engine import config_signature
from selective_seg.hybrid_backbone import HybridSpecialist
from selective_seg.hybrid_cli import main
from selective_seg.hybrid_evidence import PatchSelector, pair_update, specialist_maps
from selective_seg.hybrid_pipeline import choose
from selective_seg.models import Segmenter


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def experiment(tmp_path):
    cfg = load_config('configs/smoke.json')
    cfg.update(classes=6, class_names=['urban','agriculture','rangeland','forest','water','barren'])
    cfg['utility']['class_weights'] = [1] * 6
    cfg['specialist'] = {'pretrained': False, 'input_size': 32, 'projection_dim': 8}
    cfg['hybrid'] = {'fusion': 'fpn', 'train_exits': True}
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    rng = np.random.default_rng(9)
    manifest = []
    for split in ('train','selector','val','test'):
        image, mask = tmp_path/f'{split}.npy', tmp_path/f'{split}_mask.npy'
        np.save(image, rng.integers(0,256,(3,55,65),dtype=np.uint8))
        truth = np.ones((55,65),dtype=np.uint8)
        truth[:,32:55] = 2
        truth[:,55:] = 4
        np.save(mask, truth)
        manifest.append({'id':split, 'split':split, 'image':str(image), 'mask':str(mask)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'base').mkdir()
    for stage in ('coarse','low','high'):
        segmenter = Segmenter(cfg, context=stage != 'coarse')
        with torch.no_grad():
            segmenter.classifier.weight.zero_()
            segmenter.classifier.bias.zero_()
            segmenter.classifier.bias[1 if stage == 'coarse' else 2] = 3
        torch.save({'stage':stage, 'signature':config_signature(cfg), 'model':segmenter.state_dict(),
                    'train_paths':[manifest[0]['image']]}, tmp_path/'base'/f'{stage}.pt')
    return cfg


@pytest.mark.parametrize('backbones,fusion', [('resnet','concat'), ('efficientnet','concat'),
                                             ('combined','concat'), ('combined','fpn'),
                                             ('combined','inception')])
def test_strides_and_gradient(experiment, backbones, fusion):
    cfg = experiment
    cfg['specialist']['backbones'] = backbones
    cfg['hybrid']['fusion'] = fusion
    model = HybridSpecialist(cfg)
    prediction = model(torch.randn(1,3,64,64))
    assert prediction.shape == (1,2,16,16)
    assert sorted(set(stage['stride'] for stage in model.feature_spec)) == [4,8,16,32]
    prediction.square().mean().backward()
    assert all(layer.weight.grad is not None for layer in model.projections)
    model.eval()
    later = []
    hook = list(model.branches.values())[0][-1].register_forward_hook(lambda *x: later.append(1))
    crop = torch.zeros(3,55,65)
    tile = (0,0,48,48)
    calibration = {'temperatures':[1,1,1], 'thresholds':{'confidence':[0,0]}}
    with torch.inference_mode():
        p, depth, _ = specialist_maps(model, crop, tile, cfg, torch.device('cpu'),
                                       torch.ones(1,48,48,dtype=torch.bool), calibration, 'confidence')
    hook.remove()
    assert p.shape == (1,2,48,48) and depth == 0 and not later


def test_protection_and_budgets():
    p = torch.tensor([.03,.55,.30,.02,.09,.01])[None,:,None,None].expand(1,6,1,3).clone()
    p[0,:,0,1] = torch.tensor([.03,.21,.15,.02,.58,.01])
    conditional = torch.tensor([.05,.95])[None,:,None,None].expand(1,2,1,3)
    result = pair_update(p, conditional, [1,2], torch.tensor([[[True,True,False]]]))
    assert result.argmax(1).tolist() == [[[2,4,1]]]
    torch.testing.assert_close(result[:,[0,3,4,5]], p[:,[0,3,4,5]])
    torch.testing.assert_close(result.sum(1), torch.ones(1,1,3))
    gains = torch.tensor([[1., 9.], [3., 2.], [-1.,0.]])
    allowed = torch.tensor([[False], [True], [True]])
    assert choose(gains, torch.ones(2), allowed, .05).tolist() == [0,1,0]
    assert not choose(gains, torch.ones(2), allowed, 0).any()
    for mode in ('none','mean','learned','conv'):
        selector = PatchSelector(10, mode, 2)
        assert selector(torch.rand(1,10), (1,1)).shape == (1,2)
        assert selector(torch.rand(6,10), (2,3)).shape == (6,2)


def test_cli_train_calibrate_route_and_pair(experiment, tmp_path, monkeypatch):
    root = tmp_path

    def run(command, output, *extra):
        monkeypatch.setattr(sys, 'argv', ['hybrid', command, '--config', str(root/'config.json'),
                         '--manifest', str(root/'manifest.json'), '--base-run', str(root/'base'),
                         '--output', str(root/output), '--device', 'cpu', '--passes', '1',
                         '--warmup', '0', '--train-exits', '--support-mass', '0',
                         '--candidate-fraction', '1', '--neighbourhood', 'conv',
                         '--actions', 'specialist', 'low', 'high', *map(str,extra)])
        main()

    run('train-specialist', 'trained', '--steps', 1)
    run('calibrate','cal','--specialist-run',root/'trained')
    cal = root/'cal'/'calibration.json'
    shared = ('--specialist-run',root/'trained','--calibration',cal,'--exit-policy','confidence')
    run('specialist','binary',*shared)
    assert sum(json.loads((root/'binary'/'metrics.json').read_text())['exit_counts']) == 4
    for split in ('selector','val'):
        run('cache',f'caches/{split}',*shared,'--split',split)
    run('train-selector','selector','--selector-run',root/'caches','--epochs',2,*shared)
    run('evaluate','evaluation',*shared,'--selector-run',root/'selector','--split','test',
        '--all-budgets','--save-maps')
    report = json.loads((root/'evaluation'/'budget_1'/'metrics.json').read_text())
    assert report['image_ids'] == ['test'] and report['specialist_invocations'] == 4
    assert report['NLL'] >= 0 and 'selective_risk_coverage' in report
    from selective_seg.hybrid_metrics import paired_bootstrap
    assert paired_bootstrap(report['images'], report['images'], 10, 1)['delta_mIoU_CI95'] == [0,0]
    with pytest.raises(FileExistsError):
        run('evaluate','evaluation',*shared,'--selector-run',root/'selector')
