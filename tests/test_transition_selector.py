import json

import numpy as np
import pytest
import torch

from selective_seg.data import load_config
from selective_seg.engine import config_signature
from selective_seg.models import Segmenter
from selective_seg.objectives import confusion_matrix
from selective_seg.transition_selector import (TransitionSelector, allocate, correction_damage,
                                              transition_counts, transition_loss)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_pixel_joint_preserves_both_marginals_and_damage():
    y = torch.tensor([0, 1, 2, 1, 2, 255, 0])
    c = torch.tensor([0, 1, 1, 2, 2, 1, 1])
    r = torch.tensor([0, 2, 2, 1, 2, 2, 2])
    joint = transition_counts(y, c, r, 3)
    before, after = confusion_matrix(c, y, 3), confusion_matrix(r, y, 3)
    assert torch.equal(joint.sum(-1), before)
    assert torch.equal(joint.sum(-2), after)
    corrected, damaged = correction_damage(joint)
    assert corrected.tolist() == [0, 1, 1]
    assert damaged.tolist() == [0, 1, 0]
    assert joint[0, 1, 2] == 1  # wrong-to-wrong, lost by four-outcome summaries
    assert torch.equal(after.diagonal() - before.diagonal(), corrected - damaged)


def test_shared_model_conservation_gradients_and_void_mass():
    torch.manual_seed(4)
    model = TransitionSelector(8)
    histogram = torch.tensor([[2, 4, 3, 1, 0, 2], [0, 1, 0, 0, 0, 0.]])
    z = torch.randn(2, 8)
    output = model(z, histogram)
    joint, before = output['joint'], output['before']
    torch.testing.assert_close(joint.sum(-1), before[:, None].expand(-1, 2, -1, -1))
    torch.testing.assert_close(joint.sum((-2, -1)), before.sum(-1)[:, None].expand(-1, 2, -1))
    assert (before.sum(1) <= histogram).all()
    for coarse in (0, 3, 4, 5):
        others = [i for i in range(6) if i != coarse]
        assert not joint[..., coarse, others].any()
    target = torch.zeros_like(joint)
    target[0, :, 1, 1, 2] = 3
    target[0, :, 2, 2, 1] = 2
    row = dict(joint=target, tile_coarse_counts=target[:, 0].sum(-1), histogram=histogram)
    loss = transition_loss(output, row)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # An entirely ignored batch remains finite, and cannot supervise class transitions.
    row['joint'].zero_()
    row['tile_coarse_counts'].zero_()
    assert torch.isfinite(transition_loss(model(z, histogram), row))


def test_independent_ablation_can_disagree_on_before_marginals():
    model = TransitionSelector(4, independent=True)
    p = model(torch.randn(2, 4), torch.ones(2, 6))
    assert not torch.allclose(p['joint'].sum(-1)[:, 0], p['before'])
    assert not torch.allclose(p['joint'].sum(-1)[:, 0], p['joint'].sum(-1)[:, 1])


def test_allocator_time_candidate_and_damage_caps():
    # Tile 0 corrects ten rangeland pixels but damages one agriculture pixel.
    # Tile 1 corrects two with no damage. Tile 2 is forbidden despite a large gain.
    before = torch.zeros(3, 3, 3)
    joint = torch.zeros(3, 1, 3, 3, 3)
    for i, recovered in enumerate([10, 2, 100]):
        before[i, 1, 1] = 20
        before[i, 2, 1] = recovered
        joint[i, 0, 1, 1, 1] = 20 - (i == 0)
        joint[i, 0, 1, 1, 2] = i == 0
        joint[i, 0, 2, 1, 2] = recovered
    allowed = torch.tensor([True, True, False])
    selected, info = allocate(before, joint, allowed, [2.], 1., max_ms=2., damage_fraction=0.)
    assert selected.tolist() == [0, 1, 0]
    assert info['estimated_action_ms'] == 2
    assert info['predicted_damage'] == [0, 0, 0]
    selected, info = allocate(before, joint, allowed, [2.], 1., max_ms=2.)
    assert selected.tolist() == [1, 0, 0]
    assert info['predicted_after_mIoU'] > info['predicted_before_mIoU']
    assert not allocate(before, joint, allowed, [2.], 0.)[0].any()
    assert not allocate(before, joint, allowed, [2.], 1., max_ms=0.)[0].any()
    with pytest.raises(ValueError):
        allocate(before, joint, allowed, [float('nan')], 1.)


def test_full_cache_train_replay_live_and_resume(tmp_path, monkeypatch):
    from selective_seg import transition_experiment as experiment
    cfg = load_config('configs/smoke.json')
    cfg.update(classes=6, class_names=['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren'])
    cfg['utility']['class_weights'] = [1] * 6
    config = tmp_path/'config.json'
    config.write_text(json.dumps(cfg))
    records = []
    for i, split in enumerate(('selector', 'selector', 'val', 'val')):
        image, mask = tmp_path/f'{i}.npy', tmp_path/f'{i}_mask.npy'
        np.save(image, np.random.default_rng(i).integers(0, 256, (3, 55, 65), dtype=np.uint8))
        target = np.ones((55, 65), dtype=np.uint8)
        target[:, 30:] = 2
        target[:2] = 255
        np.save(mask, target)
        records.append(dict(id=str(i), image=str(image), mask=str(mask), split=split))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(records))
    base = tmp_path/'base'
    base.mkdir()
    for stage in ('coarse', 'low', 'high'):
        model = Segmenter(cfg, context=stage != 'coarse')
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.bias[1 if stage == 'coarse' else 2] = 3
        torch.save(dict(stage=stage, signature=config_signature(cfg), model=model.state_dict(), train_paths=[]), base/f'{stage}.pt')
    args = ['--config', str(config), '--manifest', str(manifest), '--base-run', str(base),
            '--output', str(tmp_path/'run'), '--device', 'cpu', '--threads', '1', '--epochs', '2',
            '--seeds', '42', '--passes', '1']
    experiment.main(args)
    result = tmp_path/'run'
    assert json.loads((result/'status.json').read_text())['status'] == 'complete'
    protocol = json.loads((result/'protocol.json').read_text())
    assert not set(protocol['fit_ids']) & set(protocol['tune_ids'])
    assert not (set(protocol['fit_ids']) | set(protocol['tune_ids'])) & set(protocol['evaluation_ids'])
    for name in ('scalar', 'independent', 'shared', 'shared_cap01'):
        report = json.loads((result/'seed42'/name/'live_budget_0.2.json').read_text())
        assert report['live_replay_match']
        assert report['timing']['mean_image_latency_ms'] > 0
    monkeypatch.setattr(experiment, 'cache_row', lambda *a: pytest.fail('Completed cache recomputed'))
    experiment.main(args)
    config.write_text(config.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        experiment.main(args)
