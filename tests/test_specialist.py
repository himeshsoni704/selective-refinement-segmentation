import copy
from pathlib import Path
import pytest
import torch

pytest.importorskip('torchvision')
from selective_seg.data import load_config
from selective_seg.specialist import (MultiScaleSpecialist, binary_target, masked_loss,
    counts_metrics, train_specialist, evaluate_specialist, signature)


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    c = load_config(Path(__file__).parents[1]/'configs/deepglobe_specialist.json')
    c.update(backbone='tiny',tile_size=32,border=8,coarse_size=64,low_size=32)
    c['specialist'].update(pretrained=False,input_size=64,projection_dim=4,steps=1,validate_every=1)
    return c


@pytest.mark.parametrize('branch,levels,n', [('combined','all',8),('resnet','shallow',2),
                                             ('efficientnet','deep',2)])
def test_real_backbones_shapes_and_gradients(cfg,branch,levels,n):
    cfg['specialist'].update(backbones=branch,levels=levels)
    model = MultiScaleSpecialist(cfg)
    out = model(torch.rand(1,3,65,61))
    assert out.shape[:2] == (1,2) and len(model.feature_spec) == n
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d))


def test_resnet50_actual_channels(cfg):
    cfg['specialist'].update(backbones='resnet',resnet='resnet50')
    model = MultiScaleSpecialist(cfg)
    assert [v['channels'] for v in model.feature_spec] == [256,512,1024,2048]


def test_ignore_labels_and_gradients(cfg):
    y = torch.tensor([[[0,1,2,3,4,5,255]]])
    target = binary_target(y,cfg)
    assert target.tolist() == [[[255,0,1,255,255,255,255]]]
    logits = torch.randn(1,2,1,7,requires_grad=True)
    masked_loss(logits,target).backward()
    assert torch.equal(logits.grad[:,:,:,0],torch.zeros_like(logits.grad[:,:,:,0]))
    assert masked_loss(logits,torch.full_like(target,255)) == 0
    assert counts_metrics(torch.zeros(2,2))['mIoU'] is None


def test_checkpoint_training_evaluation_and_leakage(cfg,tmp_path):
    import numpy as np
    cfg['mask_colors'] = None
    records = []
    for split in ('train','val','test'):
        image = tmp_path/f'{split}.npy'
        mask = tmp_path/f'{split}_mask.npy'
        np.save(image,np.full((3,35,37),128,dtype=np.uint8))
        target = np.ones((35,37),dtype=np.int64)
        target[:,18:] = 2
        target[:2] = 0
        np.save(mask,target)
        records.append(dict(id=split,split=split,image=str(image),mask=str(mask)))
    run = tmp_path/'run'
    train_specialist(cfg,records,run,'cpu',1)
    report = evaluate_specialist(cfg,records,run,'val','cpu',run/'eval.json')
    assert report['valid_pixels'] == 33*37
    assert sum(map(sum,report['specialist']['confusion'])) == 33*37
    assert report['image_ids'] == ['val']
    changed = copy.deepcopy(cfg)
    changed['specialist']['levels'] = 'deep'
    assert signature(cfg) != signature(changed)
    with pytest.raises(ValueError,match='Incompatible'):
        evaluate_specialist(changed,records,run,'val','cpu',run/'bad.json')
    overlap = [dict(records[0],split='test')]
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_specialist(cfg,overlap,run,'test','cpu',run/'bad.json')
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_specialist(cfg,[dict(records[1],split='test')],run,'test','cpu',run/'bad.json')


def test_binary_baseline_does_not_hide_six_class_errors(cfg,tmp_path):
    import numpy as np
    from selective_seg.specialist import evaluate_model
    class AlwaysAgriculture(torch.nn.Module):
        feature_spec = []
        def forward(self,x):
            logits = x.new_zeros((len(x),2,4,4))
            logits[:,0] = 10
            return logits
    class AlwaysUrban:
        def overview(self,image):
            p = torch.zeros(1,6,*image.shape[-2:])
            p[:,0],p[:,1],p[:,2] = .95,.04,.01
            return p,None
    image,mask = tmp_path/'x.npy',tmp_path/'y.npy'
    np.save(image,np.zeros((3,35,37),dtype=np.uint8))
    np.save(mask,np.ones((35,37),dtype=np.int64))
    cfg['mask_colors'] = None
    report = evaluate_model(AlwaysAgriculture(),cfg,[dict(id='x',image=str(image),mask=str(mask))],
                            torch.device('cpu'),AlwaysUrban())
    assert report['coarse_conditional_binary']['accuracy'] == 1
    assert report['coarse_six_class_on_AR_pixels']['accuracy'] == 0
    assert report['diagnostic_correctable_pixels'] == 35*37
    assert report['coarse_confident_errors'] == report['specialist_correct_on_confident_errors'] == 35*37
    assert report['diagnostic_damage_pixels'] == 0
