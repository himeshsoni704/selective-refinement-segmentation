import copy
from pathlib import Path
import pytest
import torch
from selective_seg.data import load_config
from selective_seg.models import Segmenter, OutcomeSelector, AcceptanceGate, DetailScout
from selective_seg.engine import Pipeline, config_signature
from selective_seg.geometry import tile_descriptor
from selective_seg.objectives import metric_gains, tensor_miou, acceptance_loss, selector_loss
from selective_seg.routing import choose_actions
from selective_seg.research import descriptor_dim, signature
from selective_seg.training import load_cache
from selective_seg.benchmark import summarize_latency


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return load_config(Path(__file__).parents[1]/'configs/smoke.json')


def pred(n=1):
    return {'benefit_logits':torch.ones(n,3)*10,
            'transitions':torch.tensor([.6,0,.3,.1]).expand(n,2,4).clone(),
            'confusion':torch.full((n,3,2,2),.25)}


def test_direct_feasible_low_fallback(cfg):
    cfg['routing'].update(policy='direct',action_ms=[2.,10.],max_refine_ms=3.)
    actions,_,_ = choose_actions(pred(),cfg)
    assert actions.tolist() == [1]
    cfg['routing']['max_refine_ms'] = 0
    assert not choose_actions(pred(),cfg)[0].any()


def test_metric_gains_match_count_replacement():
    counts = torch.tensor([[[[8.,1.],[1.,0.]],[[8.,1.],[0.,1.]],[[7.,2.],[0.,1.]]],
                           [[[0.,1.],[1.,8.]],[[1.,0.],[1.,8.]],[[1.,0.],[0.,9.]]]])
    gain = metric_gains(counts/10,torch.tensor([10.,10.]))
    aggregate = counts[:,0].sum(0)
    for i in range(2):
        for a in range(2):
            assert torch.allclose(gain[i,a],tensor_miou(aggregate-counts[i,0]+counts[i,a+1])-tensor_miou(aggregate))
    assert torch.isfinite(metric_gains(counts/10,torch.zeros(2))).all()


def test_gate_only_learns_decisive_pixels():
    coarse = torch.tensor([[[[.9,.9,.9,.9]],[[.1,.1,.1,.1]]]])
    refined = coarse.flip(1)
    truth = torch.tensor([[[0,1,255,2]]])
    logits = torch.zeros(1,1,4,requires_grad=True)
    loss = acceptance_loss(logits,coarse,refined,truth)
    loss.backward()
    assert logits.grad[0,0,0] > 0
    assert logits.grad[0,0,1] < 0
    assert logits.grad[0,0,2] == logits.grad[0,0,3] == 0


def test_descriptor_and_shared_marginals(cfg):
    cfg['research'] = {'descriptor_grid':2,'scout':True}
    image = torch.rand(1,3,48,48)
    pc = torch.rand(1,2,48,48).softmax(1)
    z = tile_descriptor(image,pc,torch.rand(1,64,3,3),(0,0,48,48),cfg,torch.rand(1,3,24,24))
    assert z.shape[-1] == descriptor_dim(64,cfg)
    model = OutcomeSelector(z.shape[-1],2,True)
    cm = model(z)['confusion']
    assert torch.allclose(cm[:,0].sum(-1),cm[:,2].sum(-1),atol=1e-6)


def test_baselines_without_selector_and_acceptance(cfg):
    pipe = Pipeline(cfg,*[Segmenter(cfg,i>0).eval() for i in range(3)])
    image = torch.rand(3,79,91)
    coarse = pipe.overview(image)[0].argmax(1)[0]
    out,d = pipe.predict(image,'coarse',False)
    assert torch.equal(out,coarse) and d['tiles'] == []
    assert d['refinement_calls'] == {'low':0,'high':0}
    for mode in ('low','high','random','entropy'):
        out,d = pipe.predict(image,mode)
        assert out.shape == coarse.shape
    pipe.acceptance = AcceptanceGate(2)
    with torch.no_grad():
        for p in pipe.acceptance.parameters():
            p.zero_()
        pipe.acceptance.net[-1].bias.fill_(-100)
    out,_ = pipe.predict(image,'high')
    assert torch.equal(out,coarse)


def test_escalation_charges_both_passes(cfg):
    class Positive(torch.nn.Module):
        def forward(self,z):
            return pred(len(z))
    cfg['routing'].update(T1=.99999,T2=.1,action_ms=[2.,5.],max_refine_ms=7.)
    selector = OutcomeSelector(75,2).eval()
    with torch.no_grad():
        selector.benefit.weight.zero_()
        selector.benefit.bias.fill_(1.)
    cfg['routing']['utility_veto'] = False
    pipe = Pipeline(cfg,*[Segmenter(cfg,i>0).eval() for i in range(3)],selector=selector)
    pipe.escalation = Positive()
    _,d = pipe.predict(torch.rand(3,48,48))
    assert d['escalations'] == 1
    assert d['refinement_calls'] == {'low':1,'high':1}
    assert d['estimated_refine_ms'] == 7
    cfg['routing']['max_refine_ms'] = 6
    _,d = pipe.predict(torch.rand(3,48,48))
    assert d['escalations'] == 0
    assert d['estimated_refine_ms'] == 2


def test_stale_cache_rejected_and_segmenters_reusable(cfg,tmp_path):
    old = config_signature(cfg)
    cfg['research'] = {'scout':True}
    assert config_signature(cfg) == old
    assert signature(cfg) != signature({})
    path = tmp_path/'cache.pt'
    torch.save({'split':'selector','signature':old},path)
    with pytest.raises(ValueError,match='schema'):
        load_cache(path,cfg,'selector')


def test_metric_selector_gradients_and_scout(cfg):
    cfg['utility']['mode'] = 'miou'
    model = OutcomeSelector(75,2)
    target = pred(3)
    target['valid_pixels'] = torch.tensor([4,8,16])
    loss = selector_loss(model(torch.rand(3,75)),target,cfg)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    scout = DetailScout(3)
    scout(torch.rand(1,3,31,27)).mean().backward()
    assert all(p.grad is not None for p in scout.parameters())


def test_latency_aggregation():
    stats = summarize_latency([[1,2,3],[10,20,30]])
    assert stats['mean_image_latency_ms'] == 11
    assert stats['median_image_latency_ms'] == 11


def test_oracle_diagnostic_is_validation_only(cfg):
    from selective_seg.evaluation import analyze_cache
    with pytest.raises(ValueError,match='validation-only'):
        analyze_cache(cfg,{'split':'test'})


def test_direct_rejects_invalid_costs_and_zero_support(cfg):
    cfg['routing']['policy'] = 'direct'
    with pytest.raises(ValueError,match='positive'):
        choose_actions(pred(),cfg,costs=[0.,2.])
    assert not choose_actions(pred(),cfg,support=torch.zeros(1))[0].any()
