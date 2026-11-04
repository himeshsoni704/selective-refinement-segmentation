import copy
import json
from pathlib import Path
import pytest
import torch
from selective_seg.data import load_config
from selective_seg.geometry import tiles, crop_context, center_prediction, tile_descriptor
from selective_seg.models import Segmenter, OutcomeSelector
from selective_seg.objectives import outcome_targets, utility, selector_loss, segmentation_loss
from selective_seg.routing import choose_actions
from selective_seg.engine import Pipeline
from selective_seg.data import read_manifest, read_sample
from selective_seg.training import flatten_cache
import numpy as np
from PIL import Image


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return load_config(Path(__file__).parents[1]/"configs/smoke.json")


def test_exact_corrections_damage_and_confusion(cfg):
    y = torch.tensor([[0,0,1,1,255]])
    c = torch.tensor([[0,1,0,1,0]])
    low = torch.tensor([[0,0,0,1,1]])
    high = torch.tensor([[1,0,1,1,1]])
    t = outcome_targets(c,low,high,y,cfg)
    assert t["valid_pixels"] == 4
    assert torch.allclose(t["transitions"][0],torch.tensor([.5,0,.25,.25]))
    assert torch.allclose(t["transitions"][1],torch.tensor([.25,.25,.5,0]))
    assert t["counts"][1].tolist() == [[2,0],[1,1]]
    assert outcome_targets(c,low,high,torch.full_like(y,255),cfg) is None
    cfg["utility"]["confusion_weight"] = 0
    assert torch.allclose(utility(t["transitions"],t["confusion"],cfg),torch.tensor([.25,0.]))


def test_probability_thresholds_and_veto(cfg):
    cfg["routing"].update(T1=.75,T2=.4,utility_veto=False)
    q = torch.tensor([.1,.4,.6,.75,.9])
    p = {"benefit_logits":torch.stack((torch.zeros_like(q),torch.zeros_like(q),torch.logit(q)),1),
         "transitions":torch.tensor([.6,0,.3,.1]).expand(5,2,4).clone(),
         "confusion":torch.full((5,3,2,2),.25)}
    action,_,_ = choose_actions(p,cfg)
    assert action.tolist() == [0,1,1,2,2]
    cfg["routing"]["utility_veto"] = True
    p["transitions"][4,1] = torch.tensor([.6,.3,0,.1])
    action,_,_ = choose_actions(p,cfg)
    assert action[-1] == 1  # high harmful -> useful low
    cfg["routing"]["max_fraction"] = .2
    assert int((choose_actions(p,cfg)[0]>0).sum()) <= 1
    cfg["routing"].update(max_fraction=1.,max_refine_ms=0.)
    assert not choose_actions(p,cfg)[0].any()


def test_geometry_partial_tiles_and_detail(cfg):
    image = torch.rand(1,3,79,91)
    grid = tiles(79,91,48)
    assert sum(t[2]*t[3] for t in grid) == 79*91
    last = grid[-1]
    crop = crop_context(image,last,48,8)
    assert crop.shape == (1,3,64,64)
    assert torch.equal(crop[...,8:8+last[2],8:8+last[3]],image[...,48:,48:])
    prob = torch.softmax(torch.rand(1,2,79,91),1)
    feature = torch.rand(1,64,3,3)
    descriptor = tile_descriptor(image,prob,feature,last,cfg)
    assert descriptor.shape == (1,64+2+9)
    assert torch.isfinite(descriptor).all()
    z = center_prediction(torch.rand(1,2,32,32),last,cfg)
    assert z.shape[-2:] == (31,43)


def test_spectral_class_patch_projection_gradients(cfg):
    cfg["bands"] = 8
    cfg["spectral"]["enabled"] = True
    model = Segmenter(cfg)
    x = torch.rand(1,8,64,64)
    truth = torch.randint(0,2,(1,64,64))
    out = model(x)
    loss = segmentation_loss(out,truth,cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.color.class_weights.grad.abs().sum() > 0
    assert model.color.class_predictor[0].weight.grad.abs().sum() > 0
    assert model.color.patch_weights[0].weight.grad.abs().sum() > 0
    a,_reg,_ = model.color(x,torch.tensor([[1.,0.]]))
    b,_reg,_ = model.color(x,torch.tensor([[0.,1.]]))
    assert not torch.allclose(a,b)


def test_selector_backward_and_consistent_outputs(cfg):
    model = OutcomeSelector(75,2)
    z = torch.randn(4,75)
    pred = model(z)
    target = {"transitions":torch.softmax(torch.randn(4,2,4),-1),
              "confusion":torch.softmax(torch.randn(4,3,4),-1).reshape(4,3,2,2)}
    loss = selector_loss(pred,target,cfg)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert torch.allclose(pred["confusion"].sum((-2,-1)),torch.ones(4,3))
    assert ((pred["corrected"]+pred["damaged"])<=1).all()


def test_skip_preserves_coarse_and_inference_needs_no_mask(cfg):
    models = [Segmenter(cfg,context=(i>0)).eval() for i in range(3)]
    selector = OutcomeSelector(75,2).eval()
    cfg["routing"]["max_fraction"] = 0
    pipe = Pipeline(cfg,*models,selector=selector)
    x = torch.rand(3,79,91)
    expected = pipe.overview(x)[0].argmax(1)[0]
    result,details = pipe.predict(x)
    assert torch.equal(result,expected)
    assert len(details["tiles"]) == 4


def test_actual_segformer_b0_forward_backward(cfg):
    cfg["backbone"] = "segformer_b0"
    model = Segmenter(cfg,context=True)
    x = torch.rand(1,3,64,64)
    pc = torch.softmax(torch.randn(1,2,64,64),1)
    out = model(x,pc)
    loss = segmentation_loss(out,torch.randint(0,2,(1,64,64)),cfg)
    loss.backward()
    assert out["features"].shape[1] == 256
    assert out["logits"].shape == (1,2,64,64)
    assert model.classifier.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())


def test_rgb_labels_and_cross_split_reuse_rejected(cfg,tmp_path):
    Image.fromarray(np.zeros((5,7,3),dtype=np.uint8)).save(tmp_path/"image.png")
    label = np.zeros((5,7),dtype=np.uint8)
    label[2,3] = 255
    Image.fromarray(label).save(tmp_path/"mask.png")
    cfg["mask_values"] = {"0":0,"255":1}
    record = {"id":"a","image":str(tmp_path/"image.png"),"mask":str(tmp_path/"mask.png"),"split":"train"}
    _,mask = read_sample(record,cfg)
    assert mask[2,3] == 1  # Inria 255 is a BUILDING, not an ignored pixel.
    duplicate = dict(record,id="b",split="test")
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps([record,duplicate]))
    with pytest.raises(ValueError,match="Same image"):
        read_manifest(path)


def test_ignored_tiles_excluded_from_selector_training(cfg):
    cache = {"images":[{"valid_pixels":torch.tensor([20,0]),"z":torch.rand(2,75),
                        "transitions":torch.rand(2,2,4),"confusion":torch.rand(2,3,2,2)}]}
    assert len(flatten_cache(cache)["z"]) == 1


def test_both_refinement_actions_keep_original_dimensions(cfg):
    models = [Segmenter(cfg,context=(i>0)).eval() for i in range(3)]
    selector = OutcomeSelector(75,2).eval()
    with torch.no_grad():
        selector.benefit.weight.zero_()
        selector.benefit.bias.fill_(20.)
    cfg["routing"]["utility_veto"] = False
    pipe = Pipeline(cfg,*models,selector=selector)
    image = torch.rand(3,79,91)
    mask, details = pipe.predict(image)
    assert mask.shape == (79,91)
    assert all(t["action"] == "high" for t in details["tiles"])
    with torch.no_grad():
        selector.benefit.bias.fill_(.4)  # sigmoid(.4) lies between default T2 and T1.
    mask, details = pipe.predict(image)
    assert mask.shape == (79,91)
    assert all(t["action"] == "low" for t in details["tiles"])
