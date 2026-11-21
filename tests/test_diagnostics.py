import copy
import json

import pytest
import torch

from selective_seg.diagnostics import greedy_miou, specialist_report
from selective_seg.hybrid_metrics import Metrics
from selective_seg.precision import specialist_precision
from selective_seg.routing_tradeoffs import allocate


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_metrics_ignore_and_normalization():
    m = Metrics(2)
    p = torch.tensor([[[.8, .3]], [[.2, .7]]])
    truth = torch.tensor([[0, 1]])
    m.add(p, truth)
    before = copy.deepcopy(m.report())
    assert before['per_class_IoU'] == [1., 1.]
    assert before['NLL'] == pytest.approx(-torch.tensor([.8, .7]).log().mean().item())
    m.add(p, torch.full_like(truth, 255))
    assert m.report() == before
    with pytest.raises(ValueError, match='normalized'):
        m.add(p*.5, truth)
    assert m.report() == before
    # Invalid ignored outputs still fail, before any accumulated counts change.
    broken = p.clone()
    broken[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        m.add(broken, torch.full_like(truth, 255))
    assert m.report() == before


def test_precision_is_explicit_on_cpu():
    assert specialist_precision(torch.device('cpu'), {}) == 'fp32'
    with pytest.raises(ValueError, match='requires CUDA'):
        specialist_precision(torch.device('cpu'), {'hybrid': {'precision': 'fp16'}})


def test_greedy_actual_confusion_and_budget():
    # Two tiles; each high action fixes the one error present in its own tile.
    counts = torch.tensor([[[[8, 2], [0, 0]], [[8, 2], [0, 0]], [[10, 0], [0, 0]]],
                           [[[0, 0], [2, 8]], [[0, 0], [2, 8]], [[0, 0], [0, 10]]]])
    assert not greedy_miou(counts, 0).any()
    selected = greedy_miou(counts, 2)
    assert selected.tolist() == [2, 2]
    assert counts[torch.arange(2), selected].sum(0).tolist() == [[10, 0], [0, 10]]
    assert not greedy_miou(counts, 2, torch.ones(2, 2), 0.).any()


def test_class_cap_and_compute_feasibility():
    row = {'corrected_by_class': torch.tensor([[[0, 8], [0, 6]], [[0, 4], [0, 3]]]),
           'damaged_by_class': torch.tensor([[[4, 0], [0, 0]], [[0, 0], [0, 0]]]),
           'costs': torch.tensor([[1., 2.], [1., 2.]]), 'supervised': torch.tensor([True, True]),
           'coarse_counts': torch.diag(torch.tensor([100, 100]))}
    actions, spent = allocate(row, 1., damage_rate=0.)
    assert actions.tolist() == [2, 1]
    assert spent <= 4
    actions, spent = allocate(row, .5, damage_rate=0.)
    assert int((actions > 0).sum()) <= 1
    assert spent <= 2


def test_checkpoint_numerical_failure_has_report(tmp_path):
    checkpoint = tmp_path/'bad.pt'
    torch.save({'config': {}, 'model': {'weight': torch.tensor([float('inf')])}}, checkpoint)
    output = tmp_path/'report.json'
    with pytest.raises(ValueError, match='Nonfinite checkpoint'):
        specialist_report(checkpoint, None, output, torch.device('cpu'))
    assert json.loads(output.read_text())['status'] == 'invalid_checkpoint'
