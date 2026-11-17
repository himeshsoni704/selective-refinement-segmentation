"""Streaming segmentation calibration and confidence-based risk/coverage."""
import torch

from .objectives import confusion_matrix
from .specialist import counts_metrics


class Metrics:
    def __init__(self, classes, bins=15, ignore=255):
        self.classes, self.bins, self.ignore = classes, bins, ignore
        self.cm = torch.zeros(classes, classes, dtype=torch.long)
        self.bin_count = torch.zeros(bins, dtype=torch.double)
        self.bin_correct = self.bin_count.clone()
        self.bin_confidence = self.bin_count.clone()
        self.nll = self.brier = 0.

    def add(self, probability, truth):
        probability = probability.cpu()
        pred = probability.argmax(0)
        cm = confusion_matrix(pred, truth, self.classes, self.ignore)
        self.cm += cm
        valid = (truth >= 0) & (truth < self.classes) & (truth != self.ignore)
        p = probability[:, valid]
        y = truth[valid]
        correct = (pred[valid] == y).double()
        confidence = p.max(0).values
        bins = (confidence * self.bins).long().clamp_max(self.bins - 1)
        self.bin_count += torch.bincount(bins, minlength=self.bins)
        self.bin_correct.scatter_add_(0, bins, correct)
        self.bin_confidence.scatter_add_(0, bins, confidence.double())
        true = p.gather(0, y[None])[0]
        self.nll -= float(true.clamp_min(1e-12).log().sum(dtype=torch.double))
        self.brier += float((p.square().sum(0) - 2 * true + 1).sum(dtype=torch.double))
        return cm.tolist()

    def report(self):
        report = counts_metrics(self.cm)
        n = float(self.bin_count.sum())
        retained = self.bin_count.flip(0).cumsum(0)
        correct = self.bin_correct.flip(0).cumsum(0)
        report.update(NLL=self.nll / n if n else None, Brier=self.brier / n if n else None,
                      ECE=float((self.bin_correct-self.bin_confidence).abs().sum()) / n if n else None,
                      selective_risk_coverage=[
                          {"coverage": k / n, "risk": 1 - c / k}
                          for k, c in zip(retained.tolist(), correct.tolist()) if k],
                      calibration_bins=self.bins)
        return report


def paired_bootstrap(rows, baseline, repetitions, seed):
    current_ids = [row["id"] for row in rows]
    reference = {row["id"]: row["confusion"] for row in baseline}
    if len(reference) != len(baseline) or len(set(current_ids)) != len(rows) or set(current_ids) != set(reference):
        raise ValueError("Paired reports must have the same unique image IDs")
    current = torch.tensor([row["confusion"] for row in rows])
    previous = torch.tensor([reference[key] for key in current_ids])
    generator = torch.Generator().manual_seed(seed)
    from .objectives import tensor_miou
    indices = torch.randint(len(rows), (repetitions, len(rows)), generator=generator)
    values = tensor_miou(current[indices].sum(1).double()) - tensor_miou(previous[indices].sum(1).double())
    return {"repetitions": repetitions, "delta_mIoU_CI95": torch.quantile(
        values, values.new_tensor([.025, .975])).tolist(),
        "unit": "paired images"}
