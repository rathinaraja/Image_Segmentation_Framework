"""
utils/metrics.py
----------------
Segmentation evaluation metrics: IoU, Dice, Pixel Accuracy.
All functions operate on batched tensors.
"""

import torch


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correctly classified pixels."""
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / total


def mean_iou(preds: torch.Tensor, targets: torch.Tensor, n_classes: int, eps: float = 1e-6) -> float:
    """
    Mean Intersection-over-Union across all classes.
    Ignores classes absent in both prediction and target (avoids NaN inflation).
    """
    ious = []
    for cls in range(n_classes):
        pred_mask   = (preds == cls)
        target_mask = (targets == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union        = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue   # class not present — skip
        ious.append(intersection / (union + eps))
    return float(sum(ious) / len(ious)) if ious else 0.0


def dice_score(preds: torch.Tensor, targets: torch.Tensor, n_classes: int, eps: float = 1e-6) -> float:
    """Mean Dice coefficient across all classes present."""
    scores = []
    for cls in range(n_classes):
        pred_mask   = (preds == cls).float()
        target_mask = (targets == cls).float()
        num   = 2 * (pred_mask * target_mask).sum().item()
        denom = pred_mask.sum().item() + target_mask.sum().item()
        if denom == 0:
            continue
        scores.append(num / (denom + eps))
    return float(sum(scores) / len(scores)) if scores else 0.0


class MetricTracker:
    """Accumulates metric values over an epoch and reports averages."""

    def __init__(self):
        self._sums   = {}
        self._counts = {}

    def update(self, name: str, value: float, n: int = 1):
        self._sums[name]   = self._sums.get(name, 0.0) + value * n
        self._counts[name] = self._counts.get(name, 0)  + n

    def avg(self, name: str) -> float:
        return self._sums[name] / self._counts[name] if self._counts.get(name) else 0.0

    def summary(self) -> dict:
        return {k: self.avg(k) for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()
