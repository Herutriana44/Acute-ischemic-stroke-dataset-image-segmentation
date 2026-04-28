"""Metrik evaluasi segmentasi biner (per-slice dan agregat piksel)."""

from __future__ import annotations

import torch


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)


@torch.no_grad()
def dice_per_slice(pred_prob: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    Dice per sampel (slice) shape (B,1,H,W), nilai dalam [0,1].
    Jika GT kosong: Dice=1 jika pred kosong, else 0.
    """
    pred = (pred_prob > thr).float()
    gt = (mask > 0.5).float()
    b = pred.shape[0]
    eps = 1e-6
    out = pred.new_zeros(b)
    pf = _flatten_batch(pred)
    gf = _flatten_batch(gt)
    for i in range(b):
        p, g = pf[i], gf[i]
        inter = (p * g).sum()
        gsum, psum = g.sum(), p.sum()
        if gsum <= eps:
            out[i] = 1.0 if psum <= eps else 0.0
        else:
            out[i] = (2.0 * inter + eps) / (psum + gsum + eps)
    return out


@torch.no_grad()
def iou_per_slice(pred_prob: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    pred = (pred_prob > thr).float()
    gt = (mask > 0.5).float()
    b = pred.shape[0]
    eps = 1e-6
    out = pred.new_zeros(b)
    pf = _flatten_batch(pred)
    gf = _flatten_batch(gt)
    for i in range(b):
        p, g = pf[i], gf[i]
        inter = (p * g).sum()
        union = p.sum() + g.sum() - inter
        if union <= eps:
            out[i] = 1.0 if inter <= eps else 0.0
        else:
            out[i] = (inter + eps) / (union + eps)
    return out


@torch.no_grad()
def precision_recall_per_slice(
    pred_prob: torch.Tensor, mask: torch.Tensor, thr: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = (pred_prob > thr).float()
    gt = (mask > 0.5).float()
    b = pred.shape[0]
    eps = 1e-6
    prec = pred.new_zeros(b)
    rec = pred.new_zeros(b)
    pf = _flatten_batch(pred)
    gf = _flatten_batch(gt)
    for i in range(b):
        p, g = pf[i], gf[i]
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        prec[i] = (tp + eps) / (tp + fp + eps)
        rec[i] = (tp + eps) / (tp + fn + eps)
    return prec, rec


@torch.no_grad()
def pixel_accuracy(pred_prob: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> float:
    pred = (pred_prob > thr).float()
    gt = (mask > 0.5).float()
    return float((pred == gt).float().mean())


@torch.no_grad()
def global_dice_from_confusion(pred_prob: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> float:
    """Satu Dice dari total TP, FP, FN di seluruh batch (micro / voxel-level)."""
    pred = (pred_prob > thr).float()
    gt = (mask > 0.5).float()
    inter = (pred * gt).sum()
    eps = 1e-6
    return float((2.0 * inter + eps) / (pred.sum() + gt.sum() + eps))


@torch.no_grad()
def global_specificity(pred_prob: torch.Tensor, mask: torch.Tensor, thr: float = 0.5) -> float:
    pred = (pred_prob > thr).float()
    gt = (mask > 0.5).float()
    tn = ((1 - pred) * (1 - gt)).sum()
    fp = (pred * (1 - gt)).sum()
    eps = 1e-6
    return float((tn + eps) / (tn + fp + eps))
