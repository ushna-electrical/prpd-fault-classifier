"""Calibration metrics.

A model is calibrated if, among the predictions it makes with confidence ~p,
a fraction ~p are correct. Accuracy alone says nothing about this: a model can
be 85% accurate while claiming 99% confidence on everything.

All functions take `probs` (n_samples, n_classes) and integer `labels`.
"""

from __future__ import annotations

import numpy as np


def confidence_and_correct(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 confidence and whether that top-1 prediction was right."""
    pred = probs.argmax(axis=1)
    return probs.max(axis=1), (pred == labels).astype(float)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    """Equal-width confidence bins with their accuracy, mean confidence and count.
    The gap (accuracy - confidence) is what a reliability diagram plots."""
    conf, correct = confidence_and_correct(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "count": n,
            "accuracy": float(correct[in_bin].mean()) if n else float("nan"),
            "confidence": float(conf[in_bin].mean()) if n else float("nan"),
            "gap": float(correct[in_bin].mean() - conf[in_bin].mean()) if n else float("nan"),
        })
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average |accuracy - confidence| over bins, weighted by bin count.
    0 = perfectly calibrated."""
    bins = reliability_bins(probs, labels, n_bins)
    n = len(labels)
    return float(sum(b["count"] / n * abs(b["gap"]) for b in bins if b["count"]))


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """MCE: the worst bin. Matters when a single confidence band is badly wrong."""
    bins = reliability_bins(probs, labels, n_bins)
    gaps = [abs(b["gap"]) for b in bins if b["count"]]
    return float(max(gaps)) if gaps else 0.0


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error against the one-hot target).
    Lower is better; unlike ECE it also rewards being confidently right."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())


def overconfidence(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean confidence minus accuracy. Positive = the model overstates itself."""
    conf, correct = confidence_and_correct(probs, labels)
    return float(conf.mean() - correct.mean())


def summary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    conf, correct = confidence_and_correct(probs, labels)
    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfidence": overconfidence(probs, labels),
        "ECE": expected_calibration_error(probs, labels, n_bins),
        "MCE": maximum_calibration_error(probs, labels, n_bins),
        "Brier": brier_score(probs, labels),
        "NLL": negative_log_likelihood(probs, labels),
    }
