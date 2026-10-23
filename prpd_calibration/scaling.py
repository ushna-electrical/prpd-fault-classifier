"""Post-hoc calibrators, fitted on the VALIDATION split and applied to test.

Temperature scaling (Guo et al. 2017) divides the logits by a single learned
scalar T before the softmax. T > 1 softens overconfident predictions. It cannot
change which class is the argmax, so accuracy and F1 are untouched -- only the
confidence values move. That property is why it is the standard choice.

Vector scaling learns one scale and bias per class, so it can fix a model that
is overconfident on some faults and underconfident on others, at the cost of
being able to change predictions (and of more parameters to fit on a small
validation set).
"""

from __future__ import annotations

import numpy as np


def to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Work in log-probability space so ensembles (which give probabilities,
    not logits) can be calibrated with the same code."""
    return np.log(np.clip(probs, eps, 1.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    p = softmax(logits)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


class TemperatureScaler:
    """One parameter, fitted by minimising validation NLL. Never changes the
    predicted class."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, probs: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0,
            iters: int = 60, objective: str = "nll") -> "TemperatureScaler":
        """objective="nll" is the textbook choice, but NLL is dominated by the
        many high-confidence samples; on a model that is already well calibrated
        there it can drag the good region down. objective="ece" targets the
        calibration gap itself, which is what we actually report."""
        logits = to_logits(probs)
        if objective == "ece":
            from .metrics import expected_calibration_error
            grid = np.geomspace(lo, hi, 400)
            scores = [expected_calibration_error(softmax(logits / t), labels) for t in grid]
            self.temperature = float(grid[int(np.argmin(scores))])
            return self

        gr = (np.sqrt(5) - 1) / 2  # golden-section search on a smooth 1-D objective
        a, b = lo, hi
        for _ in range(iters):
            c, d = b - gr * (b - a), a + gr * (b - a)
            if _nll(logits / c, labels) < _nll(logits / d, labels):
                b = d
            else:
                a = c
        self.temperature = float((a + b) / 2)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return softmax(to_logits(probs) / self.temperature)


class VectorScaler:
    """Per-class scale and bias, fitted by gradient descent on validation NLL.
    May change predictions, so accuracy is re-checked after fitting."""

    def __init__(self, n_classes: int):
        self.w = np.ones(n_classes)
        self.b = np.zeros(n_classes)

    def fit(self, probs: np.ndarray, labels: np.ndarray, lr: float = 0.05, steps: int = 2000) -> "VectorScaler":
        logits = to_logits(probs)
        n, k = logits.shape
        onehot = np.zeros((n, k))
        onehot[np.arange(n), labels] = 1.0
        for _ in range(steps):
            p = softmax(logits * self.w + self.b)
            diff = (p - onehot) / n
            self.w -= lr * (diff * logits).sum(axis=0)
            self.b -= lr * diff.sum(axis=0)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return softmax(to_logits(probs) * self.w + self.b)


class ConfidenceCalibrator:
    """Maps a model's top-1 confidence onto the accuracy actually observed at
    that confidence, via isotonic regression fitted on validation.

    This answers the practical question directly -- "when the system reports
    90%, how often is it right?" -- and, unlike temperature scaling, it can fix
    a model that is well calibrated in one confidence band and not in another.
    The predicted class never changes; only the reported confidence does. The
    remaining probability mass is spread over the other classes so the vector
    still sums to one.
    """

    def __init__(self, n_bins: int = 12):
        self.n_bins = n_bins
        self.x = np.array([0.0, 1.0])
        self.y = np.array([0.0, 1.0])

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "ConfidenceCalibrator":
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(float)
        # bin first (smooths a small validation set), then enforce monotonicity
        edges = np.quantile(conf, np.linspace(0, 1, self.n_bins + 1))
        edges = np.unique(edges)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        if len(xs) < 2:
            return self
        xs, ys = np.array(xs), np.array(ys)
        # pool adjacent violators (weighted by bin occupancy is unnecessary here:
        # quantile bins are near-equal size)
        v = ys.astype(float).copy()
        w = np.ones_like(v)
        i = 0
        while i < len(v) - 1:
            if v[i] > v[i + 1]:
                merged = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                v[i] = merged
                w[i] += w[i + 1]
                v = np.delete(v, i + 1)
                w = np.delete(w, i + 1)
                xs = np.delete(xs, i + 1)
                i = max(i - 1, 0)
            else:
                i += 1
        self.x, self.y = xs, np.clip(v, 0.0, 1.0)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        conf = probs.max(axis=1)
        new_conf = np.interp(conf, self.x, self.y)
        idx = np.arange(len(probs)), probs.argmax(axis=1)
        rest = 1.0 - probs[idx]

        # Lowering the top class while spreading (1 - new_conf) over the others
        # can make the runner-up the new argmax, which would change predictions
        # and therefore accuracy. Clamp so the top class stays on top: with the
        # runner-up at p2, the smallest safe confidence is p2 / (rest + p2).
        others = probs.copy()
        others[idx] = -np.inf
        p2 = others.max(axis=1)
        # p2 / (rest + p2) is the exact tie point, so a strict margin is needed:
        # landing on it leaves the top class and runner-up numerically equal and
        # argmax then picks whichever index comes first.
        floor = np.where(rest + p2 > 1e-12, p2 / np.maximum(rest + p2, 1e-12), 0.0)
        new_conf = np.minimum(np.maximum(new_conf, floor * (1 + 1e-6) + 1e-9), 1.0)

        out = probs.copy()
        scale = np.where(rest > 1e-12, (1.0 - new_conf) / np.maximum(rest, 1e-12), 0.0)
        out = out * scale[:, None]
        out[idx] = new_conf
        out = out / out.sum(axis=1, keepdims=True)

        # belt and braces: repair any row where rounding still lost the argmax
        others = out.copy()
        others[idx] = -np.inf
        lost = out[idx] <= others.max(axis=1)
        if lost.any():
            out[np.arange(len(out))[lost], probs.argmax(axis=1)[lost]] = others.max(axis=1)[lost] * (1 + 1e-6)
            out = out / out.sum(axis=1, keepdims=True)
        return out
