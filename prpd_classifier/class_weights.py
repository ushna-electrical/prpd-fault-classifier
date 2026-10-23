"""Inverse-frequency class weights for the training loss.

This is the "cheap fix" tried before committing to GAN/few-shot data
augmentation (Stage 2). Two things were tried on this dataset:

1. Full balanced weighting (power=1.0, no cap): weight_c = N / (K * count_c).
   Result on the real run: WORSE than the unweighted baseline (test macro-F1
   0.074 vs. 0.214) -- the resulting weights spanned a ~15x range (6.28x for
   fault_1 down to 0.42x for fault_7), which was aggressive enough to
   destabilize training; early stopping triggered at epoch 6 with its best
   point at epoch 2, well before the model had a chance to settle.

2. Dampened weighting (power=0.5, i.e. sqrt of the balanced weight): pulls the
   same ratio down to ~3.9x (2.51x down to 0.65x on this dataset) -- gentler,
   still favors scarce classes, less likely to destabilize early training.
   `max_ratio` is exposed as an extra, independent safety cap on top of
   dampening (clips the highest weight to at most `max_ratio` times the
   lowest), configurable rather than hardcoded, per the plan's Stage 2 item 1.
"""

from __future__ import annotations

from collections import Counter

import torch


def compute_class_weights(
    rows: list[dict],
    class_list: list[str],
    power: float = 1.0,
    max_ratio: float | None = None,
) -> torch.Tensor:
    counts = Counter(row["fault_type"] for row in rows)
    n_total = len(rows)
    n_classes = len(class_list)

    raw_weights = []
    for class_name in class_list:
        count = counts.get(class_name, 0)
        balanced = n_total / (n_classes * count) if count > 0 else 0.0
        raw_weights.append(balanced**power if balanced > 0 else 0.0)

    if max_ratio is not None:
        nonzero = [w for w in raw_weights if w > 0]
        if nonzero:
            min_weight = min(nonzero)
            cap = min_weight * max_ratio
            raw_weights = [min(w, cap) if w > 0 else 0.0 for w in raw_weights]

    return torch.tensor(raw_weights, dtype=torch.float32)
