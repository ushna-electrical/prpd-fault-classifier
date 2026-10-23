"""N-way, N-shot evaluation: build one prototype embedding per class (the
mean embedding of N support examples drawn from the TRAIN split), then
classify each evaluation image by nearest prototype (L2 distance in
embedding space). This generalizes the paper's K-way ONE-shot support set
(exactly 1 example per class) to N-shot by averaging N support embeddings
per class -- standard, well-precedented (their own related-work citations
include prototypical-network-style few-shot methods), and more robust than
a single support example per class given we have more than one available for
every class.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from prpd_classifier.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1_per_class


def _load_batch(rows: list[dict], transform, device: str) -> torch.Tensor:
    imgs = [transform(Image.open(r["filepath"]).convert("RGB")) for r in rows]
    return torch.stack(imgs).to(device)


def build_class_prototypes(
    model,
    train_rows: list[dict],
    transform,
    class_list: list[str],
    n_shot: int,
    device: str,
    seed: int = 0,
    batch_size: int = 32,
) -> torch.Tensor:
    """Returns a (num_classes, embedding_dim) tensor, one prototype per class
    in class_list order. Raises if a class has zero support rows available.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_class[row["fault_type"]].append(row)

    model.eval()
    prototypes = []
    with torch.no_grad():
        for class_name in class_list:
            pool = by_class.get(class_name, [])
            if not pool:
                raise ValueError(f"No support examples available for class {class_name!r}.")
            chosen = rng.sample(pool, min(n_shot, len(pool)))

            embeddings = []
            for i in range(0, len(chosen), batch_size):
                batch = _load_batch(chosen[i : i + batch_size], transform, device)
                embeddings.append(model.embed(batch))
            embeddings = torch.cat(embeddings, dim=0)
            prototypes.append(embeddings.mean(dim=0))

    return torch.stack(prototypes, dim=0)


def classify_by_nearest_prototype(
    model,
    rows: list[dict],
    transform,
    class_list: list[str],
    prototypes: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(class_list)}
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            batch = _load_batch(batch_rows, transform, device)
            embeddings = model.embed(batch)  # (B, D)
            # pairwise L2 distance to every prototype: (B, K)
            dists = torch.cdist(embeddings, prototypes, p=2)
            preds = dists.argmin(dim=1).tolist()
            y_pred.extend(preds)
            y_true.extend(class_to_idx[r["fault_type"]] for r in batch_rows)

    cm = confusion_matrix(y_true, y_pred, len(class_list))
    return {
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "confusion_matrix": cm,
        "per_class": {name: r for name, r in zip(class_list, precision_recall_f1_per_class(cm))},
        "y_true": y_true,
        "y_pred": y_pred,
    }
