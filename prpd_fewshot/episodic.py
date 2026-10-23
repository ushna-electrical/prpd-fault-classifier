"""Episode sampling and loss for Prototypical Network-style few-shot training
(Snell et al. 2017). Each episode draws n_shot support + n_query query images
per class; the model is trained to classify queries by nearest support
prototype, which is exactly the objective evaluate.py uses at test time --
unlike the pairwise same/different verification loss in dataset.py/train.py,
which only pulls/pushes individual pairs and never directly optimizes for a
class's mean embedding landing in its own decision region.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def _draw(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    """k rows without replacement when the pool allows it, else with replacement."""
    if len(pool) >= k:
        return rng.sample(pool, k)
    return [rng.choice(pool) for _ in range(k)]


def sample_episode(
    by_class: dict[str, list[dict]],
    class_list: list[str],
    n_shot: int,
    n_query: int,
    rng: random.Random,
    session_disjoint: bool = False,
) -> tuple[list[dict], list[dict], list[int]]:
    """Returns (support_rows, query_rows, query_labels).

    support_rows is class-major: the first n_shot rows belong to class_list[0],
    the next n_shot to class_list[1], etc. query_labels are class indices into
    class_list, aligned with query_rows.

    Classes with fewer than n_shot + n_query available rows are sampled with
    replacement rather than raising, so every class can still participate in
    every episode.

    session_disjoint=True mirrors the test condition: all of a class's query
    rows come from ONE randomly chosen recording session, and its support rows
    come only from that class's OTHER sessions. Without this, support and query
    usually share a session, which rewards matching a recording's particular
    activity state rather than recognising the fault. Requires a "session_id"
    field; a class with a single session falls back to plain random sampling.
    """
    support_rows: list[dict] = []
    query_rows: list[dict] = []
    query_labels: list[int] = []

    for class_idx, class_name in enumerate(class_list):
        pool = by_class.get(class_name, [])
        if not pool:
            raise ValueError(f"No training examples available for class {class_name!r}.")

        sessions: dict[str, list[dict]] = {}
        if session_disjoint:
            for row in pool:
                sessions.setdefault(row["session_id"], []).append(row)

        if len(sessions) >= 2:
            query_session = rng.choice(sorted(sessions))
            support_pool = [r for sid, rows in sessions.items() if sid != query_session for r in rows]
            support_rows.extend(_draw(support_pool, n_shot, rng))
            query_rows.extend(_draw(sessions[query_session], n_query, rng))
        else:
            chosen = _draw(pool, n_shot + n_query, rng)
            support_rows.extend(chosen[:n_shot])
            query_rows.extend(chosen[n_shot:])
        query_labels.extend([class_idx] * n_query)

    return support_rows, query_rows, query_labels


def prototypical_loss(
    support_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: list[int],
    n_way: int,
    n_shot: int,
) -> tuple[torch.Tensor, float]:
    """support_embeddings is (n_way * n_shot, D), class-major (matching
    sample_episode's ordering). Returns (loss, query_accuracy)."""
    device = support_embeddings.device
    embedding_dim = support_embeddings.shape[1]
    prototypes = support_embeddings.view(n_way, n_shot, embedding_dim).mean(dim=1)  # (n_way, D)

    sq_dists = torch.cdist(query_embeddings, prototypes, p=2) ** 2  # (n_query_total, n_way)
    logits = -sq_dists  # smaller distance -> larger logit
    labels = torch.tensor(query_labels, device=device, dtype=torch.long)

    loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy
