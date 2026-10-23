"""Training loop for the Siamese network: binary cross-entropy on balanced
same/different pairs. Early stopping and checkpoint selection are on N-shot
N-way macro-F1 (evaluated on the val split each epoch, support drawn from
train) -- NOT on pair accuracy, since pair accuracy can look good while the
actual downstream classification (what we report and compare against the
softmax baselines) stays poor.
"""

from __future__ import annotations

import copy
import random
import time

import torch
from torch.utils.data import DataLoader

from .dataset import group_rows_by_class
from .episodic import prototypical_loss, sample_episode
from .evaluate import _load_batch, build_class_prototypes, classify_by_nearest_prototype


def train_siamese(
    model: torch.nn.Module,
    pair_loader: DataLoader,
    train_rows: list[dict],
    val_rows: list[dict],
    transform,
    class_list: list[str],
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    n_shot: int,
    seed: int = 0,
    log_fn=print,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        t0 = time.time()

        for img1, img2, label in pair_loader:
            img1, img2 = img1.to(device), img2.to(device)
            label = label.to(device).float()

            optimizer.zero_grad()
            logits = model(img1, img2)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
            n_seen += img1.size(0)

        train_loss = running_loss / n_seen if n_seen else 0.0

        prototypes = build_class_prototypes(model, train_rows, transform, class_list, n_shot, device, seed=seed)
        val_metrics = classify_by_nearest_prototype(model, val_rows, transform, class_list, prototypes, device)
        elapsed = time.time() - t0

        log_fn(
            f"epoch {epoch}/{epochs}  train_pair_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_pair_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "elapsed_seconds": elapsed,
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log_fn(f"Early stopping at epoch {epoch} (no val macro-F1 improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_f1


def train_prototypical(
    model: torch.nn.Module,
    train_rows: list[dict],
    val_rows: list[dict],
    train_transform,
    eval_transform,
    class_list: list[str],
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    n_shot: int,
    n_query: int,
    episodes_per_epoch: int,
    seed: int = 0,
    session_disjoint_prob: float = 0.0,
    log_fn=print,
):
    """Prototypical Network-style training: each episode samples n_shot
    support + n_query query images per class, builds prototypes from the
    support embeddings, and backprops cross-entropy loss on the queries'
    distance-to-prototype logits. Directly optimizes the same nearest-
    prototype objective evaluate.py uses for val/test scoring, unlike
    train_siamese's pairwise verification loss.

    train_transform (augmented) is used for episode support/query images;
    eval_transform (no augmentation) is used for the per-epoch val prototype-
    building and classification, matching train_siamese's same split.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    by_class = group_rows_by_class(train_rows)
    n_way = len(class_list)

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        t0 = time.time()

        for _ in range(episodes_per_epoch):
            # all-disjoint episodes rescued fault_9 but collapsed heterogeneous fault_2; mixing keeps both signals
            session_disjoint = rng.random() < session_disjoint_prob
            support_rows, query_rows, query_labels = sample_episode(
                by_class, class_list, n_shot, n_query, rng, session_disjoint=session_disjoint
            )
            support_batch = _load_batch(support_rows, train_transform, device)
            query_batch = _load_batch(query_rows, train_transform, device)

            optimizer.zero_grad()
            support_embeddings = model.embed(support_batch)
            query_embeddings = model.embed(query_batch)
            loss, acc = prototypical_loss(support_embeddings, query_embeddings, query_labels, n_way, n_shot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc

        train_loss = running_loss / episodes_per_epoch
        train_acc = running_acc / episodes_per_epoch

        prototypes = build_class_prototypes(model, train_rows, eval_transform, class_list, n_shot, device, seed=seed)
        val_metrics = classify_by_nearest_prototype(model, val_rows, eval_transform, class_list, prototypes, device)
        elapsed = time.time() - t0

        log_fn(
            f"epoch {epoch}/{epochs}  train_episode_loss={train_loss:.4f}  train_episode_acc={train_acc:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_episode_loss": train_loss,
                "train_episode_acc": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "elapsed_seconds": elapsed,
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log_fn(f"Early stopping at epoch {epoch} (no val macro-F1 improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_f1
