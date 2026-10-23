"""Training loop: cross-entropy + AdamW + cosine LR schedule, early stopping on
validation macro-F1. Shared across all three backbones -- only the model
differs.
"""

from __future__ import annotations

import copy
import time

import torch
from torch.utils.data import DataLoader

from .metrics import accuracy, confusion_matrix, macro_f1


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: str, num_classes: int) -> dict:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    all_true, all_pred = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            all_true.extend(labels.tolist())
            all_pred.extend(logits.argmax(dim=1).tolist())

    cm = confusion_matrix(all_true, all_pred, num_classes)
    return {
        "loss": total_loss / len(all_true) if all_true else 0.0,
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "confusion_matrix": cm,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def train_one_backbone(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_classes: int,
    epochs: int,
    lr: float,
    patience: int,
    class_weights: torch.Tensor | None = None,
    log_fn=print,
):
    model.to(device)
    # Only pass trainable params -- some may be frozen (see models.freeze_backbone_features).
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Class weights (if given) apply only to the training objective -- val/test
    # loss and metrics stay unweighted so they remain directly comparable to
    # the unweighted Stage 1 baseline.
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        scheduler.step()
        train_loss = running_loss / n_seen if n_seen else 0.0

        val_metrics = evaluate_loader(model, val_loader, device, num_classes)
        elapsed = time.time() - t0

        log_fn(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  ({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
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
