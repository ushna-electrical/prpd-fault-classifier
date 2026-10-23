"""Training loop following Mück section 4.4.2 / 4.5: AdamW (weight decay 1e-2),
cosine annealing from 5e-4 to 1e-6, layer-wise LR scaling, batch size 32.

Both output modes are supported:
  multilabel=False -> softmax + cross-entropy (our single-label task, directly
                      comparable with the prototypical results)
  multilabel=True  -> independent sigmoids + BCE (his setup); a sample may get
                      several labels or none.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from prpd_classifier.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1_per_class

from . import config
from .model import layerwise_param_groups


@torch.no_grad()
def evaluate(model, loader, class_list, device, multilabel: bool, threshold: float = 0.5) -> dict:
    model.eval()
    y_true, y_pred, probs, targets = [], [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        if multilabel:
            p = torch.sigmoid(logits).cpu()
            probs.append(p)
            targets.append(yb)
            y_true.extend(yb.argmax(dim=1).tolist())
        else:
            y_true.extend(yb.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())  # argmax works for both modes

    cm = confusion_matrix(y_true, y_pred, len(class_list))
    out = {
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "confusion_matrix": cm,
        "per_class": {c: r for c, r in zip(class_list, precision_recall_f1_per_class(cm))},
    }
    if multilabel:  # his own metric: independent per-label decisions at a threshold
        p = torch.cat(probs).numpy()
        t = torch.cat(targets).numpy()
        pred = (p >= threshold).astype(int)
        tp = (pred * t).sum(axis=0)
        fp = (pred * (1 - t)).sum(axis=0)
        fn = ((1 - pred) * t).sum(axis=0)
        prec = np.divide(tp, tp + fp, out=np.zeros(len(class_list)), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros(len(class_list)), where=(tp + fn) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(class_list)), where=(prec + rec) > 0)
        out["multilabel"] = {
            "macro_f1": float(f1.mean()),
            "macro_precision": float(prec.mean()),
            "macro_recall": float(rec.mean()),
            "exact_match": float((pred == t).all(axis=1).mean()),
            "no_label_rate": float((pred.sum(axis=1) == 0).mean()),
            "multi_label_rate": float((pred.sum(axis=1) > 1).mean()),
            "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])} for i, c in enumerate(class_list)},
        }
    return out


def train_resnet(model, train_loader, val_loader, class_list, device, epochs: int, lr: float,
                 multilabel: bool, weight_decay: float = config.WEIGHT_DECAY, lr_scales: dict | None = None, log_fn=print):
    model.to(device)
    optimizer = torch.optim.AdamW(layerwise_param_groups(model, lr, lr_scales), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.LR_MIN)
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    best_f1, best_state, history = -1.0, copy.deepcopy(model.state_dict()), []
    for epoch in range(1, epochs + 1):
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        running, n, t0 = 0.0, 0, time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()

        val = evaluate(model, val_loader, class_list, device, multilabel)
        elapsed = time.time() - t0
        log_fn(f"epoch {epoch}/{epochs}  train_loss={running / n:.4f}  val_acc={val['accuracy']:.4f}  "
               f"val_macro_f1={val['macro_f1']:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "train_loss": running / n, "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "elapsed_seconds": elapsed})
        if val["macro_f1"] > best_f1:
            best_f1, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, best_f1
