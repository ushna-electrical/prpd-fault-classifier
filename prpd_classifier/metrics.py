"""Pure-Python classification metrics (no sklearn/scipy): confusion matrix,
per-class precision/recall/F1, macro-F1, accuracy.

Implemented directly rather than via sklearn because this machine's installed
scipy/sklearn require numpy>=2.0 while this torch build requires numpy<2 --
avoiding that whole dependency conflict is simpler than fighting it, and these
metrics are a handful of lines of arithmetic.
"""

from __future__ import annotations


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def precision_recall_f1_per_class(cm: list[list[int]]) -> list[dict]:
    num_classes = len(cm)
    results = []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][r] for r in range(num_classes) if r != c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({"precision": precision, "recall": recall, "f1": f1, "support": sum(cm[c])})
    return results


def macro_f1(cm: list[list[int]]) -> float:
    per_class = precision_recall_f1_per_class(cm)
    if not per_class:
        return 0.0
    return sum(r["f1"] for r in per_class) / len(per_class)


def accuracy(cm: list[list[int]]) -> float:
    total = sum(sum(row) for row in cm)
    if total == 0:
        return 0.0
    correct = sum(cm[i][i] for i in range(len(cm)))
    return correct / total
