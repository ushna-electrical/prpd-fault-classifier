"""Test-split evaluation: overall accuracy, macro-F1, per-class precision/
recall/F1, and a confusion matrix saved as both an image and a CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .metrics import accuracy, macro_f1, precision_recall_f1_per_class
from .train import evaluate_loader


def save_confusion_matrix_csv(cm: list[list[int]], class_list: list[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_fault_type"] + class_list)
        for class_name, row in zip(class_list, cm):
            writer.writerow([class_name] + row)


def save_confusion_matrix_image(cm: list[list[int]], class_list: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_list)))
    ax.set_xticklabels(class_list, rotation=90)
    ax.set_yticks(range(len(class_list)))
    ax.set_yticklabels(class_list)
    ax.set_xlabel("Predicted fault_type")
    ax.set_ylabel("True fault_type")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_per_class_report_csv(per_class: list[dict], class_list: list[str], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fault_type", "precision", "recall", "f1", "support"])
        for class_name, r in zip(class_list, per_class):
            writer.writerow([class_name, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", r["support"]])


def evaluate_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
    class_list: list[str],
    output_dir: Path,
    backbone_name: str,
) -> dict:
    metrics = evaluate_loader(model, test_loader, device, len(class_list))
    cm = metrics["confusion_matrix"]
    per_class = precision_recall_f1_per_class(cm)

    save_confusion_matrix_csv(cm, class_list, output_dir / f"{backbone_name}_confusion_matrix.csv")
    save_confusion_matrix_image(
        cm, class_list, output_dir / f"{backbone_name}_confusion_matrix.png",
        title=f"{backbone_name} -- test confusion matrix",
    )
    save_per_class_report_csv(per_class, class_list, output_dir / f"{backbone_name}_per_class_report.csv")

    return {
        "accuracy": accuracy(cm),
        "macro_f1": macro_f1(cm),
        "per_class": {class_name: r for class_name, r in zip(class_list, per_class)},
        "confusion_matrix": cm,
    }
