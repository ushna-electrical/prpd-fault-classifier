"""CLI entry point for Stage 1: train and compare backbones on the Stage 0
manifests.

Usage:
    python -m prpd_classifier.cli \
        --manifests-dir /path/to/Code/manifests \
        --output-dir /path/to/Code/stage1_output \
        --backbones resnet50 efficientnet_b0 mobilenetv3_large
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import config
from .class_weights import compute_class_weights
from .dataset import ManifestDataset
from .evaluate import evaluate_test
from .models import build_model
from .train import train_one_backbone


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests-dir", required=True, type=Path, help="Directory with train/val/test_manifest.csv from Stage 0.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where checkpoints, reports, and the comparison table go.")
    parser.add_argument("--backbones", nargs="+", default=list(config.BACKBONES), choices=config.BACKBONES)
    parser.add_argument("--image-size", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", default=None, help="cpu/cuda/mps; auto-detected if omitted.")
    parser.add_argument("--limit-train", type=int, default=None, help="Debug/smoke-test only: cap number of training images.")
    parser.add_argument("--limit-eval", type=int, default=None, help="Debug/smoke-test only: cap number of val/test images.")
    parser.add_argument(
        "--freeze-backbone-until",
        type=int,
        default=None,
        help="Freeze model.features[:N] (only architectures with a .features Sequential, e.g. "
        "mobilenetv3_large, efficientnet_b0), leaving features[N:] and the classifier head "
        "trainable. Mitigates fast overfitting when fine-tuning the full backbone on a small "
        "dataset (e.g. a single sensor's data). For mobilenetv3_large (17 feature blocks), 16 "
        "freezes everything except the final conv block.",
    )
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        help="Weight the training cross-entropy loss by inverse class frequency. "
        "Val/test metrics stay unweighted for comparability with unweighted runs.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Exponent applied to the balanced weight (1.0 = full inverse-frequency, "
        "0.5 = sqrt-dampened). Only used with --class-weighted-loss.",
    )
    parser.add_argument(
        "--class-weight-max-ratio",
        type=float,
        default=None,
        help="Optional cap: clip the highest class weight to at most this many times the "
        "lowest. Only used with --class-weighted-loss.",
    )
    return parser


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: list | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    class_list = config.FAULT_CLASSES

    train_ds = ManifestDataset(args.manifests_dir / "train_manifest.csv", class_list, args.image_size, augment=True)
    val_ds = ManifestDataset(args.manifests_dir / "val_manifest.csv", class_list, args.image_size, augment=False)
    test_ds = ManifestDataset(args.manifests_dir / "test_manifest.csv", class_list, args.image_size, augment=False)

    if args.limit_train:
        train_ds.rows = train_ds.rows[: args.limit_train]
    if args.limit_eval:
        val_ds.rows = val_ds.rows[: args.limit_eval]
        test_ds.rows = test_ds.rows[: args.limit_eval]

    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(
            train_ds.rows, class_list, power=args.class_weight_power, max_ratio=args.class_weight_max_ratio
        )
        print(f"Using class-weighted loss (power={args.class_weight_power}, max_ratio={args.class_weight_max_ratio}). Weights:")
        for class_name, weight in zip(class_list, class_weights.tolist()):
            print(f"  {class_name}: {weight:.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for backbone_name in args.backbones:
        print(f"\n=== {backbone_name} ===")
        model = build_model(backbone_name, len(class_list), freeze_backbone_until=args.freeze_backbone_until)

        model, history, best_val_f1 = train_one_backbone(
            model, train_loader, val_loader, device, len(class_list),
            epochs=args.epochs, lr=args.lr, patience=args.patience,
            class_weights=class_weights,
        )

        ckpt_path = args.output_dir / f"{backbone_name}_best.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        with open(args.output_dir / f"{backbone_name}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"Evaluating {backbone_name} on test split ...")
        test_metrics = evaluate_test(model, test_loader, device, class_list, args.output_dir, backbone_name)
        print(f"  test_accuracy={test_metrics['accuracy']:.4f}  test_macro_f1={test_metrics['macro_f1']:.4f}")

        f2 = test_metrics["per_class"]["fault_2"]["f1"]
        f3 = test_metrics["per_class"]["fault_3"]["f1"]
        print(f"  fault_2/fault_3 pair (visually similar sub-variants): f1_fault_2={f2:.4f}  f1_fault_3={f3:.4f}")

        row = {
            "backbone": backbone_name,
            "best_val_macro_f1": best_val_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        }
        for class_name in class_list:
            row[f"f1_{class_name}"] = test_metrics["per_class"][class_name]["f1"]
        comparison_rows.append(row)

    comparison_path = args.output_dir / "backbone_comparison.csv"
    fieldnames = ["backbone", "best_val_macro_f1", "test_accuracy", "test_macro_f1"] + [f"f1_{c}" for c in class_list]
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)
    print(f"\nSaved comparison table: {comparison_path}")

    if comparison_rows:
        winner = max(comparison_rows, key=lambda r: r["test_macro_f1"])
        print(f"Winner by test macro-F1: {winner['backbone']} ({winner['test_macro_f1']:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
