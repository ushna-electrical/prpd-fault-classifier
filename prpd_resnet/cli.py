"""Train the Mück-style ResNet-18 classifier on our sensor manifests.

Usage:
    python -m prpd_resnet.cli --manifests-dir manifests_sensor_2 \
        --output-dir stage2_output/resnet_sensor_2_singlelabel \
        --exclude-classes fault_1 --mode single
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prpd_classifier.evaluate import save_confusion_matrix_csv, save_confusion_matrix_image, save_per_class_report_csv

from . import config
from .dataset import PRPDMatrixDataset, build_cache, load_manifest_rows
from .model import PRPDResNet
from .train import evaluate, train_resnet


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("cache_prpd_matrices"))
    p.add_argument("--mode", choices=["single", "multi"], default="single",
                   help="single: softmax + cross-entropy (comparable to our other models). "
                        "multi: independent sigmoids + BCE (the paper's multi-label setup).")
    p.add_argument("--exclude-classes", default="")
    p.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    p.add_argument("--lr-scale", choices=sorted(config.LR_SCALE_PRESETS), default="paper",
                   help="layer-wise LR multipliers: 'paper' (table 4.3, early layers near-frozen) "
                        "or 'relaxed' (lets the backbone adapt).")
    p.add_argument("--no-noise-mask", action="store_true", help="skip the denoising heuristic")
    p.add_argument("--no-augment", action="store_true", help="ablation: train without augmentation")
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p


def main(argv: list | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    multilabel = args.mode == "multi"

    class_list = config.FAULT_CLASSES
    excluded = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    if excluded:
        unknown = excluded - set(class_list)
        if unknown:
            raise SystemExit(f"--exclude-classes names unknown classes: {sorted(unknown)}")
        class_list = [c for c in class_list if c not in excluded]
        print(f"Excluding {sorted(excluded)} -> {len(class_list)}-class problem")

    splits = {}
    for name in ("train", "val", "test"):
        rows = load_manifest_rows(args.manifests_dir / f"{name}_manifest.csv")
        splits[name] = [r for r in rows if r["fault_type"] in set(class_list)]
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}  mode={args.mode}")

    print("Building matrix cache (colormap inversion, once per image) ...", flush=True)
    built = build_cache([r for rows in splits.values() for r in rows], args.cache_dir)
    print(f"  cache ready ({built} new, {args.cache_dir})", flush=True)

    def make(name, train):
        ds = PRPDMatrixDataset(splits[name], class_list, args.cache_dir, train=train and not args.no_augment,
                               multilabel=multilabel, apply_mask=not args.no_noise_mask, seed=args.seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)

    model = PRPDResNet(num_classes=len(class_list))
    print(f"augmentation={'off' if args.no_augment else 'on'}  lr_scale={args.lr_scale}")
    model, history, best_val_f1 = train_resnet(model, make("train", True), make("val", False), class_list,
                                               args.device, epochs=args.epochs, lr=args.lr, multilabel=multilabel,
                                               lr_scales=config.LR_SCALE_PRESETS[args.lr_scale])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "resnet_best.pt")
    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    test = evaluate(model, make("test", False), class_list, args.device, multilabel)
    print(f"  test_accuracy={test['accuracy']:.4f}  test_macro_f1={test['macro_f1']:.4f}")
    if multilabel:
        m = test["multilabel"]
        print(f"  [multi-label @0.5] macro_f1={m['macro_f1']:.4f}  precision={m['macro_precision']:.4f}  "
              f"recall={m['macro_recall']:.4f}  exact_match={m['exact_match']:.4f}  "
              f"no_label={m['no_label_rate']:.4f}  multi_label={m['multi_label_rate']:.4f}")

    tag = f"resnet_{args.mode}"
    save_confusion_matrix_csv(test["confusion_matrix"], class_list, args.output_dir / f"{tag}_confusion_matrix.csv")
    save_confusion_matrix_image(test["confusion_matrix"], class_list, args.output_dir / f"{tag}_confusion_matrix.png",
                                title=f"ResNet-18 ({args.mode}-label) -- test confusion matrix")
    save_per_class_report_csv([test["per_class"][c] for c in class_list], class_list, args.output_dir / f"{tag}_per_class_report.csv")
    with open(args.output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in test.items() if k != "confusion_matrix"}, f, indent=2, default=float)
    print(f"Saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
