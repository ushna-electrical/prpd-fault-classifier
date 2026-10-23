"""CLI entry point: train the Siamese few-shot classifier on one sensor's
manifests and evaluate N-way N-shot classification on its test split.

Usage:
    python -m prpd_fewshot.cli \
        --manifests-dir /path/to/Code/manifests_sensor_2 \
        --output-dir /path/to/Code/stage2_output/siamese_sensor_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prpd_classifier.evaluate import save_confusion_matrix_csv, save_confusion_matrix_image, save_per_class_report_csv

from . import config
from .dataset import PairDataset, build_transform, load_manifest_rows
from .evaluate import build_class_prototypes, classify_by_nearest_prototype
from .model import SiameseEncoder, SiameseNetwork
from .train import train_prototypical, train_siamese


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs-per-epoch", type=int, default=config.DEFAULT_PAIRS_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument("--n-shot", type=int, default=config.DEFAULT_N_SHOT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--method",
        choices=["siamese", "prototypical"],
        default="siamese",
        help="siamese: pairwise same/different verification loss (default). "
        "prototypical: Snell et al. 2017 episodic nearest-prototype training, "
        "which matches the eval-time objective directly.",
    )
    parser.add_argument("--n-query", type=int, default=config.DEFAULT_N_QUERY, help="prototypical method only")
    parser.add_argument(
        "--episodes-per-epoch", type=int, default=config.DEFAULT_EPISODES_PER_EPOCH, help="prototypical method only"
    )
    parser.add_argument(
        "--augment-episodes",
        action="store_true",
        default=False,
        help="prototypical method only. Default off: an A/B test found classical "
        "pixel-level augmentation during episodic training HURT this encoder "
        "(0.694 -> 0.589 macro-F1 on sensor_2), so it stays opt-in.",
    )
    parser.add_argument(
        "--session-disjoint-episodes",
        action="store_true",
        default=False,
        help="prototypical method only. Draw each class's episode queries from one "
        "recording session and its support from that class's other sessions, "
        "mirroring test time (support from train sessions, query from a held-out one). "
        "Equivalent to --session-disjoint-prob 1.0.",
    )
    parser.add_argument(
        "--session-disjoint-prob",
        type=float,
        default=0.0,
        help="prototypical method only. Probability that each training episode is "
        "session-disjoint; the rest are sampled at random as before.",
    )
    parser.add_argument(
        "--exclude-classes",
        default="",
        help="Comma-separated fault classes to drop entirely (from the class list AND "
        "from all three splits), e.g. 'fault_1'. Macro-F1 is then averaged over the "
        "remaining classes only, so results are NOT comparable to a full-taxonomy run.",
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
    if not 0.0 <= args.session_disjoint_prob <= 1.0:
        raise SystemExit(f"--session-disjoint-prob must be in [0, 1], got {args.session_disjoint_prob}")
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    class_list = config.FAULT_CLASSES
    excluded = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    if excluded:
        unknown = excluded - set(class_list)
        if unknown:
            raise SystemExit(f"--exclude-classes names classes not in the taxonomy: {sorted(unknown)}")
        class_list = [c for c in class_list if c not in excluded]
        print(f"Excluding {sorted(excluded)} -> {len(class_list)}-class problem")

    train_rows = load_manifest_rows(args.manifests_dir / "train_manifest.csv")
    val_rows = load_manifest_rows(args.manifests_dir / "val_manifest.csv")
    test_rows = load_manifest_rows(args.manifests_dir / "test_manifest.csv")
    if excluded:
        train_rows = [r for r in train_rows if r["fault_type"] not in excluded]
        val_rows = [r for r in val_rows if r["fault_type"] not in excluded]
        test_rows = [r for r in test_rows if r["fault_type"] not in excluded]
    print(f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    eval_transform = build_transform(augment=False)

    if args.method == "siamese":
        pair_dataset = PairDataset(train_rows, augment=True, pairs_per_epoch=args.pairs_per_epoch, seed=args.seed)
        pair_loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = SiameseNetwork(embedding_dim=config.EMBEDDING_DIM, image_size=config.IMAGE_SIZE)

        model, history, best_val_f1 = train_siamese(
            model,
            pair_loader,
            train_rows,
            val_rows,
            eval_transform,
            class_list,
            device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            n_shot=args.n_shot,
            seed=args.seed,
        )
    else:
        model = SiameseEncoder(embedding_dim=config.EMBEDDING_DIM, image_size=config.IMAGE_SIZE)
        train_transform = build_transform(augment=args.augment_episodes)

        model, history, best_val_f1 = train_prototypical(
            model,
            train_rows,
            val_rows,
            train_transform,
            eval_transform,
            class_list,
            device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            n_shot=args.n_shot,
            n_query=args.n_query,
            episodes_per_epoch=args.episodes_per_epoch,
            seed=args.seed,
            session_disjoint_prob=1.0 if args.session_disjoint_episodes else args.session_disjoint_prob,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "siamese_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Evaluating on test split ({args.n_shot}-shot support drawn from train) ...")
    prototypes = build_class_prototypes(model, train_rows, eval_transform, class_list, args.n_shot, device, seed=args.seed)
    test_metrics = classify_by_nearest_prototype(model, test_rows, eval_transform, class_list, prototypes, device)
    print(f"  test_accuracy={test_metrics['accuracy']:.4f}  test_macro_f1={test_metrics['macro_f1']:.4f}")

    backbone_name = args.method
    save_confusion_matrix_csv(test_metrics["confusion_matrix"], class_list, args.output_dir / f"{backbone_name}_confusion_matrix.csv")
    title_label = "Siamese" if args.method == "siamese" else "Prototypical"
    save_confusion_matrix_image(
        test_metrics["confusion_matrix"], class_list, args.output_dir / f"{backbone_name}_confusion_matrix.png",
        title=f"{title_label} {args.n_shot}-shot -- test confusion matrix",
    )
    save_per_class_report_csv(
        [test_metrics["per_class"][c] for c in class_list], class_list, args.output_dir / f"{backbone_name}_per_class_report.csv"
    )
    print(f"Saved evaluation artifacts to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
