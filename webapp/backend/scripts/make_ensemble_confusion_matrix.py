"""Produce the 10-class ensemble confusion matrix for the deployed Sensor-2
final model, which was missing from stage2_output/ (only the two backbones'
individual confusion matrices existed -- prototypical_sensor_2_merged23 and
resnet_sensor_2_merged23 -- not the ensemble that is actually served).

Runs the exported final_model_sensor2_10class ensemble (same FinalModel class
test_model.py and the backend use) over the held-out test split of
manifests_sensor_2_merged23, restricted to the ten shared classes (fault_1 is
excluded, matching the "excluded ['fault_1', 'fault_3', 'fault_5i']" protocol
of stage2_output/results_sensor_2_merged23.txt). This also doubles as an
end-to-end verification: the printed accuracy/macro-F1 should reproduce the
thesis-reported 0.855 / 0.864 (chapters/08_sensors.tex).

    py webapp/backend/scripts/make_ensemble_confusion_matrix.py

Writes to stage2_output/:
    ensemble_confusion_matrix_sensor2_merged23_10class_counts.csv
    ensemble_confusion_matrix_sensor2_merged23_10class_row_percent.csv
    ensemble_confusion_matrix_sensor2_merged23_10class_per_class_report.csv
    ensemble_confusion_matrix_sensor2_merged23_10class.png   (300 DPI)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]  # scripts/ -> backend/ -> webapp/ -> Code/
sys.path.insert(0, str(CODE_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from prpd_classifier.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1_per_class  # noqa: E402
from prpd_fewshot.dataset import load_manifest_rows  # noqa: E402
from test_model import FinalModel  # noqa: E402

TEST_MANIFEST = CODE_ROOT / "manifests_sensor_2_merged23" / "test_manifest.csv"
OUT_DIR = CODE_ROOT / "stage2_output"
STEM = "ensemble_confusion_matrix_sensor2_merged23_10class"


def main() -> int:
    model = FinalModel("2")
    class_list = model.classes  # the 10 shared classes, in the model's own order
    class_to_idx = {c: i for i, c in enumerate(class_list)}

    rows = [r for r in load_manifest_rows(TEST_MANIFEST) if r["fault_type"] in class_to_idx]
    print(f"evaluating {len(rows)} test images over {len(class_list)} classes ...")

    y_true, y_pred = [], []
    for i, row in enumerate(rows):
        result = model.predict(CODE_ROOT / row["filepath"])
        y_true.append(class_to_idx[row["fault_type"]])
        y_pred.append(class_to_idx[result["fault"]])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    cm = confusion_matrix(y_true, y_pred, len(class_list))
    acc = accuracy(cm)
    mf1 = macro_f1(cm)
    per_class = precision_recall_f1_per_class(cm)

    OUT_DIR.mkdir(exist_ok=True)

    with open(OUT_DIR / f"{STEM}_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_fault_type"] + class_list)
        for name, row in zip(class_list, cm):
            w.writerow([name] + row)

    with open(OUT_DIR / f"{STEM}_row_percent.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_fault_type"] + class_list)
        for name, row in zip(class_list, cm):
            total = sum(row) or 1
            w.writerow([name] + [f"{100 * v / total:.1f}" for v in row])

    with open(OUT_DIR / f"{STEM}_per_class_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fault_type", "precision", "recall", "f1", "support"])
        for name, r in zip(class_list, per_class):
            w.writerow([name, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", r["support"]])
        w.writerow(["OVERALL", "", "", f"{mf1:.4f}", sum(r["support"] for r in per_class)])

    fig, ax = plt.subplots(figsize=(9, 8))
    row_pct = [[v / (sum(row) or 1) for v in row] for row in cm]
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_list)))
    ax.set_xticklabels(class_list, rotation=90)
    ax.set_yticks(range(len(class_list)))
    ax.set_yticklabels(class_list)
    ax.set_xlabel("Predicted fault_type")
    ax.set_ylabel("True fault_type")
    ax.set_title(f"Sensor-2 ensemble -- 10-class test confusion matrix (acc={acc:.3f}, macro-F1={mf1:.3f})")
    for r in range(len(class_list)):
        for c in range(len(class_list)):
            if cm[r][c]:
                ax.text(c, r, str(cm[r][c]), ha="center", va="center",
                        color="white" if row_pct[r][c] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300)
    plt.close(fig)

    print(f"\naccuracy={acc:.4f}  macro_f1={mf1:.4f}")
    print("(thesis chapters/08_sensors.tex / stage2_output/sensor_comparison_10class.csv report 0.8553 / 0.8640)")
    for name, r in zip(class_list, per_class):
        print(f"  {name:10s} n={r['support']:3d}  precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1']:.3f}")
    print(f"\nwrote {OUT_DIR / (STEM + '_counts.csv')}")
    print(f"wrote {OUT_DIR / (STEM + '_row_percent.csv')}")
    print(f"wrote {OUT_DIR / (STEM + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
