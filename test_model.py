"""Classify PRPD images with the final model (sensor 2, 10 classes, prototypical + ResNet-18 ensemble).

    py test_model.py image.png --voltage 4.5
    py test_model.py folder_of_images/ another.png --voltage 4.5 --top 5 --csv results.csv
    py test_model.py sensor1_image.png --sensor 1 --voltage 4.5

--sensor picks that sensor's own models (default 2); a model only works on its own sensor's images.

--voltage is the voltage (kV) at which the discharge was observed. It drives a
final safety check: when the ensemble cannot separate two or more classes (their
probabilities are within --tie-margin of the top one), the classes whose
metadata pdiv range [min, max] contains the voltage win; among those, the one
the model ranks highest. If the model is not tied, its answer stands and the
voltage only produces a warning when it falls outside the predicted class's range.
One voltage applies to every image in the call.

Needs final_model_sensor<N>_10class/ (create it once with `py export_final_model.py --sensor N`)
and voltage_ranges.json (`py build_voltage_ranges.py`).

The input must be a PRPD plot rendered exactly like the training images
(1800x1050 px, same axes/colorbar layout): both models read fixed pixel
windows, so a differently sized or laid-out image gives meaningless output.
The confidence shown is the raw 50/50 average of the two models' softmax
outputs -- it is not calibrated for images outside the held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prpd_fewshot import config as pconf
from prpd_fewshot.dataset import build_transform
from prpd_fewshot.model import SiameseEncoder
from prpd_resnet.dataset import MATRIX_SIZE, to_tensor
from prpd_resnet.model import PRPDResNet
from prpd_resnet.represent import noise_mask, png_to_intensity

ROOT = Path(__file__).resolve().parent
VOLTAGE_RANGES = ROOT / "voltage_ranges.json"  # one table for all sensors (the metadata has no sensor column)


def model_dir_for(sensor: str) -> Path:
    return ROOT / f"final_model_sensor{sensor}_10class"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_TIE_MARGIN = 0.15  # classes within this probability of the top class count as tied


class FinalModel:
    def __init__(self, sensor: str = "2"):
        model_dir = model_dir_for(sensor)
        if not (model_dir / "meta.json").exists():
            raise SystemExit(f"{model_dir} not found -- run `py export_final_model.py --sensor {sensor}` first.")
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        self.expected_size = tuple(meta["expected_image_size"])
        self.w_proto = meta["ensemble_weights"]["prototypical"]
        self.w_resnet = meta["ensemble_weights"]["resnet18"]

        self.tf = build_transform(augment=False)
        self.enc = SiameseEncoder(embedding_dim=pconf.EMBEDDING_DIM, image_size=pconf.IMAGE_SIZE)
        self.enc.load_state_dict(torch.load(model_dir / "prototypical_encoder.pt", map_location="cpu", weights_only=True))
        self.enc.eval()
        self.prototypes = torch.load(model_dir / "prototypes.pt", map_location="cpu", weights_only=True)  # (draws, classes, dim)

        # sensors 1/3 were trained on extra classes; the ResNet softmax is restricted to the 10 shared ones
        self.resnet = PRPDResNet(num_classes=meta.get("resnet_num_classes", len(self.classes)), pretrained=False)
        self.resnet.load_state_dict(torch.load(model_dir / "resnet18.pt", map_location="cpu", weights_only=True))
        self.resnet.eval()
        self.keep_idx = meta.get("resnet_keep_idx", list(range(len(self.classes))))

        if not VOLTAGE_RANGES.exists():
            raise SystemExit(f"{VOLTAGE_RANGES} not found -- run `py build_voltage_ranges.py` first.")
        self.ranges = json.loads(VOLTAGE_RANGES.read_text(encoding="utf-8"))["ranges"]

    def voltage_check(self, result: dict, voltage: float, margin: float = DEFAULT_TIE_MARGIN) -> dict:
        """Safety layer. result is the output of predict(); returns the final class and what the check did."""
        inside = lambda c: self.ranges[c]["min_kv"] <= voltage <= self.ranges[c]["max_kv"]
        top_class, top_p = result["top"][0]
        tied = [(c, p) for c, p in result["top"] if top_p - p <= margin]  # already sorted by probability
        check = {"voltage": voltage, "model_fault": top_class, "final": top_class, "tied": [c for c, _ in tied]}
        if len(tied) == 1:
            check["action"] = "not tied" + ("" if inside(top_class) else
                                            f"; WARNING: {voltage:g} kV is outside {top_class}'s range "
                                            f"({self.ranges[top_class]['min_kv']:g}-{self.ranges[top_class]['max_kv']:g} kV)")
            return check
        matching = [c for c, _ in tied if inside(c)]
        if not matching:
            check["action"] = "tied, but the voltage matches none of the tied classes -- model answer kept, review manually"
        elif len(matching) == len(tied):
            check["action"] = "tied, but the voltage is consistent with all of them -- cannot separate, model answer kept"
        else:
            check["final"] = matching[0]
            check["action"] = ("tied; voltage confirms the model's answer" if matching[0] == top_class
                               else f"tied; voltage overrides {top_class} -> {matching[0]}")
        return check

    def predict(self, path: Path) -> dict:
        img = Image.open(path)
        size_ok = img.size == self.expected_size
        img = img.convert("RGB")

        with torch.inference_mode():
            e = self.enc.embed(self.tf(img).unsqueeze(0))
            p_proto = torch.stack([F.softmax(-(torch.cdist(e, pr) ** 2), dim=1) for pr in self.prototypes]).mean(0)[0].numpy()

            # the ResNet was trained on a float16 cache of the colormap-inverted matrix
            matrix = png_to_intensity(str(path), out_size=MATRIX_SIZE).astype(np.float16).astype(np.float32)
            x = to_tensor(np.where(noise_mask(matrix), matrix, 0.0)).unsqueeze(0)
            p_res = F.softmax(self.resnet(x)[:, self.keep_idx], dim=1)[0].numpy()

        probs = self.w_proto * p_proto + self.w_resnet * p_res
        order = np.argsort(-probs)
        return {
            "fault": self.classes[order[0]],
            "confidence": float(probs[order[0]]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "top": [(self.classes[i], float(probs[i])) for i in order],
            "prototypical": (self.classes[int(p_proto.argmax())], float(p_proto.max())),
            "resnet18": (self.classes[int(p_res.argmax())], float(p_res.max())),
            "size": img.size,
            "size_ok": size_ok,
        }


def collect_images(paths: list[str]) -> list[Path]:
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("._"))
        elif p.is_file():
            found.append(p)
        else:
            print(f"skipping (not found): {p}")
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files and/or folders (searched recursively)")
    ap.add_argument("--sensor", choices=["1", "2", "3"], default="2",
                    help="which sensor's final model to use; images must come from that sensor (default 2)")
    ap.add_argument("--voltage", type=float, required=True, help="voltage in kV at which the discharge occurs (applies to all images)")
    ap.add_argument("--tie-margin", type=float, default=DEFAULT_TIE_MARGIN,
                    help=f"classes within this probability of the top class are 'tied' and settled by voltage (default {DEFAULT_TIE_MARGIN})")
    ap.add_argument("--top", type=int, default=3, help="how many top classes to show per image (default 3)")
    ap.add_argument("--csv", type=Path, help="also write all class probabilities to this CSV")
    args = ap.parse_args(argv)

    images = collect_images(args.inputs)
    if not images:
        raise SystemExit("no images found")

    model = FinalModel(args.sensor)
    results = []
    for path in images:
        r = model.predict(path)
        r["check"] = model.voltage_check(r, args.voltage, args.tie_margin)
        results.append((path, r))
        agree = r["prototypical"][0] == r["resnet18"][0]
        print(f"\n{path}")
        if not r["size_ok"]:
            print(f"  WARNING: image is {r['size'][0]}x{r['size'][1]}, training plots are "
                  f"{model.expected_size[0]}x{model.expected_size[1]} -- the result is not trustworthy")
        chk = r["check"]
        print(f"  model said  {r['fault']}  ({100 * r['confidence']:.1f}%)")
        print("  top " + str(args.top) + "      " + ", ".join(f"{c} {100 * p:.1f}%" for c, p in r["top"][: args.top]))
        print(f"  per model   prototypical: {r['prototypical'][0]} {100 * r['prototypical'][1]:.0f}%,  "
              f"resnet18: {r['resnet18'][0]} {100 * r['resnet18'][1]:.0f}%  "
              f"-> {'agree' if agree else 'DISAGREE, treat with caution'}")
        print(f"  voltage     {args.voltage:g} kV: {chk['action']}")
        if len(chk["tied"]) > 1:
            print("              tied: " + ", ".join(f"{c} ({model.ranges[c]['min_kv']:g}-{model.ranges[c]['max_kv']:g} kV)" for c in chk["tied"]))
        print(f"  FINAL       {chk['final']}")

    if len(results) > 1:
        counts: dict[str, int] = {}
        for _, r in results:
            counts[r["check"]["final"]] = counts.get(r["check"]["final"], 0) + 1
        print("\nsummary: " + ", ".join(f"{c}: {n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "final_fault", "model_fault", "confidence", "voltage_kv", "voltage_action", *model.classes])
            for path, r in results:
                w.writerow([str(path), r["check"]["final"], r["fault"], f"{r['confidence']:.4f}", args.voltage, r["check"]["action"],
                            *(f"{r['probs'][c]:.4f}" for c in model.classes)])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
