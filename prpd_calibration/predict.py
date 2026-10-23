"""Classify a single PRPD image and report how much the answer can be trusted.

    python -m prpd_calibration.predict <image.png> [--sensor sensor_2]

Returns the predicted fault, its confidence, the reliability of that confidence
band measured on held-out data, and a recommendation to accept or review.

The confidence comes from the equal-weight ensemble of the prototypical network
and the ResNet-18. It is reported uncalibrated: on this data the ensemble's raw
confidence is already close to calibrated (test ECE 0.043), while calibrators
fitted on the validation split did not transfer to test (see
stage2_output/calibration_sensor_2.png).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prpd_fewshot import config as pconf
from prpd_fewshot.dataset import build_transform, load_manifest_rows
from prpd_fewshot.model import SiameseEncoder
from prpd_resnet import config as rconf
from prpd_resnet.dataset import to_tensor
from prpd_resnet.model import PRPDResNet
from prpd_resnet.represent import noise_mask, png_to_intensity

from .probs import ROOT, SENSOR_RUNS, SHARED_CLASSES

MATRIX_SIZE = (360, 512)
DRAWS, N_SHOT = [0, 1, 2, 3, 4], 5
OUT = ROOT / "stage2_output"


def _reliability_table(sensor: str) -> list[dict]:
    """Measured accuracy per confidence band, from the operating-point CSV."""
    path = OUT / f"calibration_operating_points_{sensor}.csv"
    if not path.exists():
        return []
    import csv

    rows = []
    for r in csv.DictReader(open(path)):
        if r["min_confidence"] != "all":
            rows.append({"min_confidence": float(r["min_confidence"]),
                         "accuracy": float(r["accuracy"]), "coverage": float(r["coverage"])})
    return sorted(rows, key=lambda r: -r["min_confidence"])


class Predictor:
    def __init__(self, sensor: str = "sensor_2", classes: list[str] = None):
        self.sensor = sensor
        self.classes = classes or SHARED_CLASSES
        mdir, proto_dir, resnet_dir, excluded = SENSOR_RUNS[sensor]
        trained = [c for c in pconf.FAULT_CLASSES if c not in excluded]
        self.keep_idx = [trained.index(c) for c in self.classes]

        self.tf = build_transform(augment=False)
        self.enc = SiameseEncoder(embedding_dim=pconf.EMBEDDING_DIM, image_size=pconf.IMAGE_SIZE)
        self.enc.load_state_dict(torch.load(OUT / proto_dir / "siamese_best.pt", map_location="cpu"))
        self.enc.eval()

        self.resnet = PRPDResNet(num_classes=len(trained), pretrained=False)
        self.resnet.load_state_dict(torch.load(OUT / resnet_dir / "resnet_best.pt", map_location="cpu"))
        self.resnet.eval()

        # class prototypes, built once from the training split (same 5 support
        # draws used throughout, averaged)
        train_rows = [r for r in load_manifest_rows(ROOT / mdir / "train_manifest.csv")
                      if r["fault_type"] in set(self.classes)]
        x = torch.stack([self.tf(Image.open(r["filepath"]).convert("RGB")) for r in train_rows])
        with torch.no_grad():
            emb = torch.cat([self.enc.embed(x[i : i + 64]) for i in range(0, len(train_rows), 64)])
        by_class = {c: [i for i, r in enumerate(train_rows) if r["fault_type"] == c] for c in self.classes}
        self.prototypes = []
        for draw in DRAWS:
            g = torch.Generator().manual_seed(draw)
            self.prototypes.append(torch.stack(
                [emb[torch.tensor(by_class[c])[torch.randperm(len(by_class[c]), generator=g)[:N_SHOT]]].mean(0)
                 for c in self.classes]))
        self.reliability = _reliability_table(sensor)

    def predict(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            e = self.enc.embed(self.tf(img).unsqueeze(0))
            p_proto = np.mean([F.softmax(-(torch.cdist(e, pr) ** 2), dim=1).numpy() for pr in self.prototypes], axis=0)[0]

            matrix = png_to_intensity(image_path, crop=rconf.CROP_BOX, out_size=MATRIX_SIZE)
            t = to_tensor(np.where(noise_mask(matrix), matrix, 0.0)).unsqueeze(0)
            p_res = F.softmax(self.resnet(t)[:, self.keep_idx], dim=1).numpy()[0]

        probs = 0.5 * p_proto + 0.5 * p_res
        order = np.argsort(-probs)
        conf = float(probs[order[0]])
        band = next((r for r in self.reliability if conf >= r["min_confidence"]), None)
        return {
            "fault": self.classes[order[0]],
            "confidence": conf,
            "band": band,
            "runner_up": (self.classes[order[1]], float(probs[order[1]])),
            "top3": [(self.classes[i], float(probs[i])) for i in order[:3]],
            "models": {"prototypical": (self.classes[int(p_proto.argmax())], float(p_proto.max())),
                       "resnet": (self.classes[int(p_res.argmax())], float(p_res.max()))},
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=str)
    ap.add_argument("--sensor", default="sensor_2", choices=sorted(SENSOR_RUNS))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    r = Predictor(args.sensor).predict(args.image)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"\n  image      {Path(args.image).name}")
    print(f"  prediction {r['fault']}")
    print(f"  confidence {100 * r['confidence']:.1f}%")
    if r["band"]:
        b = r["band"]
        print(f"  reliability: predictions at >= {100 * b['min_confidence']:.0f}% confidence were correct "
              f"{100 * b['accuracy']:.1f}% of the time on held-out test data ({100 * b['coverage']:.0f}% of images reach this band)")
    else:
        print("  reliability: below the lowest measured band -- treat as unreliable and review manually")
    verdict = "accept" if r["confidence"] >= 0.9 else ("review" if r["confidence"] >= 0.7 else "manual inspection advised")
    print(f"  recommendation: {verdict}")
    print(f"  runner-up  {r['runner_up'][0]} at {100 * r['runner_up'][1]:.1f}%")
    print("  per model  " + ", ".join(f"{k}: {v[0]} {100 * v[1]:.0f}%" for k, v in r["models"].items()))
    agree = r["models"]["prototypical"][0] == r["models"]["resnet"][0]
    print(f"             the two models {'agree' if agree else 'DISAGREE -- treat with caution'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
