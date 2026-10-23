"""Produce (and cache) the class-probability matrices the calibrator needs:
validation for fitting, test for evaluation.

Both trained models are reused as-is -- calibration is post-hoc and never
retrains or changes a prediction's ranking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from prpd_fewshot import config as pconf
from prpd_fewshot.dataset import build_transform, load_manifest_rows
from prpd_fewshot.model import SiameseEncoder
from prpd_resnet.dataset import cache_path, to_tensor
from prpd_resnet.model import PRPDResNet
from prpd_resnet.represent import noise_mask

ROOT = Path("/Users/zohairahmed/Documents/Ushna/Code")
OUT = ROOT / "stage2_output"
SHARED_CLASSES = ["fault_2", "fault_4", "fault_5", "fault_6", "fault_7", "fault_8", "fault_9", "fault_10", "fault_12", "fault_13"]
DRAWS, N_SHOT = [0, 1, 2, 3, 4], 5

SENSOR_RUNS = {
    "sensor_1": ("manifests_sensor_1", "prototypical_sensor_1", "resnet_sensor_1", {"fault_1", "fault_5i"}),
    "sensor_2": ("manifests_sensor_2", "prototypical_sensor_2_newdata", "resnet_sensor_2_newdata", {"fault_1"}),
    "sensor_3": ("manifests_sensor_3", "prototypical_sensor_3", "resnet_sensor_3", {"fault_1", "fault_3", "fault_5i"}),
}


def compute_probabilities(sensor: str, classes: list[str] = None) -> dict:
    """Returns val/test probability matrices for the prototypical model, the
    ResNet and their equal-weight ensemble, plus the labels and the rows."""
    classes = classes or SHARED_CLASSES
    mdir, proto_dir, resnet_dir, excluded = SENSOR_RUNS[sensor]
    trained_classes = [c for c in pconf.FAULT_CLASSES if c not in excluded]
    keep_idx = [trained_classes.index(c) for c in classes]
    keep = set(classes)

    rows = {s: [r for r in load_manifest_rows(ROOT / mdir / f"{s}_manifest.csv") if r["fault_type"] in keep]
            for s in ("train", "val", "test")}
    labels = {s: np.array([classes.index(r["fault_type"]) for r in rows[s]]) for s in ("val", "test")}

    tf = build_transform(augment=False)
    enc = SiameseEncoder(embedding_dim=pconf.EMBEDDING_DIM, image_size=pconf.IMAGE_SIZE)
    enc.load_state_dict(torch.load(OUT / proto_dir / "siamese_best.pt", map_location="cpu"))
    enc.eval()
    emb = {}
    for split in ("train", "val", "test"):
        x = torch.stack([tf(Image.open(r["filepath"]).convert("RGB")) for r in rows[split]])
        with torch.no_grad():
            emb[split] = torch.cat([enc.embed(x[i : i + 64]) for i in range(0, len(rows[split]), 64)])
    by_class = {c: [i for i, r in enumerate(rows["train"]) if r["fault_type"] == c] for c in classes}

    def proto(split):
        acc = []
        for draw in DRAWS:
            g = torch.Generator().manual_seed(draw)
            protos = torch.stack([emb["train"][torch.tensor(by_class[c])[torch.randperm(len(by_class[c]), generator=g)[:N_SHOT]]].mean(0)
                                  for c in classes])
            acc.append(F.softmax(-(torch.cdist(emb[split], protos) ** 2), dim=1).numpy())
        return np.mean(acc, axis=0)

    rm = PRPDResNet(num_classes=len(trained_classes), pretrained=False)
    rm.load_state_dict(torch.load(OUT / resnet_dir / "resnet_best.pt", map_location="cpu"))
    rm.eval()

    def resnet(split):
        x = torch.stack([to_tensor(np.where(noise_mask(v), v, 0.0)) for v in
                         (np.load(cache_path(r["filepath"], ROOT / "cache_prpd_matrices")).astype(np.float32) for r in rows[split])])
        with torch.no_grad():
            logits = torch.cat([rm(x[i : i + 32]) for i in range(0, len(rows[split]), 32)])
        return F.softmax(logits[:, keep_idx], dim=1).numpy()

    out = {"classes": classes, "labels": labels, "rows": rows}
    for split in ("val", "test"):
        p, r = proto(split), resnet(split)
        out[split] = {"Prototypical": p, "ResNet-18": r, "Ensemble": 0.5 * p + 0.5 * r}
    return out


def cached_probabilities(sensor: str, cache: Path = None) -> dict:
    """compute_probabilities with an npz cache (the forward passes take minutes)."""
    cache = cache or OUT / f"calibration_probs_{sensor}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {
            "classes": list(z["classes"]),
            "labels": {s: z[f"labels_{s}"] for s in ("val", "test")},
            **{s: {m: z[f"{s}_{m}"] for m in ("Prototypical", "ResNet-18", "Ensemble")} for s in ("val", "test")},
        }
    data = compute_probabilities(sensor)
    np.savez(cache, classes=np.array(data["classes"]),
             **{f"labels_{s}": data["labels"][s] for s in ("val", "test")},
             **{f"{s}_{m}": data[s][m] for s in ("val", "test") for m in data[s]})
    return data
