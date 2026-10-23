"""Manifest loading and pair-sampling dataset for Siamese training.

Loads strictly from the Stage 0 manifest CSVs, same discipline as
prpd_classifier -- never re-scans Dataset/ directly, so the leakage-safe
split is respected.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from . import config
from prpd_classifier.transforms import (
    CircularPhaseShift,
    CropToPlotArea,
    ElasticJitter,
    RandomVerticalStretch,
    RandomZoom,
)


def load_manifest_rows(manifest_csv: Path) -> list[dict]:
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_transform(augment: bool) -> T.Compose:
    steps = [CropToPlotArea(config.CROP_BOX), T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]
    if augment:
        steps.append(CircularPhaseShift(config.MAX_PHASE_SHIFT_FRACTION))
        steps.append(ElasticJitter(config.ELASTIC_GRID_SIZE, config.ELASTIC_MAX_DISPLACEMENT_FRACTION))
        steps.append(RandomZoom(config.ZOOM_SCALE_RANGE))
        steps.append(RandomVerticalStretch(config.VERTICAL_STRETCH_RANGE))
    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD))
    return T.Compose(steps)


def group_rows_by_class(rows: list[dict]) -> dict[str, list[dict]]:
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["fault_type"]].append(row)
    return dict(by_class)


class PairDataset(Dataset):
    """Yields (image1, image2, label) where label=1.0 if same fault_type,
    else 0.0. Pairs are sampled fresh on every __getitem__ call (not fixed at
    construction time), so each pass over the dataset -- i.e. each training
    epoch, since a DataLoader calls __getitem__ once per index per epoch --
    sees a newly-sampled, balanced (50/50 same/different) set of pairs rather
    than memorizing one fixed pairing.
    """

    def __init__(self, rows: list[dict], augment: bool, pairs_per_epoch: int, seed: int | None = None):
        self.by_class = group_rows_by_class(rows)
        self.classes = [c for c, r in self.by_class.items() if len(r) >= 1]
        if len(self.classes) < 2:
            raise ValueError("PairDataset needs at least 2 classes with >=1 example each.")
        self.transform = build_transform(augment)
        self.pairs_per_epoch = pairs_per_epoch
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _load(self, row: dict):
        return self.transform(Image.open(row["filepath"]).convert("RGB"))

    def __getitem__(self, idx: int):
        same = idx % 2 == 0
        if same:
            cls = self.rng.choice(self.classes)
            pool = self.by_class[cls]
            if len(pool) >= 2:
                r1, r2 = self.rng.sample(pool, 2)
            else:
                r1 = r2 = pool[0]
            label = 1.0
        else:
            cls1, cls2 = self.rng.sample(self.classes, 2)
            r1 = self.rng.choice(self.by_class[cls1])
            r2 = self.rng.choice(self.by_class[cls2])
            label = 0.0
        return self._load(r1), self._load(r2), label
