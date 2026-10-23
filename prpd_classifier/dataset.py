"""Loads images strictly from the Stage 0 manifest CSVs, never by scanning
Dataset/ directly -- this is what makes the leakage-safe split effective:
whichever images Stage 0 assigned to train/val/test is exactly what gets used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from . import config
from .transforms import (
    AddGaussianNoise,
    CircularPhaseShift,
    CropToPlotArea,
    ElasticJitter,
    RandomVerticalStretch,
    RandomZoom,
)


def load_manifest_rows(manifest_csv: Path) -> list[dict]:
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_transform(image_size: int, augment: bool) -> T.Compose:
    steps = [CropToPlotArea(config.CROP_BOX), T.Resize((image_size, image_size))]
    if augment:
        # CircularPhaseShift: our own phase-periodicity-justified augmentation.
        # ElasticJitter / RandomZoom / RandomVerticalStretch: adopted from the
        # PRPD dataset paper's (#10) validated augmentation set -- see
        # transforms.py docstring for why flips/shearing are excluded.
        steps.append(CircularPhaseShift(config.MAX_PHASE_SHIFT_FRACTION))
        steps.append(ElasticJitter(config.ELASTIC_GRID_SIZE, config.ELASTIC_MAX_DISPLACEMENT_FRACTION))
        steps.append(RandomZoom(config.ZOOM_SCALE_RANGE))
        steps.append(RandomVerticalStretch(config.VERTICAL_STRETCH_RANGE))
    steps.append(T.ToTensor())
    if augment:
        steps.append(AddGaussianNoise(config.GAUSSIAN_NOISE_STD))
    steps.append(T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD))
    return T.Compose(steps)


class ManifestDataset(Dataset):
    """One split (train/val/test) of the Stage 0 manifest.

    `augment=True` must only ever be used for the training split -- val/test
    must see the model's un-augmented view of the data.
    """

    def __init__(self, manifest_csv: Path, class_list: list[str], image_size: int, augment: bool):
        self.rows = load_manifest_rows(manifest_csv)
        self.class_to_idx = {name: idx for idx, name in enumerate(class_list)}
        self.transform = build_transform(image_size, augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["fault_type"]]
        return image, label
