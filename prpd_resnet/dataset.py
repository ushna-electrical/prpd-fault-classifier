"""Dataset over cached PRPD matrices.

The colormap inversion costs ~0.1 s per image, so every image is inverted once
into a 512x360 float16 .npy cache (the paper's histogram size) and the training
loop reads those. Augmentation and the final 224x224 conversion happen per
epoch on the cached matrix.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .augment import augment
from .represent import noise_mask, png_to_intensity

MATRIX_SIZE = (360, 512)  # (width=phase bins, height=charge bins), as in the paper


def load_manifest_rows(manifest_csv: Path) -> list[dict]:
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cache_path(filepath: str, cache_dir: Path) -> Path:
    return cache_dir / (hashlib.md5(filepath.encode()).hexdigest() + ".npy")


def build_cache(rows: list[dict], cache_dir: Path, log_every: int = 250) -> int:
    cache_dir.mkdir(parents=True, exist_ok=True)
    built = 0
    for i, row in enumerate(rows):
        out = cache_path(row["filepath"], cache_dir)
        if not out.exists():
            np.save(out, png_to_intensity(row["filepath"], out_size=MATRIX_SIZE).astype(np.float16))
            built += 1
        if log_every and (i + 1) % log_every == 0:
            print(f"  cached {i + 1}/{len(rows)}", flush=True)
    return built


def to_tensor(x: np.ndarray, size: int = config.IMAGE_SIZE) -> torch.Tensor:
    """Matrix -> (3, size, size) ImageNet-standardised tensor: log(1+x),
    per-image [0,1] normalisation, grayscale replicated to 3 channels."""
    x = np.log1p(np.asarray(x, dtype=np.float32))
    peak = float(x.max())
    if peak > 0:
        x = x / peak
    t = torch.from_numpy(x)[None, None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]
    t = t.repeat(3, 1, 1)
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


class PRPDMatrixDataset(Dataset):
    def __init__(self, rows: list[dict], class_list: list[str], cache_dir: Path, train: bool,
                 multilabel: bool, apply_mask: bool = True, aug_scale: float = 1.0, seed: int = 0):
        self.rows = rows
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        self.num_classes = len(class_list)
        self.cache_dir = cache_dir
        self.train = train
        self.multilabel = multilabel
        self.apply_mask = apply_mask
        self.aug_scale = aug_scale
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch  # keeps augmentation fresh each epoch but reproducible

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = np.load(cache_path(row["filepath"], self.cache_dir)).astype(np.float32)
        if self.apply_mask:
            x = np.where(noise_mask(x), x, 0.0)
        if self.train:
            rng = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 10_007 + idx))
            x = augment(x, rng, scale=self.aug_scale)

        label_idx = self.class_to_idx[row["fault_type"]]
        if self.multilabel:
            y = torch.zeros(self.num_classes)
            y[label_idx] = 1.0
        else:
            y = torch.tensor(label_idx, dtype=torch.long)
        return to_tensor(x), y
