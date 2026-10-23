"""Recover an approximate PRPD count matrix from our rendered PNGs, then apply
Mück's input chain (noise mask -> log scaling -> [0,1] -> 224x224, grayscale
replicated to 3 channels).

Why inversion is needed: his pipeline starts from raw MPD 540 event lists and
builds a 512x360 log-spaced charge/phase histogram. We only have matplotlib
renderings of that histogram, so we invert the viridis colormap to get back a
per-pixel intensity.

What we recover, precisely: our plots use a logarithmic colour normalisation,
so the value we read back is (log c - log c_min) / (log c_max - log c_min) in
[0, 1] -- the count on a log scale, normalised per image. His chain computes
log(1+c) and then normalises to [0, 1]. The two agree up to an affine map in
log space, so this is the same representation, NOT absolute event counts
(the colourbar limits differ per image and are not recoverable from pixels).
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PIL import Image

from . import config

_VIRIDIS_LUT = (np.asarray([colormaps["viridis"](i / 255.0)[:3] for i in range(256)]) * 255.0)

_BITS = 6  # quantise each channel to 64 levels -> 262144-entry direct lookup
_LEVELS = 1 << _BITS
_SHIFT = 8 - _BITS
_RGB_TO_INTENSITY = None


def _build_lookup() -> np.ndarray:
    """RGB (6 bits/channel) -> nearest viridis position, as uint8. Built once;
    a per-pixel nearest-neighbour search over the 256-entry LUT is ~40s per
    image, this is ~0.2s."""
    global _RGB_TO_INTENSITY
    if _RGB_TO_INTENSITY is None:
        centres = (np.arange(_LEVELS) << _SHIFT) + (1 << (_SHIFT - 1))
        grid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
        out = np.empty(grid.shape[0], dtype=np.uint8)
        for start in range(0, grid.shape[0], 16384):  # chunked to bound memory
            chunk = grid[start : start + 16384]
            out[start : start + 16384] = ((chunk[:, None, :] - _VIRIDIS_LUT[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        _RGB_TO_INTENSITY = out
    return _RGB_TO_INTENSITY


def png_to_intensity(path: str, crop: tuple[int, int, int, int] = config.CROP_BOX, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Returns a float array in [0, 1]: 0 = no discharge events recorded,
    higher = more events (log scale). out_size is (width, height); the paper's
    histogram is 360 phase bins x 512 charge bins."""
    img = Image.open(path).convert("RGB").crop(crop)
    rgb = np.asarray(img)

    # Unpainted cells are the white plot background -> genuinely zero events.
    background = rgb.min(axis=2) >= config.BACKGROUND_MIN_CHANNEL

    q = (rgb >> _SHIFT).astype(np.int32)
    idx = (q[:, :, 0] * _LEVELS + q[:, :, 1]) * _LEVELS + q[:, :, 2]
    intensity = _build_lookup()[idx].astype(np.float32) / 255.0
    intensity[background] = 0.0

    if out_size is not None:
        intensity = np.asarray(Image.fromarray(intensity).resize(out_size, Image.BILINEAR), dtype=np.float32)
    return intensity


def noise_mask(intensity: np.ndarray) -> np.ndarray:
    """Mück's denoising heuristic: the noise floor has no phase dependence,
    while locally significant deviations are signal. A row (constant charge)
    that is populated across almost the whole phase axis at a uniform level is
    noise; cells that stand out against their row's level are kept.

    Returns a boolean mask, True = keep as signal.
    """
    h, w = intensity.shape
    occupied = intensity > 0
    row_occupancy = occupied.mean(axis=1)  # fraction of phase axis populated
    row_level = np.where(occupied.sum(axis=1) > 0, intensity.sum(axis=1) / np.maximum(occupied.sum(axis=1), 1), 0.0)

    noise_rows = row_occupancy >= config.NOISE_ROW_OCCUPANCY
    keep = occupied.copy()
    # inside a noise row, keep only cells clearly brighter than that row's typical level
    standout = intensity > (row_level[:, None] * config.NOISE_ROW_STANDOUT)
    keep[noise_rows] = (occupied & standout)[noise_rows]
    return keep


def to_model_input(intensity: np.ndarray, apply_mask: bool = True, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """intensity -> (3, size, size) float32, ImageNet-standardised."""
    x = intensity.copy()
    if apply_mask:
        x = np.where(noise_mask(x), x, 0.0)

    # his log(1+x) + per-image [0,1] normalisation (our values are already on a
    # log scale, so log1p here only mildly re-compresses; normalisation is what
    # removes the absolute event count, which is the stated intent)
    x = np.log1p(x)
    peak = x.max()
    if peak > 0:
        x = x / peak

    x = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)
    mean = np.asarray(config.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(config.IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (x - mean) / std
