"""The nine augmentations of Mück table 4.2, applied to the recovered PRPD
matrix (float array in [0, 1], rows = charge, columns = phase).

Parameters are drawn uniformly from fixed intervals per sample. Every
technique fires independently, except the noise band and the exponentially
decaying noise, which the paper forbids from being applied together (to avoid
destroying too much information).

No rotations or shears: the paper imposes the same physical-plausibility
constraint we do.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def phase_shift(x, rng):
    """Cyclic shift along the phase axis -- an imperfectly synchronised phase reference."""
    return np.roll(x, rng.integers(0, x.shape[1]), axis=1)


def charge_scaling(x, rng, lo=0.85, hi=1.18):
    """Stretch/compress the charge axis."""
    h, w = x.shape
    new_h = max(2, int(round(h * rng.uniform(lo, hi))))
    y = np.asarray(Image.fromarray(x).resize((w, new_h), Image.BILINEAR), dtype=np.float32)
    if new_h >= h:  # crop back, keeping the vertical centre
        top = (new_h - h) // 2
        return y[top : top + h]
    out = np.zeros((h, w), dtype=np.float32)
    top = (h - new_h) // 2
    out[top : top + new_h] = y
    return out


def frequency_scaling(x, rng, lo=0.5, hi=2.0):
    """Scale all entries (a different measurement duration), saturating at one."""
    return np.clip(x * rng.uniform(lo, hi), 0.0, 1.0)


def gaussian_blur(x, rng, lo=0.4, hi=1.2):
    """Separable Gaussian convolution. Done in numpy because PIL cannot filter
    float ('F' mode) images, and converting to uint8 would quantise the matrix."""
    sigma = rng.uniform(lo, hi)
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    out = np.zeros_like(x)
    padded = np.pad(x, ((0, 0), (radius, radius)), mode="wrap")  # phase axis is periodic
    for k, off in zip(kernel, offsets):
        out += k * padded[:, radius + off : radius + off + x.shape[1]]
    blurred = np.zeros_like(out)
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")  # charge axis is not
    for k, off in zip(kernel, offsets):
        blurred += k * padded[radius + off : radius + off + x.shape[0], :]
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def event_dropout(x, rng, lo=0.05, hi=0.30):
    """Randomly remove individual events (stochastic measurement dropouts)."""
    keep = rng.random(x.shape) >= rng.uniform(lo, hi)
    return np.where(keep, x, 0.0).astype(np.float32)


def white_noise(x, rng, lo=0.01, hi=0.05):
    return np.clip(x + rng.normal(0.0, rng.uniform(lo, hi), x.shape), 0.0, 1.0).astype(np.float32)


def noise_band(x, rng, std_lo=0.02, std_hi=0.10, centre=0.75, intensity=(0.4, 0.9)):
    """Phase-uncorrelated normally distributed band of noise. The paper centres
    it at 0.25 of the tensor height measured from the charge axis' low end; our
    matrices have low charge at the bottom, hence centre=0.75 in row terms."""
    h, w = x.shape
    rows = np.arange(h)[:, None]
    sigma = max(1.0, rng.uniform(std_lo, std_hi) * h)
    profile = np.exp(-0.5 * ((rows - centre * h) / sigma) ** 2)
    band = profile * rng.uniform(*intensity) * rng.random((h, w))
    return np.clip(x + band, 0.0, 1.0).astype(np.float32)


def exponential_noise(x, rng, decay=(2.0, 8.0), intensity=(0.2, 0.6)):
    """Charge-dependent noise decaying exponentially away from the low-charge end."""
    h, w = x.shape
    depth = (h - 1 - np.arange(h)[:, None]) / h  # 0 at the bottom row (low charge)
    envelope = np.exp(-rng.uniform(*decay) * depth)
    return np.clip(x + envelope * rng.uniform(*intensity) * rng.random((h, w)), 0.0, 1.0).astype(np.float32)


def vertical_stripes(x, rng, count=(3, 25), width=(1, 3), intensity=(0.6, 1.0)):
    """Dotted vertical lines at random phases -- external transient disturbers
    (EMC bursts), which are phase-uncorrelated."""
    out = x.copy()
    h, w = x.shape
    for _ in range(rng.integers(*count)):
        c = rng.integers(0, w)
        half = rng.integers(*width)
        top, bottom = sorted(rng.integers(0, h, size=2))
        dotted = rng.random((bottom - top + 1, 1)) > 0.5  # strich-punkt pattern
        block = dotted * rng.uniform(*intensity)
        lo, hi = max(0, c - half), min(w, c + half + 1)
        out[top : bottom + 1, lo:hi] = np.maximum(out[top : bottom + 1, lo:hi], block)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# (name, function, probability) -- the paper applies each at 80-100%
AUGMENTATIONS = [
    ("phase_shift", phase_shift, 0.9),
    ("charge_scaling", charge_scaling, 0.9),
    ("frequency_scaling", frequency_scaling, 0.9),
    ("gaussian_blur", gaussian_blur, 0.9),
    ("event_dropout", event_dropout, 0.9),
    ("white_noise", white_noise, 0.9),
    ("vertical_stripes", vertical_stripes, 0.9),
]
EXCLUSIVE_NOISE = [("noise_band", noise_band, 0.9), ("exponential_noise", exponential_noise, 0.9)]


def augment(x: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """scale multiplies every firing probability (used by the mixup variant,
    where the paper reduces probabilities from 80-100% to 10-20%)."""
    out = x.astype(np.float32, copy=True)
    for _, fn, p in AUGMENTATIONS:
        if rng.random() < p * scale:
            out = fn(out, rng)
    name, fn, p = EXCLUSIVE_NOISE[rng.integers(0, 2)]  # never both
    if rng.random() < p * scale:
        out = fn(out, rng)
    return out
