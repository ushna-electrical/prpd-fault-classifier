"""Custom PIL-level transforms for PRPD images.

Augmentation set (train split only), physically valid for a phase-resolved
partial discharge plot:

- CropToPlotArea: applied to every split, not an augmentation -- removes the
  rendered title/axis-label/colorbar chrome before anything else happens.
- CircularPhaseShift: the phase (x) axis is periodic 0-360 deg, so wrapping
  content around the horizontal axis is a genuine re-reference of the phase
  origin, not a distortion.
- ElasticJitter, RandomZoom, RandomVerticalStretch: adopted from the
  augmentation-validity analysis in the PRPD dataset paper (Zorrilla Henao et
  al., Data in Brief 2023, `Research papers/10_...md`) -- they explicitly
  tested which classical distortions preserve the phase/amplitude value a
  discharge pulse is recorded at (the information that actually encodes fault
  type) versus which destroy it. Their "acceptable" set (mild positional
  jitter, zoom, vertical stretch) is implemented here. Their "acceptable"
  set also included horizontal/vertical flips (justified in their generator-
  stator context by polarity symmetry); we deliberately do NOT adopt flips,
  since our phase axis is a literal 0-360 deg cycle with no polarity-mirror
  equivalent in our fault taxonomy -- a flip would reverse the direction of
  increasing phase, which is physically invalid here even though it wasn't
  in theirs. Their "unacceptable" transforms (shearing, heavy distortion) are
  excluded from both.
- AddGaussianNoise: mild noise added to the tensor, applied last.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance


class CropToPlotArea:
    """Crop to a fixed pixel box (see config.CROP_BOX). Applied to all splits."""

    def __init__(self, box: tuple[int, int, int, int]):
        self.box = box

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.crop(self.box)


class CircularPhaseShift:
    """Randomly roll image content around the horizontal (phase) axis.

    Wrapping is deliberate: phase 359 deg is adjacent to phase 0 deg, so a
    horizontal circular shift models sampling the same discharge pattern with
    a different phase-reference origin -- unlike a flip, which would reverse
    the direction of increasing phase.
    """

    def __init__(self, max_fraction: float):
        self.max_fraction = max_fraction

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_shift = max(1, int(w * self.max_fraction))
        shift = random.randint(-max_shift, max_shift) % w
        if shift == 0:
            return img
        wrapped_part = img.crop((w - shift, 0, w, h))
        remaining_part = img.crop((0, 0, w - shift, h))
        out = Image.new(img.mode, (w, h))
        out.paste(wrapped_part, (0, 0))
        out.paste(remaining_part, (shift, 0))
        return out


class ElasticJitter:
    """Mild, smooth per-pixel positional jitter -- approximates Augmentor's
    "Gaussian Deformation", validated in the PRPD dataset paper (#10) as
    preserving classification-relevant structure ("registered DP point
    manifestations are not lost; their distribution merely changes in
    controlled areas").

    Implemented without a scipy dependency: two independent random
    displacement fields are sampled on a coarse grid, smoothly upsampled to
    full resolution via bilinear interpolation (equivalent to a Gaussian-
    smoothed random field), then used to remap the image via nearest-
    neighbor sampling. Displacement magnitude is capped small relative to
    image size so the pattern's overall shape survives.
    """

    def __init__(self, grid_size: tuple[int, int] = (10, 7), max_displacement_fraction: float = 0.015):
        self.grid_size = grid_size
        self.max_displacement_fraction = max_displacement_fraction

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        gw, gh = self.grid_size
        max_dx = w * self.max_displacement_fraction
        max_dy = h * self.max_displacement_fraction

        rng = np.random.default_rng()
        coarse_dx = rng.uniform(-max_dx, max_dx, size=(gh, gw)).astype(np.float32)
        coarse_dy = rng.uniform(-max_dy, max_dy, size=(gh, gw)).astype(np.float32)

        dx = np.array(Image.fromarray(coarse_dx).resize((w, h), Image.BILINEAR))
        dy = np.array(Image.fromarray(coarse_dy).resize((w, h), Image.BILINEAR))

        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        src_x = np.clip(xs + dx, 0, w - 1).astype(np.int32)
        src_y = np.clip(ys + dy, 0, h - 1).astype(np.int32)

        arr = np.asarray(img)
        warped = arr[src_y, src_x]
        return Image.fromarray(warped, mode=img.mode)


class RandomZoom:
    """Random scale (zoom in/out), then crop or pad back to the original size.

    Matches the PRPD dataset paper's (#10) "scaling" operation: "zooms in on
    the images based on the factor selected" -- expands the sample area
    without altering the distribution of pulse points, per their acceptance
    criterion.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.95, 1.08)):
        self.scale_range = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        out = Image.new(img.mode, (w, h), color=0)
        if scale >= 1.0:
            # zoomed in: center-crop back down to (w, h)
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            out = resized.crop((left, top, left + w, top + h))
        else:
            # zoomed out: paste centered onto a blank canvas
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            out.paste(resized, (left, top))
        return out


class RandomVerticalStretch:
    """Random vertical stretch/compression of the magnitude (y) axis only,
    then crop or pad back to the original height. Matches the PRPD dataset
    paper's (#10) "vertical distortion": "vertical stretching increases the
    magnitude of DP without altering any pattern behavior, only expanding the
    area vertically." Width (the phase axis) is left untouched, since
    stretching phase would distort the periodic 0-360 deg structure.
    """

    def __init__(self, stretch_range: tuple[float, float] = (0.95, 1.08)):
        self.stretch_range = stretch_range

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        factor = random.uniform(*self.stretch_range)
        new_h = max(1, round(h * factor))
        resized = img.resize((w, new_h), Image.BILINEAR)

        out = Image.new(img.mode, (w, h), color=0)
        if factor >= 1.0:
            top = (new_h - h) // 2
            out = resized.crop((0, top, w, top + h))
        else:
            top = (h - new_h) // 2
            out.paste(resized, (0, top))
        return out


class MagnitudeIntensityJitter:
    """Small random brightness jitter, standing in for magnitude-axis (y)
    intensity variation. Never touches hue/saturation -- these are false-color
    scalar heatmaps, not natural images, so hue shifts would be physically
    meaningless.

    Kept available but not used in the default training augmentation set as
    of the paper-#10-aligned recipe (superseded by RandomVerticalStretch,
    which is the more literal match to their validated "vertical distortion").
    """

    def __init__(self, strength: float):
        self.strength = strength

    def __call__(self, img: Image.Image) -> Image.Image:
        factor = 1.0 + random.uniform(-self.strength, self.strength)
        return ImageEnhance.Brightness(img).enhance(factor)


class AddGaussianNoise:
    """Adds mild Gaussian noise to an already-normalized tensor (applied last,
    after ToTensor)."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std
