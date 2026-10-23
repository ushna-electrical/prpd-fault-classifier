"""Image preprocessing must match what the models were trained on. These tests
pin the exact crop boxes and output shapes used by test_model.FinalModel's two
branches (prototypical network, ResNet-18) against known-correct values, and
exercise the real preprocessing functions -- not a reimplementation of them --
on a real dataset image.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from app.inference import CODE_ROOT
from prpd_fewshot import config as fewshot_config
from prpd_fewshot.dataset import build_transform, load_manifest_rows
from prpd_resnet import config as resnet_config
from prpd_resnet.dataset import MATRIX_SIZE, to_tensor
from prpd_resnet.represent import noise_mask, png_to_intensity

TEST_MANIFEST = CODE_ROOT / "manifests_sensor_2_merged23" / "test_manifest.csv"


@pytest.fixture(scope="module")
def sample_image_path() -> Path:
    rows = load_manifest_rows(TEST_MANIFEST)
    row = next(r for r in rows if r["fault_type"] == "fault_4")
    path = CODE_ROOT / row["filepath"]
    assert path.exists(), f"fixture image missing: {path}"
    return path


def test_prototypical_crop_box_unchanged():
    # Regression guard: test_model.py's prototypical branch uses this value via
    # prpd_fewshot.dataset.build_transform. Changing it changes what every
    # trained prototype/encoder weight was fit against.
    assert fewshot_config.CROP_BOX == (100, 48, 1450, 968)
    assert fewshot_config.IMAGE_SIZE == 128


def test_resnet_crop_box_unchanged():
    # Deliberately different from the prototypical branch's crop box (two
    # separate pipelines feeding two separately-trained models) -- see
    # prpd_resnet/config.py's own comment. Pinned here so the two can't drift
    # into each other by accident.
    assert resnet_config.CROP_BOX == (135, 62, 1438, 955)
    assert resnet_config.IMAGE_SIZE == 224
    assert MATRIX_SIZE == (360, 512)


def test_training_images_are_1800x1050(sample_image_path):
    # FinalModel.expected_size (from meta.json) must match what training
    # images actually look like, or the Classify page's size-mismatch warning
    # is meaningless.
    with Image.open(sample_image_path) as img:
        assert img.size == (1800, 1050)


def test_prototypical_transform_has_no_augmentation_at_inference():
    # test_model.py always calls build_transform(augment=False); augmented
    # steps (phase shift, elastic jitter, zoom, stretch) must not sneak in.
    tf = build_transform(augment=False)
    step_names = [type(s).__name__ for s in tf.transforms]
    for augmenting_step in ("CircularPhaseShift", "ElasticJitter", "RandomZoom", "RandomVerticalStretch"):
        assert augmenting_step not in step_names


def test_prototypical_pipeline_output_shape(sample_image_path):
    tf = build_transform(augment=False)
    tensor = tf(Image.open(sample_image_path).convert("RGB"))
    assert tensor.shape == (3, fewshot_config.IMAGE_SIZE, fewshot_config.IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_resnet_pipeline_output_shape(sample_image_path):
    matrix = png_to_intensity(str(sample_image_path), out_size=MATRIX_SIZE)
    assert matrix.shape == (MATRIX_SIZE[1], MATRIX_SIZE[0])  # (height, width)
    mask = noise_mask(matrix)
    assert mask.shape == matrix.shape
    assert mask.dtype == bool

    import numpy as np
    masked = np.where(mask, matrix, 0.0)
    tensor = to_tensor(masked)
    assert tensor.shape == (3, resnet_config.IMAGE_SIZE, resnet_config.IMAGE_SIZE)
