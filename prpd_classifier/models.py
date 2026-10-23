"""Backbone factory: ImageNet-pretrained resnet50 / efficientnet_b0 /
mobilenetv3_large, each with its classification head replaced for
config.NUM_CLASSES.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
)


def freeze_backbone_features(model: nn.Module, freeze_until_block: int) -> tuple[int, int]:
    """Freeze model.features[:freeze_until_block] (requires_grad=False), leaving
    model.features[freeze_until_block:] and the classifier head trainable.

    Only applies to architectures exposing a `.features` Sequential (mobilenetv3,
    efficientnet) -- see the Stage 1 baseline's overfitting diagnosis: fully
    fine-tuning all of MobileNetV3Large's ~3M backbone parameters on ~1900
    single-sensor images overfit to the majority classes within one epoch
    (val_loss climbed every subsequent epoch while train_loss kept dropping),
    and a 6x lower learning rate only softened, not fixed, that pattern --
    pointing at capacity-vs-data-size, not step size, as the actual cause.

    Returns (n_frozen_params, n_trainable_params) for a build-time sanity log.
    """
    if not hasattr(model, "features"):
        raise NotImplementedError(
            f"freeze_backbone_features only supports architectures with a `.features` "
            f"Sequential (mobilenetv3, efficientnet); got {type(model).__name__}."
        )

    n_blocks = len(model.features)
    if not (0 <= freeze_until_block <= n_blocks):
        raise ValueError(f"freeze_until_block must be in [0, {n_blocks}], got {freeze_until_block}")

    for block in model.features[:freeze_until_block]:
        for p in block.parameters():
            p.requires_grad = False

    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_frozen, n_trainable


def build_model(backbone_name: str, num_classes: int, freeze_backbone_until: int | None = None) -> nn.Module:
    if backbone_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone_name == "mobilenetv3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name!r}")

    if freeze_backbone_until:
        n_frozen, n_trainable = freeze_backbone_features(model, freeze_backbone_until)
        print(f"  froze features[:{freeze_backbone_until}]: {n_frozen:,} params frozen, {n_trainable:,} trainable")

    return model
