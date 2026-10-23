"""ResNet-18 (ImageNet) + Mück's classification head, and his layer-wise
learning-rate scaling (report sections 4.4 and 4.4.2).

Head: the final feature tensor is pooled channel-wise by BOTH average and max
pooling; the two 512-vectors are concatenated to 1024, then a 256-unit ReLU
hidden layer with 20% dropout, then one output per class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from . import config


class PRPDResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, hidden: int = config.HEAD_HIDDEN, dropout: float = config.DROPOUT):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.conv1, self.bn1, self.relu, self.maxpool = backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.head = nn.Sequential(nn.Linear(512 * 2, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)  # 512 avg + 512 max
        return self.head(pooled)


def layerwise_param_groups(model: nn.Module, base_lr: float, scales: dict | None = None) -> list[dict]:
    """One parameter group per backbone stage, each with its own LR multiplier
    (paper table 4.3: conv1/layer1 0.001, layer2 0.01, layer3 0.1, layer4 0.5,
    head 1.0) -- early layers keep their ImageNet features, the head trains at
    full rate."""
    scales = scales or config.LAYER_LR_SCALE
    groups = {name: [] for name in scales}
    for pname, p in model.named_parameters():
        stage = pname.split(".")[0]
        key = stage if stage in scales else "head"
        groups[key].append(p)
    return [{"params": ps, "lr": base_lr * scales[name], "name": name} for name, ps in groups.items() if ps]
