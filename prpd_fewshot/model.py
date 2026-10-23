"""Siamese network: shared-weight CNN encoder + learned distance-to-logit head.

Architecture follows Tuyet-Doan et al. (Sensors 2020, see
Research papers/05_OneShotSiamese-GIS_Sensors2020.md): four conv+pool+
batchnorm+dropout blocks feeding a dense embedding layer, deliberately small
and shallow (their stated reason: reduce parameters to train and avoid
overfitting on small datasets -- the same problem we hit fine-tuning all of
MobileNetV3Large on a single sensor's data in Stage 3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """Maps a (3, IMAGE_SIZE, IMAGE_SIZE) image to an `embedding_dim` vector."""

    def __init__(self, embedding_dim: int = 64, image_size: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            flat_size = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Linear(flat_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward -- gives this encoder the same `.embed()`
        interface as SiameseNetwork, so prototype-based evaluate.py code can
        take either model interchangeably."""
        return self.forward(x)


class SiameseNetwork(nn.Module):
    """Wraps a shared SiameseEncoder with a learned L2-distance-to-logit head.

    logit = -softplus(alpha) * ||e1 - e2||_2 + beta
    softplus(alpha) keeps the distance scale positive (larger distance -> more
    negative logit -> lower similarity probability under BCEWithLogitsLoss),
    while alpha/beta are learned rather than fixed, matching the paper's
    "sigmoid(w * distance + b)" formulation.
    """

    def __init__(self, embedding_dim: int = 64, image_size: int = 128):
        super().__init__()
        self.encoder = SiameseEncoder(embedding_dim, image_size)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        distance = torch.norm(e1 - e2, p=2, dim=1)
        logit = -F.softplus(self.alpha) * distance + self.beta
        return logit
