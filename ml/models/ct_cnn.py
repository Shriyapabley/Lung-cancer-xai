# ml/models/ct_cnn.py

import torch
import torch.nn as nn
from torchvision import models


class CTCNNModel(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        # Use ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Adapt for 1-channel CT (grayscale)
        # Original conv1: (in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        old_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None,
        )

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
