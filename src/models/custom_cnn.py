import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_filters=None,
        dropout: float = 0.4,
        fc_units: int = 512,
    ) -> None:
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128, 256]

        layers = []
        current_channels = in_channels
        for filters in conv_filters:
            layers.extend(
                [
                    nn.Conv2d(current_channels, filters, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_channels = filters

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_filters[-1], fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


