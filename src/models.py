from dataclasses import dataclass

import torch
from torch import nn

from .preprocessing import CLASS_NAMES


@dataclass(frozen=True)
class CNNConfig:
    channels: tuple[int, int, int] = (8, 16, 32)


@dataclass(frozen=True)
class CRNNConfig:
    channels: tuple[int, int, int] = (3, 6, 12)
    projection_size: int = 24
    gru_hidden: int = 12


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool | tuple[int, int]) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2 if pool is True else pool))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class AudioCNN(nn.Module):
    def __init__(self, config: CNNConfig = CNNConfig(), num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        first, second, third = config.channels
        self.features = nn.Sequential(
            ConvBlock(1, first, pool=True),
            ConvBlock(first, second, pool=True),
            ConvBlock(second, third, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(third, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)).flatten(1))


class AudioCRNN(nn.Module):
    def __init__(self, config: CRNNConfig = CRNNConfig(), num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        first, second, third = config.channels
        self.cnn = nn.Sequential(
            ConvBlock(1, first, pool=(2, 2)),
            ConvBlock(first, second, pool=(2, 2)),
            ConvBlock(second, third, pool=(2, 1)),
        )
        self.projection = nn.Sequential(
            nn.Linear(third * 8, config.projection_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
        )
        self.gru = nn.GRU(
            input_size=config.projection_size,
            hidden_size=config.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(2 * config.gru_hidden, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.cnn(inputs).permute(0, 3, 1, 2).flatten(start_dim=2)
        _, hidden = self.gru(self.projection(sequence))
        return self.classifier(torch.cat([hidden[-2], hidden[-1]], dim=1))


MODEL_NAMES = ("cnn", "crnn")


def build_model(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "cnn":
        return AudioCNN()
    if normalized == "crnn":
        return AudioCRNN()
    raise ValueError(f"Unknown model: {name}. Expected one of {MODEL_NAMES}")
