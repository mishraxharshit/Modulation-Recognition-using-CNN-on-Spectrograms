import torch
from torch import nn


class Cnn2d(nn.Module):
    """Small 2-D CNN; adaptive pooling lets it take spectrograms (64x127) and constellations (32x32)."""
    def __init__(self, n_classes: int = 3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d((2, 2)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class Cnn1d(nn.Module):
    """1-D CNN on the symbol-rate I/Q sequence (2 x N)."""
    def __init__(self, n_classes: int = 3):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(block(2, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool1d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))
