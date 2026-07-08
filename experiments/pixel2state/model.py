import torch
from torch import nn

class PixelToGridNet(nn.Module):
    """
    Predicts an 8x10 semantic grid from a 128x160 RGB observation.

    Input:
        x: float tensor, shape (B, 3, 128, 160), values in [0, 1]

    Output:
        logits: float tensor, shape (B, num_tile_classes, 8, 10)
    """

    def __init__(self, num_tile_classes: int = 12):
        super().__init__()
        self.num_tile_classes = num_tile_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),   # 64x80
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x40
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x20
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 8x10
            nn.ReLU(inplace=True),
        )

        self.grid_head = nn.Conv2d(128, num_tile_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.grid_head(features)

def make_model(num_tile_classes: int = 12) -> PixelToGridNet:
    return PixelToGridNet(num_tile_classes=num_tile_classes)