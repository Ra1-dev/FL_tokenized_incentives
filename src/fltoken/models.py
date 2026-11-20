"""The unpadded LeNet model used in the reproduction protocol."""

from torch import Tensor, nn


class LeNet5(nn.Module):
    """Map normalized grayscale images of shape (N, 1, 28, 28) to logits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)
