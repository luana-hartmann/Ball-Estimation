import torch.nn as nn


class BallPatchCNN(nn.Module):
    """
    Small CNN for binary classification of 32x32 candidate patches: is
    this crop the ball, or something else (paddle edge, hand, shadow)?

    Deliberately small -- the input is tiny and the decision is simple
    compared to general image classification, so a handful of conv
    layers is enough. Keeps training fast even on CPU, and stays
    relevant to the project brief's interest in eventual embedded/phone
    deployment (a huge network would work against that goal).
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32 -> 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # -> 64x1x1
        )
        self.classifier = nn.Linear(64, 2)  # 2 logits: [not-ball, ball]

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
