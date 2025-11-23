import torch
import torch.nn as nn


class FruitNetMultiTask(nn.Module):
    """Custom CNN with exactly 3 convolutional layers and two heads.

    - Shared conv backbone: 3 Conv2d layers (conv1, conv2, conv3)
    - Two heads: fruit classification and quality classification
    """

    def __init__(self, num_fruit_classes: int = 6, num_quality_classes: int = 3):
        super().__init__()

        # Convolutional backbone (exactly 3 conv layers)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 poolings on 224x224 -> 28x28 feature maps with 128 channels
        # We'll use an adaptive pooling to reduce to a fixed size then FC
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        flattened = 128 * 7 * 7

        # Shared fully connected layer
        self.fc_shared = nn.Sequential(
            nn.Linear(flattened, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Head 1: fruit classification
        self.fc_fruit = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_fruit_classes),
        )

        # Head 2: quality classification
        self.fc_quality = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_quality_classes),
        )

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        # Adaptive pool + flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Shared FC
        shared = self.fc_shared(x)

        # Heads
        fruit_logits = self.fc_fruit(shared)
        quality_logits = self.fc_quality(shared)

        return fruit_logits, quality_logits


if __name__ == "__main__":
    # quick sanity check
    model = FruitNetMultiTask(num_fruit_classes=6, num_quality_classes=3)
    x = torch.randn(2, 3, 224, 224)
    f, q = model(x)
    print(f.shape, q.shape)
