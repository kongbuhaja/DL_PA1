from torch import nn
from torchvision.models.alexnet import AlexNet as Alex
import torch

class AlexNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, cfg.CONV_SIZE, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(cfg.CONV_SIZE, cfg.CONV_SIZE*3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(cfg.CONV_SIZE*3, cfg.CONV_SIZE*6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.CONV_SIZE*6, cfg.CONV_SIZE*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.CONV_SIZE*4, cfg.CONV_SIZE*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(cfg.CONV_SIZE*4 * 3 * 3, cfg.MLP_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(cfg.MLP_SIZE, cfg.MLP_SIZE),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.MLP_SIZE, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x