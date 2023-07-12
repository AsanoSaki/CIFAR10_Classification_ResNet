from torch import nn
from torch.nn import functional as F
from torchvision import models

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet34(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 4))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 6))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 3))
        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, output_channels))

    def forward(self, X):
        Y = self.net(X)
        return Y

class PretrainedResNet34(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(512, output_channels)

    def forward(self, X):
        return self.net(X)
