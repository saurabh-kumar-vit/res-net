import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                            nn.BatchNorm2d(out_channels, eps=0.001),
                            nn.ReLU(inplace=True)
                        )

    def forward(self, x):
        x = self.features(x)
        return x

class BasicConv2d_no_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d_no_ReLU, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
                            nn.BatchNorm2d(out_channels, eps=0.001)
                        )

    def forward(self, x):
        x = self.features(x)
        return x

class BasicResidualBlock(nn.Module):
    def __init__(self, num_ftrs):
        super(BasicResidualBlock, self).__init__()
        self.features = nn.Sequential(
                            BasicConv2d(num_ftrs, num_ftrs, kernel_size=3, padding=1),
                            BasicConv2d_no_ReLU(num_ftrs, num_ftrs, kernel_size=3, padding=1)
                        )

    def forward(self, x):
        y = self.features(x)
        residue = F.relu(y + x, inplace=True)

        return residue

class PoolingResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PoolingResidualBlock, self).__init__()
        self.features = nn.Sequential(
                            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
                            BasicConv2d_no_ReLU(out_channels, out_channels, kernel_size=3, padding=1)
                        )

        self.projection = BasicConv2d_no_ReLU(in_channels, out_channels, kernel_size=1, stride=2)


    def forward(self, x):
        y = self.features(x)
        x_projection = self.projection(x)

        residue = F.relu(y + x_projection, inplace=True)

        return residue

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(
                            BasicConv2d(3, 64, kernel_size=7, padding=3, stride=2),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                            BasicResidualBlock(64),
                            BasicResidualBlock(64),
                            PoolingResidualBlock(64, 128),
                            BasicResidualBlock(128),
                            PoolingResidualBlock(128, 256),
                            BasicResidualBlock(256),
                            PoolingResidualBlock(256, 512),
                            BasicResidualBlock(512)
                        )

        self.fc = nn.Linear(512, 2)
    def forward(self, x):
        x = self.features(x)

        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
