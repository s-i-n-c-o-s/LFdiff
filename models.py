import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

# LPENet
class LPENet(nn.Module):
    def __init__(self, in_channels=6, num_residual_blocks=5):
        super(LPENet, self).__init__()
        layers = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(64))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.lpe_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.lpe_net(x)

# Feature Refinement Module
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FeatureRefinementModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        w = self.avgpool(out)
        w = torch.flatten(w, 1)
        w = self.fc(w).unsqueeze(2).unsqueeze(3)
        out = out * w
        out = self.conv2(out)
        return out + x

# Prior Integration Module
class PriorIntegrationModule(nn.Module):
    def __init__(self, in_channels, prior_channels):
        super(PriorIntegrationModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(prior_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(prior_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, prior):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(prior).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(prior).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# DHRNet
class DHRNet(nn.Module):
    def __init__(self, in_channels=6, num_blocks=5):
        super(DHRNet, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([FeatureRefinementModule(64) for _ in range(num_blocks)])
        self.pims = nn.ModuleList([PriorIntegrationModule(64, 64) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, prior):
        x = F.relu(self.initial_conv(x))
        for block, pim in zip(self.blocks, self.pims):
            x = block(x)
            x = pim(x, prior)
        x = self.final_conv(x)
        return x
