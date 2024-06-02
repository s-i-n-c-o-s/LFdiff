from typing import Any
from lfdiff.models import AlignmentModule, LPENet, DHRNet
import lightning as L
import torch
import torch.nn as nn


class LFDiffPretrainModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.alignment_module = AlignmentModule(in_channels=6, feature_channels=60)
        self.pixelunshuffle = nn.PixelUnshuffle(4)
        self.lpenet = LPENet(in_channels=6 * 4**2, out_channels=3, feature_channels=128)
        self.dhrnet = DHRNet(in_channels=60, prior_channels=3, num_layers=[3, 3, 3])

        self.out_conv = nn.Conv2d(60, 3, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3, gt):
        z = self.pixelunshuffle(gt)
        z = self.lpenet(z)

        f0 = self.alignment_module(x1, x2, x3)

        out = self.out_conv(self.dhrnet(f0, z))

        return out


if __name__ == "__main__":
    model = LFDiffPretrainModule().cuda()
    print(model)

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # test dummy input
    dummy = torch.randn(1, 24, 128, 128).cuda()
    print(model(dummy[:, 0:6], dummy[:, 6:12], dummy[:, 12:18], dummy[:, 18:24]).shape)
