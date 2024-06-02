from typing import Any

from torch.utils.data import DataLoader
from lfdiff.models import AlignmentModule, LPENet, DHRNet, mu_tonemap
from lfdiff.losses import ReconstructionLoss
from lfdiff.datasets import SIG17_Training_Dataset, SIG17_Validation_Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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

        self.loss = ReconstructionLoss()

    def forward(self, x1, x2, x3, gt):
        z = self.pixelunshuffle(gt)
        z = self.lpenet(z)

        f0 = self.alignment_module(x1, x2, x3)

        out = self.out_conv(self.dhrnet(f0, z))

        return out

    def training_step(self, batch):
        pred = self(
            batch["x1"],
            batch["x2"],
            batch["x3"],
            torch.cat([batch["gt"], batch["gttm"]], dim=1),
        )

        # clamp to 0-1 to avoid nan loss
        pred = mu_tonemap(pred.clamp(0, 1))

        loss = self.loss(pred, batch["gttm"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        pred = self(
            batch["x1"],
            batch["x2"],
            batch["x3"],
            torch.cat([batch["gt"], batch["gttm"]], dim=1),
        )

        # clamp to 0-1 to avoid nan loss
        pred = mu_tonemap(pred.clamp(0, 1))

        loss = self.loss(pred, batch["gttm"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4
        )
        return optimizer


def collator(samples):
    return {
        "x1": torch.stack([s["x1"] for s in samples]),
        "x2": torch.stack([s["x2"] for s in samples]),
        "x3": torch.stack([s["x3"] for s in samples]),
        "gt": torch.stack([s["gt"] for s in samples]),
        "gttm": torch.stack([s["gttm"] for s in samples]),
    }


if __name__ == "__main__":
    model = LFDiffPretrainModule()
    train_dataset = SIG17_Training_Dataset(
        "./dataset/sig17", sub_set="sig17_training_crop128_stride64_aug"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, collate_fn=collator, num_workers=32, shuffle=True
    )

    valid_dataset = SIG17_Validation_Dataset("./dataset/sig17")
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=4, collate_fn=collator, num_workers=32, shuffle=False
    )
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                save_last="link", every_n_epochs=1, enable_version_counter=True
            )
        ],
        # precision="16-mixed"
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
