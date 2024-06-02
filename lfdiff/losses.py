from torch import nn
from torchvision import models


class ReconstructionLoss(nn.Module):
    def __init__(self, perceptual_weight=1e-2):
        super(ReconstructionLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.vgg = (
            models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            .features[:28]
            .eval()
        )  # Use up to the fourth maxpooling layer
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # L1 Loss
        l1_loss = self.l1_loss(pred, target)

        # Perceptual Loss
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        perceptual_loss = self.l1_loss(pred_features, target_features)

        return l1_loss + perceptual_loss * perceptual_loss
