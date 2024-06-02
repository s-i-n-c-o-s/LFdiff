import torch
import torch.nn as nn
import torchvision.models as models 
import torch.nn.functional as F
 
# DM
class DiffusionProcess:
    def __init__(self, num_steps, beta_start, beta_end, device='cuda'):
        self.num_steps = num_steps
        self.beta = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.device = device

    def sample_noise(self, x_0, t):
        noise = torch.randn_like(x_0).to(self.device)
        alpha_bar_t = self.alpha_bar[t]
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    def reverse_process(self, x_t, t, model):
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        pred_noise = model(x_t, t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        variance = beta_t
        return mean + torch.sqrt(variance) * torch.randn_like(x_t).to(self.device)

    def ddim_reverse_process(self, x_t, S, model):
        for t in reversed(range(S)):
            x_t = self.reverse_process(x_t, t, model)
        return x_t

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 32 * 32, 3 * 32 * 32)

    def forward(self, x, t):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 3, 32, 32)
        return x



# Reconstruction Loss
class ReconstructionLoss(nn.Module):
    def __init__(self, perceptual_weight : float = 1e-2):
        super(ReconstructionLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.vgg = models.vgg19(pretrained=True).features[:16].eval()  # Use up to the third convolutional block
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # L1 Loss
        l1_loss = self.l1_loss(pred, target)
        
        # Perceptual Loss
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        perceptual_loss = self.l1_loss(pred_features, target_features)
        
        return l1_loss + perceptual_loss * self.perceptual_weight


