import os
import time
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from unet import DiffusionUNet
import math
from lfdiff.models import LPENet
import itertools
from lfdiff.datasets import SIG17_Training_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

def data_transform(X):
    return X

def inverse_data_transform(X):
    return X

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

class l1_loss_mu(nn.Module):
    def __init__(self, mu=5000):
        super(l1_loss_mu, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        return nn.L1Loss()(pred, label)

class JointReconPerceptualLossAfterMuLaw(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super(JointReconPerceptualLossAfterMuLaw, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = l1_loss_mu()

    def forward(self, input, target):
        loss_recon = self.loss_recon(input, target)
        loss = loss_recon
        return loss

def noise_estimation_loss(model, x0, t, e, b, feature):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 18:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :18, :, :], x], dim=1), t.float(), feature)
    x0_t = (x - output * (1 - a).sqrt()) / a.sqrt()
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + nn.L1Loss()(x0_t, x0[:, 18:, :, :])

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # Initialize LPENet model
        self.lpenet = LPENet()
        self.lpenet.to(self.device)
        self.lpenet = torch.nn.DataParallel(self.lpenet)

        num_params = sum(param.numel() for param in self.model.parameters())
        print(num_params)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = torch.optim.Adam(itertools.chain(self.model.parameters(), self.lpenet.parameters()), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, "cpu")
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['ddpm_state_dict'], strict=True)
        self.lpenet.load_state_dict(checkpoint['cnn_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def load_ddm_ckpt_test(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, "cpu")
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['ddpm_state_dict'], strict=True)
        self.lpenet.load_state_dict(checkpoint['cnn_state_dict'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        _, val_loader = DATASET.get_loaders()
        train_dataset = SIG17_Training_Dataset(root_dir='./data', sub_set='sig17_training_crop128_stride64_aug', is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            with tqdm(total=train_loader.__len__()) as pbar:
                for i, x in enumerate(train_loader):
                    x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                    n = x.size(0)
                    data_time += time.time() - data_start
                    self.model.train()
                    self.step += 1

                    x = x.to(self.device)
                    x = data_transform(x)

                    attenfeature = self.lpenet(x[:, :6, :, :], x[:, 6:12, :, :], x[:, 12:18, :, :])

                    e = torch.randn_like(x[:, 18:, :, :])
                    b = self.betas

                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                    loss = noise_estimation_loss(self.model, x, t, e, b, attenfeature)

                    if self.step % 1000 == 0:
                        print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.ema_helper.update(self.model)
                    data_start = time.time()

                    pbar.set_postfix(loss=float(loss.cpu()), epoch=epoch)
                    pbar.update(1)

                    if self.step % self.config.training.validation_freq == 0:
                        self.model.eval()
                        self.sample_validation_patches(val_loader, self.step, self.lpenet)

                    if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'ddpm_state_dict': self.model.state_dict(),
                            'cnn_state_dict': self.lpenet.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm'+str(self.step)))

    def sample_image(self, x_cond, x, attenfeature=None, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.lpenet, self.betas, eta=0., corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, attenfeature, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs

    def sample_validation_patches(self, val_loader, step, lpenet):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :18, :, :].to(self.device)
            x_cond = data_transform(x_cond)

            attenfeature = lpenet(x_cond[:, :6, :, :], x[:, 6:12, :, :], x[:, 12:18, :, :])

            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x, attenfeature)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.logging.save_image_png(x_cond[i, 0:3, :, :], os.path.join(image_folder, str(step), f"{i}_cond_s.png"))
                utils.logging.save_image_png(x_cond[i, 6:9, :, :], os.path.join(image_folder, str(step), f"{i}_cond_m.png"))
                utils.logging.save_image_png(x_cond[i, 12:15, :, :], os.path.join(image_folder, str(step), f"{i}_cond_l.png"))
                utils.logging.save_image_png(x[i], os.path.join(image_folder, str(step), f"{i}.png"))
