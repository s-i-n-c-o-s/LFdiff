import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image

# Tonemapping function
def tonemap(image, gamma=5000):
    gamma = torch.tensor(gamma, dtype=torch.float)
    return torch.log(1 + gamma ** image) / torch.log( 1 + gamma)

# Dataset Class
class HDRDataset(Dataset):
    def __init__(self, ldr_dir, hdr_dir, transform=None):
        self.ldr_dir = ldr_dir
        self.hdr_dir = hdr_dir
        self.transform = transform
        self.ldr_images = sorted(os.listdir(ldr_dir))
        self.hdr_images = sorted(os.listdir(hdr_dir))

    def __len__(self):
        return len(self.ldr_images)

    def __getitem__(self, idx):
        ldr_path = os.path.join(self.ldr_dir, self.ldr_images[idx])
        hdr_path = os.path.join(self.hdr_dir, self.hdr_images[idx])
        
        ldr_image = Image.open(ldr_path).convert('RGB')
        hdr_image = Image.open(hdr_path).convert('RGB')
        
        if self.transform:
            ldr_image = self.transform(ldr_image)
            hdr_image = self.transform(hdr_image)
        
        return ldr_image, hdr_image


    # Your main code here

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def save_image(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)
