import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
from models import LPENet, DHRNet
from utils import HDRDataset, tonemap


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor()
    ])

    # Create datasets
    ldr_dir = './LDR'
    hdr_dir = './HDR'
    dataset = HDRDataset(ldr_dir, hdr_dir, transform=transform)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)


    # Assume all model definitions and dataset class definitions are already provided as above.
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    lpe_net = LPENet().to(device)
    dhr_net = DHRNet().to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(list(lpe_net.parameters()) + list(dhr_net.parameters()), lr=0.001)

    # Number of epochs
    num_epochs = 50

    # Training loop
    for epoch in range(num_epochs):
        for i, (ldr, hdr) in enumerate(data_loader):
            ldr, hdr = ldr.to(device), hdr.to(device)
            
            optimizer.zero_grad()
            
            # Apply tonemapping
            ldr_tonemapped = tonemap(ldr)
            
            # Concatenate LDR and tonemapped LDR images
            input_tensor = torch.cat([ldr, ldr_tonemapped], dim=1)
            
            # Forward pass through LPENet
            lpr = lpe_net(input_tensor)
            
            # Forward pass through DHRNet
            hdr_pred = dhr_net(input_tensor, lpr)
            
            # Compute loss
            loss = criterion(hdr_pred, hdr)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print loss every 100 steps
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    
    torch.save(lpe_net.state_dict(), f'model_lpe_epoch_{epoch+1}.pth')
    torch.save(dhr_net.state_dict(), f'model_dhr_epoch_{epoch+1}.pth')
    print(f'Saved model weights for epoch {epoch+1}')

