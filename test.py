import torch
import torchvision.transforms as transforms
from models import LPENet, DHRNet
from utils import tonemap, load_image, save_image

def test_model(ldr_path, model_lpe, model_dhr, output_path, device):
    ldr_image = load_image(ldr_path, transforms).unsqueeze(0).to(device)
    ldr_tonemapped = tonemap(ldr_image)
    input_tensor = torch.cat([ldr_image, ldr_tonemapped], dim=1)

    with torch.no_grad():
        lpr = model_lpe(input_tensor)
        hdr_pred = model_dhr(input_tensor, lpr)
    
    save_image(hdr_pred, output_path)
    print(f'Saved HDR image to {output_path}')

if __name__ == '__main__':
    # Paths
    ldr_path = './test/_MDF0079.jpg'
    model_lpe_path = './model_lpe.pth'
    model_dhr_path = './model_dhr.pth'
    output_path = './output/hdr_output.jpg'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    model_lpe = LPENet().to(device)
    model_dhr = DHRNet().to(device)

    # Load trained model weights
    model_lpe.load_state_dict(torch.load(model_lpe_path))
    model_dhr.load_state_dict(torch.load(model_dhr_path))

    # Test the model
    test_model(ldr_path, model_lpe, model_dhr, output_path, device)