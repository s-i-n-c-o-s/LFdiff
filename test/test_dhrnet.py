from lfdiff.models import DHRNet

import torch


def test_dhrnet():
    model = DHRNet(in_channels=60, prior_channels=3, num_layers=[3, 3, 3])
    
    # Print the total number of parameters in the model
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.randn(1, 60, 128, 128)
    z = torch.randn(1, 3, 128 // 4, 128 // 4)
    y = model(x, z)
    assert y.shape == x.shape


def main():
    test_dhrnet()


if __name__ == "__main__":
    main()
