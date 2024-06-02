from lfdiff.models import LPENet

import torch


def test_lpenet():
    module = LPENet()

    # Print the total number of parameters
    print(f"Total number of parameters: {sum(p.numel() for p in module.parameters())}")

    dummy_input = torch.randn(1, 6, 128, 128)

    output = module(dummy_input)

    assert output.shape == (1, 3, 128, 128)

def main():
    test_lpenet()

if __name__ == '__main__':
    main()