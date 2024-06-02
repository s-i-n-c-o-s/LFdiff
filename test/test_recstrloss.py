from lfdiff.losses import ReconstructionLoss
import torch


def test_recstrloss():
    recstrloss = ReconstructionLoss()

    # Test forward
    pred = torch.randn(2, 3, 128, 128)
    target = torch.randn(2, 3, 128, 128)

    loss = recstrloss(pred, target)

    assert loss.shape == torch.Size([])

def main():
    test_recstrloss()

if __name__ == '__main__':
    main()