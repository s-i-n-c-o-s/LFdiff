from lfdiff.datasets import SIG17_Training_Dataset


def test_SIG17_Training_Dataset():
    dataset = SIG17_Training_Dataset(
        root_dir=f"./dataset/sig17", sub_set="sig17_training_crop128_stride64_aug"
    )

    print(len(dataset))

    print(dataset[0].shape)


def main():
    test_SIG17_Training_Dataset()


if __name__ == "__main__":
    main()
