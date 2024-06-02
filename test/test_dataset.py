from lfdiff.datasets import SIG17_Training_Dataset, SIG17_Validation_Dataset


def test_SIG17_Training_Dataset():
    dataset = SIG17_Training_Dataset(
        root_dir="./dataset/sig17", sub_set="sig17_training_crop128_stride64_aug"
    )

    print(len(dataset))

    for k, v in dataset[0].items():
        print(k, v.shape)


def test_SIG17_Validation_Dataset():
    dataset = SIG17_Validation_Dataset(root_dir="./dataset/sig17")

    print(len(dataset))

    for k, v in dataset[0].items():
        print(k, v.shape)


def main():
    test_SIG17_Training_Dataset()
    test_SIG17_Validation_Dataset()


if __name__ == "__main__":
    main()
