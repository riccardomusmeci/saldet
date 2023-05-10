from saldet.dataset import InferenceDataset, SaliencyDataset
from saldet.transform import SaliencyTransform


def test_train_dataset():

    dataset = SaliencyDataset(
        root_dir="tests/data/dataset/test_dataset",
        train=True,
        transform=SaliencyTransform(train=True, input_size=224),
    )

    image, mask = dataset[0]
    assert image.shape[1:] == mask.shape[1:], "Image and mask must be of same dim."
    assert (
        mask.max() == 1
    ), f"Mask must be a binary mask. Current mask max value {mask.max()}"


def test_train_dataset():

    dataset = SaliencyDataset(
        root_dir="tests/data/dataset/test_dataset",
        train=True,
        transform=SaliencyTransform(train=True, input_size=224),
    )

    image, mask = dataset[0]
    assert image.shape[1:] == mask.shape[1:]
