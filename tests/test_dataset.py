import os

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
    assert mask.max() == 1, "Mask bust be binary"
    assert image.shape[1:] == mask.shape[1:], f"Mask and image must have same w, h"


def test_inference_dataset():

    root_dir = "tests/data/dataset/test_dataset/train/images"
    dataset = InferenceDataset(root_dir=root_dir)
    image, image_path = dataset[0]
    assert len(image.shape) == 3, f"Image must be RGB"
    assert os.path.exists(image_path), f"{image_path} does not exist."
