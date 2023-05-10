import os
import sys
from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from saldet.io import read_mask, read_rgb


class SaliencyDataset(Dataset):
    """Saliency Detection dataset (with images and masks dir)

    Args:
        root_dir (str): dataset dir with images and masks dirs in it.
        train (bool, optional): if True looks for a train folder, else for a val folder. Defaults to True.
        transform (Callable, optional): data augmentations. Defaults to None.
        max_samples (int, optional): maximum number of samples to load. Defaults to None.
    """

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Callable = None,
        max_samples: int = None,
    ) -> None:

        super().__init__()

        root_dir = os.path.join(root_dir, "train" if train else "val")

        assert os.path.exists(root_dir), f"{root_dir} does not exists."
        assert "images" in os.listdir(
            root_dir
        ), f"{root_dir} must have 'images' folder in it."
        assert "masks" in os.listdir(
            root_dir
        ), f"{root_dir} must have 'masks' folder in it."

        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")

        self.images, self.masks = self._load_data()

        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]

        self.transform = transform

        print(f"#" * 12 + f" {'train' if train else 'val'} dataset recap " + "#" * 12)
        print(f"\t> Root Dir: {root_dir}")
        print(f"\t> Images: {len(self.images)}")
        print(f"\t> Masks: {len(self.masks)}")
        print(f"#" * 40)
        print(f"\n")

    def _load_data(self) -> Tuple[List[str], List[str]]:
        """loads data from images dir and masks dir (checks if an image has its mask)

        Returns:
            Tuple[List[str], List[str]]: images and masks paths
        """

        _images = [f for f in os.listdir(self.images_dir) if not f.startswith(".")]
        _masks = [f for f in os.listdir(self.masks_dir) if not f.startswith(".")]

        images, masks = [], []
        for image in _images:
            image_no_ext = os.path.splitext(image)[0]
            for mask in _masks:
                if os.path.splitext(mask)[0] == image_no_ext:
                    images.append(os.path.join(self.images_dir, image))
                    masks.append(os.path.join(self.masks_dir, mask))
                    _masks.remove(mask)
                    break

        if len(_images) != len(images):
            print(
                f"[WARNING] Found {len(_images)-len(images)} images with no masks. Removing them from dataset."
            )
        return images, masks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image, mask) with augmentations applied at given indec

        Args:
            index (int): index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, mask)
        """

        try:
            image = read_rgb(self.images[index])
        except Exception as e:
            print(f"Found error while loading image: {self.images[index]}")
            print(f"Exception {e}")
            sys.exit(0)
        try:
            mask = read_mask(self.masks[index])

        except Exception as e:
            print(f"Found error while loading mask: {self.masks[index]}")
            print(f"Exception {e}")
            sys.exit(0)

        if self.transform:
            image, mask = self.transform(image=image, mask=mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.images)
