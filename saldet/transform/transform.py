from typing import Tuple, Union

import albumentations as A
import numpy as np
import PIL
import torch
from PIL import Image


class SaliencyTransform:
    """Saliency based transformation

    Args:
        train (bool): train/val mode.
        input_size (Union[int, list, tuple]): image input size.
        interpolation (int, optional): resize interpolation. Defaults to 3.
        mean (list, optional): normalization mean. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): normalization std. Defaults to [0.229, 0.224, 0.225].
        random_crop_p (float, optional): random crop probability. Defaults to 0.1.
        h_flip_p (float, optional): horizontal flip probability. Defaults to 0.1.
        v_flip_p (float, optional): vertical flip probability. Defaults to 0.1.
    """

    def __init__(
        self,
        train: bool,
        input_size: Union[int, list, tuple],
        interpolation: int = 3,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        random_crop_p: float = 0.1,
        h_flip_p: float = 0.1,
        v_flip_p: float = 0.1,
    ) -> None:

        if isinstance(input_size, tuple) or isinstance(input_size, list):
            height = input_size[0]
            width = input_size[1]
        else:
            height = input_size
            width = input_size

        if train:
            self.transform = A.Compose(
                [
                    A.Resize(
                        height=height,
                        width=width,
                        interpolation=interpolation,
                        always_apply=True,
                    ),
                    A.RandomResizedCrop(height=height, width=width, p=random_crop_p),
                    A.HorizontalFlip(p=h_flip_p),
                    A.VerticalFlip(p=v_flip_p),
                    A.Normalize(mean=mean, std=std),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(
                        height=height,
                        width=width,
                        interpolation=interpolation,
                        always_apply=True,
                    ),
                    A.Normalize(mean=mean, std=std),
                ]
            )

    def __call__(
        self,
        image: Union[np.array, PIL.Image.Image],
        mask: Union[np.array, PIL.Image.Image] = None,
    ) -> Tuple[np.array, np.array]:
        """Apply augmentations

        Args:
            img (Union[np.array, PIL.Image.Image]): input image
            mask (Union[np.array, PIL.Image.Image], optional): input mask. Defaults to None.

        Returns:
            Tuple[np.array, np.array, np.array]: vanilla img (resize + normalize), view 1, view 2
        """

        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(mask, Image.Image):
            image = np.array(image)

        if mask is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        else:
            image = self.transform(image=image)["image"]
            mask = None

        image = torch.from_numpy(image.transpose(2, 0, 1))
        if mask is not None:
            mask = torch.from_numpy(mask).long()
            mask = mask.unsqueeze(dim=0)

        return image, mask
