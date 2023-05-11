import os
from pathlib import Path

import cv2
import numpy as np


def read_rgb(file_path: Path) -> np.array:
    """Load an image from file_path as a numpy array

    Args:
        file_path (Path): path to image

    Raises:
        FileNotFoundError: if file is not found

    Returns:
        np.array: image
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Unable to read {file_path}.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_mask(file_path: Path) -> np.array:
    """Load a segmentation mask from file_path as a numpy array

    Args:
        file_path (Path): path to image

    Raises:
        FileNotFoundError: if file is not found

    Returns:
        np.array: mask
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read {file_path}.")

    img = img.astype("float64")
    if img.max() == 255:
        img /= 255
    return img


def save_image(image: np.array, output_path: str):
    """Save an image at given path making sure the folder exists

    Args:
        image (np.array): image to save
        output_path (str): output path
    """
    output_dir = output_path.replace(os.path.basename(output_path), "")
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        cv2.imwrite(output_path, image)
    except Exception as excp:
        print(
            f"[ERROR] While saving image at \
                path {output_path} found an error - {excp}"
        )
