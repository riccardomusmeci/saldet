import os

import numpy as np
from PIL import Image


def read_binary(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("L")
    image = np.array(image, dtype=np.float32) / 255
    return image


def read_gray(file_path: str) -> Image:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("L")
    return image


def read_rgb(file_path: str) -> Image:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("RGB")
    image = np.array(image)
    return image
