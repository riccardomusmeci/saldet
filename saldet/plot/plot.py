import cv2
import matplotlib.pyplot as plt
import numpy as np

from saldet.utils import device


def prepare_mask(
    mask: np.array, width: int, height: int, threshold: float = 0.5
) -> np.array:
    """Prepare mask to save

    Args:
        mask (np.array): mask to save
        width (int): resize width
        height (int): resize height
        threshold (float, optional): mask threshold. Defaults to .5.

    Returns:
        np.array: prepared mask
    """
    if get_device() != "cpu":
        mask = mask.squeeze().cpu().numpy()
    else:
        mask = mask.squeeze().numpy()
    mask = cv2.resize(mask, (width, height))
    if threshold is not None:
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
    return mask


def apply_mask(image: np.array, mask: np.array, alpha: float = 1.0) -> np.array:
    """Apply segmentation masks to an image

    Args:
        image (np.array): image to be blended with the mask.
        mask (np.array): mask to be blended with the image.
        alpha (float): relative weight of the mask, 1.0 means no transparency, 0.0 means the mask is completely transparent.
            Defaults to 1.0.

    Returns:
        np.array: image with masks applied
    """
    out = image.copy()
    cmap = plt.cm.tab10
    for cat_id in np.unique(mask)[1:]:
        bool_mask = mask == cat_id
        color = np.array(cmap(cat_id - 1)[:3]) * 255
        cat_mask = (np.expand_dims(mask, axis=2) * color)[bool_mask].astype(np.uint8)
        out[bool_mask] = cv2.addWeighted(
            out[bool_mask], 1.0 - alpha, cat_mask, alpha, 0.0
        )

    return out


def plot_predictions(
    image: np.array, gt_mask: np.array, mask: np.array, alpha: float = 1.0
):
    """Plot ground truth and predicted mask on original image in one row.

    Args:
        image (np.array): original image
        gt_mask (np.array): ground truth image (can be None)
        mask (np.array): predicted mask
        alpha (float): relative weight of the mask, 1.0 means no transparency, 0.0 means the mask is completely transparent.
            Defaults to 1.0.
    """

    plt.figure(figsize=(32, 9))
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    if gt_mask is not None:
        gt_image = apply_mask(image=image, mask=gt_mask, alpha=alpha)
    else:
        gt_image = image

    plt.imshow(gt_image)

    plt.subplot(1, 3, 2)
    pred_image = apply_mask(image=image, mask=mask, alpha=alpha)
    plt.title("Model prediction")
    plt.imshow(pred_image)
    plt.show()
