import torch
import torch.nn.functional as F

from saldet.utils import device


def flat(mask: torch.Tensor) -> torch.tensor:
    """Flat a mask

    Args:
        mask (torch.Tensor): input mask

    Returns:
        torch.tensor: flattened mask
    """
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask, size=(int(h), int(h)), mode="bilinear")
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1)
    # print(x.shape)  b 28*28 1
    if device() == "mps":
        x = x.float()
    g = x @ x.transpose(-2, -1)  # b 28*28 28*28
    g = g.unsqueeze(1)  # b 1 28*28 28*28
    return g


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor based on max and min of tensor

    Args:
        x (torch.Tensor): input tensor

    Returns:
        torch.Tensor: normalized tensor
    """
    t_max = torch.max(x)
    t_min = torch.min(x)
    out = (x - t_min) / (t_max - t_min)
    return out
