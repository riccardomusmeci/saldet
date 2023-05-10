import torch
import torch.nn as nn
from torch.nn import functional as F


class BCEIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        size = pred.size()[2:]
        mask = F.interpolate(mask, size=size, mode="bilinear")
        wbce = F.binary_cross_entropy_with_logits(pred, mask.float())
        pred = torch.sigmoid(pred)
        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()
