import torch
import torch.nn as nn
from torch.nn import functional as F

from saldet.ops import flat


class AttentionGuidedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, mask: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> torch.Tensor:

        g = flat(mask)
        np4 = torch.sigmoid(p4.detach())
        np5 = torch.sigmoid(p5.detach())
        p4 = flat(np4)
        p5 = flat(np5)
        w1 = torch.abs(g - p4)
        w2 = torch.abs(g - p5)
        w = (w1 + w2) * 0.5 + 1
        attn_bce = F.binary_cross_entropy_with_logits(
            pred, g, weight=w * 1.0, reduction="mean"
        )
        return attn_bce
