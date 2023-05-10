from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F

from saldet.loss.attn_guided import AttentionGuidedLoss
from saldet.loss.bce_iou_loss import BCEIoULoss


class PGLoss(nn.Module):
    def __init__(self, w: float = 0.125) -> None:
        super().__init__()
        self.attn_loss = AttentionGuidedLoss()
        self.bce_iou_loss = BCEIoULoss()
        self.w = w

    def forward(self, preds: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:

        p1, wr, ws, attn_map = preds
        attn_loss = self.attn_loss(attn_map, target, wr, ws)  # attention guided loss
        loss1 = self.bce_iou_loss(p1, target)  # loss_b+i
        loss2 = (
            self.bce_iou_loss(wr, target) * self.w
            + self.bce_iou_loss(ws, target) * self.w
        )

        return loss1 + loss2 + attn_loss
