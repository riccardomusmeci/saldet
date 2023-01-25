import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Union, Any

class MultiBCELoss(nn.Module):
    
    def __init__(
        self,
        weight: Tensor = None,
        size_average: bool = True,
        reduce: Any = None
    ) -> None:
        super().__init__()
        # self.bce_loss = nn.BCEWithLogitsLoss(
        #     weight=weight,
        #     size_average=size_average,
        #     reduce=reduce
        # )
        self.bce_loss = nn.BCELoss(
            weight=weight,
            size_average=size_average,
            reduce=reduce
        )
        
    def forward(
        self,
        preds: Union[List[Tensor], Tensor],
        target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """forward method

        Args:
            preds (Union[List[Tensor], Tensor]): single pred or list of preds
            target (Tensor): target mask

        Returns:
            Tuple[Tensor, Tensor]: total loss
        """
        
        if not hasattr(preds, "__iter__"):
            return self.bce_loss(preds, target)
        else:
            loss = None
            for pred in preds:
                if loss is None: loss = self.bce_loss(pred, target)
                else: loss += self.bce_loss(pred, target)
            return loss