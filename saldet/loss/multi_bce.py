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
        
        if not isinstance(preds, list) or not isinstance(preds, tuple):
            return self.bce_loss(preds, target)
        
        else:
            losses = []
            for pred in preds:
                losses.append(self.bce_loss(pred, target))
            losses = torch.tensor(losses)
            return torch.sum(losses)
        