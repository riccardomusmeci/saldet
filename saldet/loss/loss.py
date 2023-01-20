import torch.nn as nn
from saldet.loss.multi_bce import MultiBCELoss

FACTORY = {
    "multibce": MultiBCELoss,
    "bce": nn.BCELoss
}

def create_criterion(
    criterion: str,
    **kwargs
) -> nn.Module:
    return FACTORY[criterion](**kwargs)
    
