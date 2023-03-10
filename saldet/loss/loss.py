import torch.nn as nn
from saldet.loss.multi_bce import MultiBCELoss
from saldet.loss.pg import PGLoss

FACTORY = {
    "multibce": MultiBCELoss,
    "bce": nn.BCELoss,
    "pg": PGLoss
}

def create_criterion(
    criterion: str,
    **kwargs
) -> nn.Module:
    return FACTORY[criterion](**kwargs)
    
