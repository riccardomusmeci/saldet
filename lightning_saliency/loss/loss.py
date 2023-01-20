import torch.nn as nn
from lightning_saliency.loss.multi_bce import MultiBCELoss

FACTORY = {
    "multibce": MultiBCELoss,
    "bce": nn.BCELoss
}

def loss(
    criterion: str,
    **kwargs
) -> nn.Module:
    return FACTORY[criterion](**kwargs)
    
