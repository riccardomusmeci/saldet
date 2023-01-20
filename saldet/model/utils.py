from .models import *
import torch.nn as nn

FACTORY = {
    "u2net_lite": U2NET_lite,
    "u2net_full": U2NET_full    
}

def create_model(
    model_name: str,
    checkpoint_path: str = None,
    **kwargs
) -> nn.Module:
    
    if model_name in FACTORY.keys():
        return FACTORY[model_name](**kwargs)
    else:
        print(f"> [ERROR] Model {model_name} not found.")
    
    if checkpoint_path is not None:
        raise NotImplementedError

    