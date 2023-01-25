import os
import torch
from .models import *
import torch.nn as nn
from saldet.utils import device

FACTORY = {
    "u2net_lite": U2NET_lite,
    "u2net_full": U2NET_full,
    "pgnet": PGNet
}

def create_model(
    model_name: str,
    checkpoint_path: str = None,
    **kwargs
) -> nn.Module:
    
    if model_name in FACTORY.keys():
        model = FACTORY[model_name](**kwargs) 
    else:
        print(f"> [ERROR] Model {model_name} not found.")
    
    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist."
        try:
            state_dict = torch.load(f=checkpoint_path, map_location=device())
        except Exception as e:
            print(e)
            state_dict = torch.load(f=checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict=state_dict)
        print(f"> Loaded state dict for {model_name}.")
        
    return model
        

    