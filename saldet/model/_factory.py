import os

import torch
import torch.nn as nn

from saldet.pl import SaliencyPLModel
from saldet.utils import device

from .models import *

__all__ = ["create_model", "load_checkpoint"]

FACTORY = {"u2net_lite": U2NET_lite, "u2net_full": U2NET_full, "pgnet": PGNet}


def create_model(model_name: str, checkpoint_path: str = None, **kwargs) -> nn.Module:
    """Create a saliency model

    Args:
        model_name (str): model name
        checkpoint_path (str, optional): path to model checkpoint. Defaults to None.

    Returns:
        nn.Module: saliency model
    """

    if model_name in FACTORY.keys():
        model = FACTORY[model_name](**kwargs)
    else:
        print(f"> [ERROR] Model {model_name} not found.")

    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist."
        try:
            state_dict = torch.load(f=checkpoint_path, map_location=device())
        except Exception as e:
            print(f"Found an exception while loading state dict - {e}")
            print(f"Trying on cpu")
            state_dict = torch.load(f=checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict=state_dict)
        print(f"> Loaded state dict for {model_name}.")

    return model


def load_checkpoint(
    ckpt: str,
    model_name: str = None,
) -> nn.Module:
    """Load either a checkpoint with pytorch-lightning support or a pth file

    Args:
        ckpt (str): path to ckpt/pth file
        model_name (str, optional): model name. Defaults to None.

    Returns:
        nn.Module: model
    """
    if ckpt.endswith(".pth"):
        model = create_model(model_name=model_name, checkpoint_path=ckpt)
    elif ckpt.endswith(".ckpt"):
        try:
            model = SaliencyPLModel.load_from_checkpoint(
                checkpoint_path=ckpt, map_location=device()
            )
            if hasattr(model, "criterion") and hasattr(model, "model"):
                model = model.model
            print(f"> Loaded state dict from {ckpt}.")
        except Exception as e:
            print(f"> [ERROR] Not able to load ckpt from {ckpt} -  Exception: {e}")
    else:
        print(f"> [ERROR] {os.path.splitext(ckpt)[-1]} not supported.")
        quit()

    return model
