from typing import Iterable

from torch.optim import SGD, Adam, AdamW, Optimizer

__all__ = ["create_optimizer"]

FACTORY = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}


def create_optimizer(name: str, params: Iterable, **kwargs) -> Optimizer:
    """Return an optimizer

    Args:
        name (str): optimizer name
        params (Iterable): optimizer
        split_by (str, optional): splitting parameters into base and head based on str (only for PGNet). Defaults to None.

    Returns:
        Optimizer: optimizer

    """
    name = name.lower()
    assert (
        name in FACTORY.keys()
    ), f"Only {list(FACTORY.keys())} optimizers are supported. Change {name} to one of them."
    return FACTORY[name](params, **kwargs)
