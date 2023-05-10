from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    StepLR,
    _LRScheduler,
)

__all__ = ["create_scheduler"]

FACTORY = {
    "cosine": CosineAnnealingLR,
    "cosine_restarts": CosineAnnealingWarmRestarts,
    "linear": LinearLR,
    "step": StepLR,
}


def create_scheduler(optimizer: Optimizer, name: str, **kwargs) -> _LRScheduler:
    name = name.lower()
    assert (
        name in FACTORY.keys()
    ), f"Only {list(FACTORY.keys())} lr_schedulers are supported. Change {name} to one of them."
    return FACTORY[name](optimizer, **kwargs)
