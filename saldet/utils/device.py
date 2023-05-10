import torch


def device() -> str:
    """Return device type

    Returns:
        str: device type
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.has_mps:
        return "mps"
    return "cpu"
