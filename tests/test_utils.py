import torch

from saldet.utils import device, now


def test_device():
    _device = device()
    if torch.cuda.is_available():
        assert _device == "cuda", f"cuda available but device set to {_device}"
    if torch.has_mps:
        assert _device == "mps", f"mps available but device set to {_device}"


def test_time_format():
    t = now()
    assert isinstance(t, str), f"now() did not return a string but {type(t)}"
