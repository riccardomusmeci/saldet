import torch

from saldet import create_model


def test_u2net_lite():
    input_size = 224
    model = create_model("u2net_lite", input_size=input_size)
    batch_size = 8
    x = torch.rand((batch_size, 3, input_size, input_size))
    out = model(x)
    assert (
        out[0].shape[0] == batch_size
    ), f"Num mask does not match batch_size {batch_size}"
    assert out[0].shape[1] == 1, f"Mask must be single-dim not {out.shape[1]-dim}"


def test_u2net_full():
    input_size = 300
    model = create_model("u2net_full", input_size=300)
    batch_size = 8
    x = torch.rand((batch_size, 3, input_size, input_size))
    out = model(x)
    assert (
        out[0].shape[0] == batch_size
    ), f"Num mask does not match batch_size {batch_size}"
    assert out[0].shape[1] == 1, f"Mask must be single-dim not {out.shape[1]-dim}"


def test_pgnet():
    input_size = 224
    model = create_model("pgnet", pretrained=True, input_size=input_size)
    batch_size = 8
    x = torch.rand((batch_size, 3, input_size, input_size))
    out = model(x)[0]
    assert (
        out.shape[0] == batch_size
    ), f"Num mask does not match batch_size {batch_size}"
    assert out.shape[1] == 1, f"Mask must be single-dim not {out.shape[1]-dim}"
